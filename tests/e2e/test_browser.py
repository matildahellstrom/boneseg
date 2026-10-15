"""Browser tests of the main user flows, on the demo stack with the fast classic backbone.

Run with: pip install playwright && python -m playwright install chromium && pytest tests/e2e
Skipped automatically when Playwright or its browser is missing.
"""
from __future__ import annotations

import glob
import socket
import threading
import time

import numpy as np
import pytest
import scipy.ndimage as ndi
import tifffile

playwright = pytest.importorskip("playwright.sync_api")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    import uvicorn

    from boneseg.api import create_app

    data = tmp_path_factory.mktemp("data")
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(create_app(data), host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", data
    srv.should_exit = True
    t.join(timeout=5)


@pytest.fixture
def page(server):
    try:
        pw = playwright.sync_playwright().start()
        browser = pw.chromium.launch()
    except Exception as e:  # Browser not installed
        pytest.skip(f"Chromium for Playwright is not available: {e}")
    pg = browser.new_page(viewport={"width": 1440, "height": 900})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.errors = errors
    yield pg
    browser.close()
    pw.stop()


def click_full(pg, y, x, shift=False):
    pos = pg.evaluate(f"(() => {{ const r = canvas.getBoundingClientRect(); const f = fullToDisp(); "
                      f"return [r.left + S.view.ox + {x} * f * S.view.scale, r.top + S.view.oy + {y} * f * S.view.scale]; }})()")
    if shift:
        pg.keyboard.down("Shift")
    pg.mouse.click(*pos)
    if shift:
        pg.keyboard.up("Shift")


def wait_result(pg):
    pg.wait_for_function("S.result && document.querySelector('#busy').classList.contains('hidden')", timeout=30000)
    pg.wait_for_timeout(300)


def open_demo(pg, url, data):
    pg.goto(url)
    pg.wait_for_function("S.health !== null", timeout=10000)
    pg.evaluate("api('/api/datasets/demo', {method: 'POST'}).then(d => refreshDatasets(d.id))")
    # The image shown must be the demo's own, not the one the page opened on start
    pg.wait_for_function("S.ds && S.base && S.ds.name === 'demo_bone_stack.tif' && S.base.src.includes(S.ds.id)", timeout=20000)
    pg.evaluate("S.backbone = 'classic'; rememberSettings()")   # Saved with the image, so a reload keeps it
    ds_id = pg.evaluate("S.ds.id")
    stack = tifffile.imread(glob.glob(str(data / "datasets" / ds_id / "*.tif"))[0])
    return stack


def click_cells(pg, stack, z=6, n=3):
    gt = stack[z, 2] > 0
    lab, k = ndi.label(gt)
    for cy, cx in ndi.center_of_mass(gt, lab, range(1, k + 1))[:n]:
        click_full(pg, cy, cx)
    for y, x in [(20, 20), (360, 40), (30, 480), (200, 15)]:
        click_full(pg, y, x, shift=True)


def test_click_segment_and_stack(page, server):
    url, data = server
    stack = open_demo(page, url, data)
    click_cells(page, stack)
    wait_result(page)
    assert page.inner_text("#nPos") == "3" and page.inner_text("#nNeg") == "4"
    assert "Dice" in page.inner_text("#evalBox")
    page.click("#stackBtn")
    page.wait_for_function("document.querySelector('#jobText').textContent.startsWith('Done')", timeout=60000)
    assert "12 slices" in page.inner_text("#jobText")
    assert not page.errors, page.errors


def test_correct_label_train_and_reload(page, server):
    url, data = server
    stack = open_demo(page, url, data)
    for z in (3, 8):
        page.evaluate(f"S.z = {z}; loadPlane()")
        page.wait_for_timeout(500)
        click_cells(page, stack, z)
        wait_result(page)
        page.keyboard.press("e")
        page.click("#editSave")
        page.wait_for_timeout(500)
    assert len(page.evaluate("S.labels")) == 2
    page.click("#trainBtn")
    page.wait_for_function("S.head !== null && S.method === 'learned'", timeout=30000)
    assert "held-out labelled slices" in page.inner_text("#trainResult")
    page.reload()
    page.wait_for_function("S.ds && S.base", timeout=20000)
    page.wait_for_timeout(500)
    assert page.evaluate("Object.keys(S.points).length") >= 2  # Clicks survived the reload
    assert not page.errors, page.errors


def test_names_from_files_cannot_inject_html(page, server):
    """A file name with markup in it must show as text and never run."""
    import io as _io

    import httpx

    url, data = server
    evil = '<img src=x onerror="window.__pwned = 1">'
    arr = _io.BytesIO()
    np.save(arr, np.random.default_rng(0).random((64, 64)).astype(np.float32))
    r = httpx.post(f"{url}/api/datasets/stream", params={"filename": evil + ".npy"}, content=arr.getvalue())
    assert r.status_code == 200, r.text
    open_demo(page, url, data)
    page.wait_for_timeout(800)
    assert page.evaluate("window.__pwned") is None
    assert evil + ".npy" in page.inner_text("#datasetList")
    assert not page.errors, page.errors


def test_minimal_interface(page, server, tmp_path):
    """The cut controls stay gone, and stack-only controls hide for a single image without a reference."""
    from PIL import Image as PILImage

    url, data = server
    open_demo(page, url, data)
    for gone in ["#backboneSelect", "#thrMode", "#showHeat", "#showSide", "#profileSelect", "#studyBtn", "#methodMicrosam",
                 "#finetuneBox", ".struct.add", "#segmentBtn"]:
        assert page.locator(gone).count() == 0, gone
    assert page.is_visible("#refToggle") and page.is_visible("#sliceNav") and page.is_visible("#stackSection")
    grey = tmp_path / "single.png"
    PILImage.fromarray((np.random.default_rng(1).random((120, 160)) * 255).astype(np.uint8)).save(grey)
    other = _open_path(page, grey)
    page.evaluate(f"openDataset('{other}')")
    page.wait_for_function(f"S.ds && S.ds.id === '{other}' && S.base", timeout=20000)
    assert page.is_hidden("#refToggle") and page.is_hidden("#sliceNav") and page.is_hidden("#stackSection")
    assert not page.errors, page.errors

def _open_path(pg, path):
    """Opens a file on disk through the API from inside the page, then selects it in the full app."""
    return pg.evaluate(f"api('/api/datasets/from-path', {{method: 'POST', body: {{path: {str(path)!r}}}}}).then(d => d.id)")


def test_bugfixes_full_app_clicks_undo_and_slices(page, server, tmp_path):
    from PIL import Image as PILImage

    url, data = server
    stack = open_demo(page, url, data)
    demo_id = page.evaluate("S.ds.id")
    # BUG-07: right-click removes a point, undo brings back exactly that point
    page.evaluate("S.z = 6; loadPlane()")
    page.wait_for_timeout(400)
    for y, x in [(20, 20), (360, 40), (30, 480)]:
        click_full(page, y, x, shift=True)
    pos = page.evaluate("(() => { const r = canvas.getBoundingClientRect(); const f = fullToDisp(); return [r.left + S.view.ox + 480 * f * S.view.scale, r.top + S.view.oy + 30 * f * S.view.scale]; })()")
    page.mouse.click(*pos, button="right")
    assert page.evaluate("pts().neg.length") == 2
    page.keyboard.press("Control+z")
    assert page.evaluate("JSON.stringify(pts().neg)") == "[[20,20],[360,40],[30,480]]"
    # BUG-01: a click followed at once by opening another image is still saved on the first image
    grey = tmp_path / "grey.png"
    PILImage.fromarray((np.random.default_rng(0).random((200, 300)) * 255).astype(np.uint8)).save(grey)
    other = _open_path(page, grey)
    click_full(page, 200, 15, shift=True)
    page.evaluate(f"openDataset('{other}')")
    page.wait_for_function(f"S.ds && S.ds.id === '{other}' && S.base", timeout=20000)
    page.wait_for_timeout(300)
    saved = page.evaluate(f"api('/api/datasets/{demo_id}/annotations')")
    assert len(saved["1:6"]["neg"]) == 4
    # BUG-11: no pixel size, so areas are in pixels
    for y, x in [(100, 100), (150, 200)]:
        click_full(page, y, x)
    click_full(page, 10, 10, shift=True)
    page.wait_for_function("S.result && document.querySelector('#busy').classList.contains('hidden')", timeout=30000)
    assert "px²" in page.inner_text("#statCards") and "µm²" not in page.inner_text("#statCards")
    # BUG-02: stepping quickly through slices ends with the image of the slice the app shows
    page.evaluate(f"openDataset('{demo_id}')")
    page.wait_for_function(f"S.ds && S.ds.id === '{demo_id}' && S.base", timeout=20000)
    page.evaluate("(async () => { for (const z of [7, 8, 7, 8, 9, 8]) { S.z = z; loadPlane(); await new Promise(r => setTimeout(r, 15)); } })()")
    page.wait_for_timeout(1500)
    assert page.evaluate("S.base.src.includes('z=' + S.z + '&')"), page.evaluate("[S.z, S.base.src]")
    # BUG-15: removing the open image leaves no stale panels and no errors
    page.on("dialog", lambda d: d.accept())
    page.locator(".dataset-item.active button").click()
    page.wait_for_function("S.ds === null", timeout=10000)
    assert page.is_hidden("#jobBox")
    assert not page.errors, page.errors


def test_bugfixes_reload_keeps_the_image(page, server):
    # BUG-08: a reload or a link with ?ds=&c=&z= opens the same image, channel and slice
    url, data = server
    open_demo(page, url, data)
    ds_id = page.evaluate("S.ds.id")
    page.evaluate("S.z = 7; loadPlane()")
    page.wait_for_timeout(400)
    page.reload()
    page.wait_for_function("S.ds && S.base", timeout=20000)
    assert page.evaluate("S.ds.id") == ds_id
    page.goto(f"{url}/?ds={ds_id}&c=1&z=4")
    page.wait_for_function("S.ds && S.base", timeout=20000)
    assert page.evaluate("[S.ds.id, S.c, S.z]") == [ds_id, 1, 4]
    assert not page.errors, page.errors

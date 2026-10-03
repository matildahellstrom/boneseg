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
    pg.wait_for_function("S.ds && S.base && S.ds.name === 'demo_bone_stack.tif'", timeout=20000)
    pg.select_option("#backboneSelect", "classic")
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
    assert "Estimated Dice" in page.inner_text("#trainResult")
    page.reload()
    page.wait_for_function("S.ds && S.base", timeout=20000)
    page.wait_for_timeout(500)
    assert page.evaluate("Object.keys(S.points).length") >= 2  # Clicks survived the reload
    assert not page.errors, page.errors


def test_region_and_histomorphometry(page, server):
    url, data = server
    stack = open_demo(page, url, data)
    # Bone on channel 0
    page.select_option("#channelSelect", "0")
    page.wait_for_timeout(700)
    m = ndi.gaussian_filter(stack[6, 0].astype(float), 3)
    rng = np.random.default_rng(0)
    hi, lo = np.argwhere(m > np.percentile(m, 85)), np.argwhere(m < np.percentile(m, 15))
    for y, x in hi[rng.choice(len(hi), 4, replace=False)]:
        click_full(page, y, x)
    for y, x in lo[rng.choice(len(lo), 4, replace=False)]:
        click_full(page, y, x, shift=True)
    wait_result(page)
    # Cells on channel 1
    page.select_option("#channelSelect", "1")
    page.wait_for_timeout(700)
    click_cells(page, stack)
    wait_result(page)
    page.keyboard.press("r")
    for y, x in [(30, 30), (30, 480), (350, 480), (350, 30)]:
        click_full(page, y, x)
    page.keyboard.press("Enter")
    page.wait_for_function("S.ds.roi && S.ds.roi.length === 4", timeout=10000)
    wait_result(page)
    page.click("#histoBtn")
    page.wait_for_function("!document.querySelector('#histoCards').classList.contains('hidden')", timeout=30000)
    assert "B.Ar/T.Ar" in page.inner_text("#histoCards")
    assert not page.errors, page.errors


def test_two_structures(page, server):
    url, data = server
    stack = open_demo(page, url, data)
    page.on("dialog", lambda d: d.accept("Bone matrix"))
    click_cells(page, stack)
    wait_result(page)
    page.click(".struct.add")
    page.wait_for_function("S.structures.length === 2", timeout=5000)
    assert page.evaluate("S.structures.map(s => s.name)") == ["Object", "Bone matrix"]
    bone = ndi.gaussian_filter(stack[6, 0].astype(float), 3)
    rng = np.random.default_rng(0)
    hi = np.argwhere((bone > np.percentile(bone, 85)) & ~ndi.binary_dilation(stack[6, 2] > 0, iterations=10))
    for y, x in hi[rng.choice(len(hi), 4, replace=False)]:
        click_full(page, y, x)
    page.wait_for_function("S.result && S.result.multi && document.querySelector('#busy').classList.contains('hidden')", timeout=30000)
    cards = page.inner_text("#statCards")
    assert "Object" in cards and "Bone matrix" in cards
    assert page.is_visible("#labelsExport") and not page.is_visible("#editBtn")
    page.reload()
    page.wait_for_function("S.ds && S.base", timeout=20000)
    page.wait_for_timeout(500)
    assert page.evaluate("S.structures.map(s => s.name)") == ["Object", "Bone matrix"]
    assert not page.errors, page.errors


def test_names_from_files_and_profiles_cannot_inject_html(page, server):
    """A shared profile or a channel name with markup in it must show as text and never run."""
    import io as _io

    import httpx

    from boneseg.segment import Profile

    url, data = server
    evil = '<img src=x onerror="window.__pwned = 1">'
    buf = _io.BytesIO()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
        Profile(name=evil, backbone="classic", layer_from_end=1, pos=np.zeros((1, 64), np.float32), neg=np.zeros((0, 64), np.float32),
                settings={}, description=evil, source=evil).save(tmp.name)
        buf.write(open(tmp.name, "rb").read())
    r = httpx.post(f"{url}/api/profiles/import", files={"file": ("p.npz", buf.getvalue(), "application/octet-stream")})
    assert r.status_code == 200
    arr = _io.BytesIO()
    np.save(arr, np.random.default_rng(0).random((64, 64)).astype(np.float32))
    r = httpx.post(f"{url}/api/datasets/stream", params={"filename": evil + ".npy"}, content=arr.getvalue())
    assert r.status_code == 200, r.text
    open_demo(page, url, data)
    ds_id = page.evaluate("S.ds.id")
    httpx.patch(f"{url}/api/datasets/{ds_id}", json={"group": evil})
    page.reload()
    page.wait_for_function("S.ds && S.base", timeout=20000)
    page.click("#studyBtn")
    page.wait_for_timeout(800)
    assert page.evaluate("window.__pwned") is None
    assert evil in page.inner_text("#profileSelect")
    assert evil + ".npy" in page.inner_text("#datasetList")
    assert not page.errors, page.errors


def test_stack_with_two_structures_in_browser(page, server):
    url, data = server
    stack = open_demo(page, url, data)
    page.on("dialog", lambda d: d.accept("Bone matrix"))
    click_cells(page, stack)
    wait_result(page)
    page.click(".struct.add")
    page.wait_for_function("S.structures.length === 2", timeout=5000)
    bone = ndi.gaussian_filter(stack[6, 0].astype(float), 3)
    rng = np.random.default_rng(1)
    hi = np.argwhere((bone > np.percentile(bone, 85)) & ~ndi.binary_dilation(stack[6, 2] > 0, iterations=10))
    for y, x in hi[rng.choice(len(hi), 4, replace=False)]:
        click_full(page, y, x)
    page.wait_for_function("S.result && S.result.multi && document.querySelector('#busy').classList.contains('hidden')", timeout=30000)
    page.click("#stackBtn")
    page.wait_for_function("document.querySelector('#jobText').textContent.startsWith('Done')", timeout=60000)
    text = page.inner_text("#jobText")
    assert "Object" in text and "Bone matrix" in text and "objects in 3D" in text
    assert "labels.tif" in page.inner_text("#jobDownloads")
    assert not page.errors, page.errors


def test_multi_structure_profile_in_browser(page, server):
    url, data = server
    stack = open_demo(page, url, data)
    page.on("dialog", lambda d: d.accept("Bone matrix"))
    click_cells(page, stack)
    wait_result(page)
    page.click(".struct.add")
    page.wait_for_function("S.structures.length === 2", timeout=5000)
    bone = ndi.gaussian_filter(stack[6, 0].astype(float), 3)
    rng = np.random.default_rng(2)
    hi = np.argwhere((bone > np.percentile(bone, 85)) & ~ndi.binary_dilation(stack[6, 2] > 0, iterations=10))
    for y, x in hi[rng.choice(len(hi), 4, replace=False)]:
        click_full(page, y, x)
    page.wait_for_function("S.result && S.result.multi && document.querySelector('#busy').classList.contains('hidden')", timeout=30000)
    page.click("#saveProfileBtn")
    page.fill("#profileName", "Cells and bone")
    page.click("#profileSaveConfirm")
    page.wait_for_function("S.profiles.some(p => p.kind === 'structures')", timeout=10000)
    # A fresh dataset, no clicks: picking the profile brings both structures
    page.evaluate("api('/api/datasets/demo', {method: 'POST'}).then(d => refreshDatasets(d.id))")
    page.wait_for_function("S.ds && S.base && Object.keys(S.points).length === 0", timeout=20000)
    page.select_option("#backboneSelect", "classic")
    pid = page.evaluate("S.profiles.find(p => p.kind === 'structures').id")
    page.select_option("#profileSelect", pid)
    page.wait_for_function("S.result && S.result.multi && document.querySelector('#busy').classList.contains('hidden')", timeout=30000)
    assert page.evaluate("S.structures.map(s => s.name)") == ["Object", "Bone matrix"]
    cards = page.inner_text("#statCards")
    assert "Object" in cards and "Bone matrix" in cards
    assert not page.errors, page.errors

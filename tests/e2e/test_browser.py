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
    assert "held-out labelled slices" in page.inner_text("#trainResult")
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
    assert page.is_visible("#labelsExport") and page.is_visible("#editBtn")
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


def test_batch_from_compare_dialog(page, server):
    url, data = server
    stack = open_demo(page, url, data)
    click_cells(page, stack)
    wait_result(page)
    page.click("#saveProfileBtn")
    page.fill("#profileName", "Batch cells")
    page.click("#profileSaveConfirm")
    page.wait_for_function("S.profiles.some(p => p.name === 'Batch cells')", timeout=10000)
    page.click("#studyBtn")
    page.click("#batchBox summary")
    pid = page.evaluate("S.profiles.find(p => p.name === 'Batch cells').id")
    page.select_option("#batchProfile", pid)
    page.fill("#batchStep", "4")
    page.click("#batchRun")
    page.wait_for_function("document.querySelector('#batchText').textContent.startsWith('Done')", timeout=120000)
    text = page.inner_text("#batchText")
    assert "failed" not in text
    # Samples without the channel (a one-channel file from another test) are skipped and keep no stack run
    import re
    m = re.search(r"(\d+) skipped", text)
    assert page.inner_text("#studyRows").count("no stack run") == (int(m.group(1)) if m else 0)
    assert not page.errors, page.errors



def test_teach_several_structures(page, server):
    url, data = server
    stack = open_demo(page, url, data)
    page.on("dialog", lambda d: d.accept("Bone matrix"))
    for i, z in enumerate((3, 8)):
        page.evaluate(f"S.z = {z}; loadPlane()")
        page.wait_for_timeout(500)
        if i == 0:
            click_cells(page, stack, z)
            wait_result(page)
            page.click(".struct.add")
            page.wait_for_function("S.structures.length === 2", timeout=5000)
        else:
            page.evaluate("S.active = 0; renderStructures()")
            click_cells(page, stack, z)
            page.evaluate("S.active = 1; renderStructures()")
        bone = ndi.gaussian_filter(stack[z, 0].astype(float), 3)
        rng = np.random.default_rng(z)
        hi = np.argwhere((bone > np.percentile(bone, 85)) & ~ndi.binary_dilation(stack[z, 2] > 0, iterations=10))
        for y, x in hi[rng.choice(len(hi), 4, replace=False)]:
            click_full(page, y, x)
        page.wait_for_function("S.result && S.result.multi && document.querySelector('#busy').classList.contains('hidden')", timeout=30000)
        # Correct with the brush in the active structure's colour, then save as a label
        page.keyboard.press("e")
        r = page.evaluate("canvas.getBoundingClientRect().toJSON()")
        page.mouse.move(r["x"] + r["width"] / 2 - 30, r["y"] + r["height"] / 2)
        page.mouse.down()
        page.mouse.move(r["x"] + r["width"] / 2 + 30, r["y"] + r["height"] / 2, steps=4)
        page.mouse.up()
        page.click("#editSave")
        page.wait_for_function(f"S.labels.some(l => l.z === {z})", timeout=10000)
    page.click("#trainBtn")
    page.wait_for_function("S.head && S.head.names && S.head.names.length === 2 && S.method === 'learned'", timeout=60000)
    page.evaluate("S.z = 6; loadPlane()")
    page.wait_for_function("S.result && S.result.multi && S.z === 6 && document.querySelector('#busy').classList.contains('hidden')", timeout=30000)
    cards = page.inner_text("#statCards")
    assert "Object" in cards and "Bone matrix" in cards
    assert not page.errors, page.errors


def test_histomorphometry_whole_stack(page, server):
    url, data = server
    stack = open_demo(page, url, data)
    page.select_option("#channelSelect", "0")
    page.wait_for_timeout(700)
    m = ndi.gaussian_filter(stack[6, 0].astype(float), 3)
    rng = np.random.default_rng(5)
    hi, lo = np.argwhere(m > np.percentile(m, 85)), np.argwhere(m < np.percentile(m, 15))
    for y, x in hi[rng.choice(len(hi), 4, replace=False)]:
        click_full(page, y, x)
    for y, x in lo[rng.choice(len(lo), 4, replace=False)]:
        click_full(page, y, x, shift=True)
    wait_result(page)
    page.click("#saveProfileBtn")
    page.fill("#profileName", "Demo bone")
    page.click("#profileSaveConfirm")
    page.wait_for_function("S.profiles.some(p => p.name === 'Demo bone')", timeout=10000)
    pid = page.evaluate("S.profiles.find(p => p.name === 'Demo bone').id")
    page.select_option("#hBoneC", "0")
    page.select_option("#hBoneSrc", f"profile:{pid}")
    page.select_option("#hCellC", "1")
    page.select_option("#hCellSrc", "reference")
    page.fill("#zStep", "3")
    page.click("#histoStackBtn")
    page.wait_for_function("document.querySelector('#histoStackText').textContent.includes('slices:')", timeout=60000)
    assert "Oc.Pm/B.Pm" in page.inner_text("#histoStackText")
    assert not page.errors, page.errors


def test_sam_setting_and_finetune_panel(page, server, monkeypatch):
    import boneseg.finetune as fmod

    class Fake:
        info = {"best_val_dice": 0.8, "best_step": 50}

        def save(self, path):
            path.write_bytes(b"x")

    def fake_finetune(train, val, train_blocks, steps, progress, cancelled):
        progress(1.0, "done")
        return Fake()

    monkeypatch.setattr(fmod, "finetune", fake_finetune)
    url, data = server
    open_demo(page, url, data)
    page.click("text=Clean-up and advanced")
    page.select_option("#samRefine", "agree")
    assert page.is_visible("#samHint") and page.evaluate("settings().sam_refine") == "agree"
    page.select_option("#samRefine", "off")
    page.evaluate("api(`/api/datasets/${S.ds.id}/labels/from-reference`, {method: 'POST', body: {channel: S.c, n: 3}}).then(() => refreshLabels())")
    page.wait_for_function("S.labels.length === 3", timeout=20000)
    page.click("text=Fine-tune DINOv2 on these labels")
    page.click("#finetuneBtn")
    page.wait_for_selector("#useFinetuned", timeout=60000)
    assert "fine-tuned 0.800" in page.inner_text("#finetuneText")
    assert page.evaluate("[...document.querySelectorAll('#backboneSelect option')].some(o => o.value.includes('dinov2_s14_demo'))")
    assert not page.errors, page.errors


def test_simple_mode(page, server):
    import urllib.request

    url, data = server
    page.goto(url + "/simple")
    page.wait_for_selector("#demoBtn")
    page.click("#demoBtn")
    page.wait_for_function("P.ds && P.img && P.ds.name === 'demo_bone_stack.tif'", timeout=20000)
    ds_id = page.evaluate("P.ds.id")
    stack = tifffile.imread(glob.glob(str(data / "datasets" / ds_id / "*.tif"))[0])
    page.locator("#stage").scroll_into_view_if_needed()   # Mouse clicks only land inside the visible window
    page.evaluate("P.z = 6; document.getElementById('slice').value = 6; loadSlice(true)")
    page.wait_for_timeout(500)
    assert page.is_hidden("#drop") and "demo_bone_stack" in page.inner_text("#openedName")   # Step 1 collapsed

    def click(y, x, button="left"):
        pos = page.evaluate(f"(() => {{ const r = canvas.getBoundingClientRect(); const f = fullPerImg(); "
                            f"return [r.left + (P.view.ox + {x} / f * P.view.scale) / devicePixelRatio, r.top + (P.view.oy + {y} / f * P.view.scale) / devicePixelRatio]; }})()")
        page.mouse.click(*pos, button=button)

    gt = stack[6, 2] > 0
    lab, k = ndi.label(gt)
    for cy, cx in ndi.center_of_mass(gt, lab, range(1, k + 1))[:3]:
        click(cy, cx)
    page.keyboard.press("2")
    for y, x in [(20, 20), (360, 40), (30, 480), (200, 15)]:
        click(y, x)
    page.wait_for_function("P.mask !== null", timeout=30000)
    assert "Bone area" in page.inner_text("#numbers") and "mask is wrong" in page.inner_text("#hint")
    # Right-click removes the nearest click; Ctrl+Z brings exactly that click back; the next Ctrl+Z undoes the last added
    count = lambda: page.evaluate("[P.clicks['1:6'].pos.length, P.clicks['1:6'].neg.length]")  # noqa: E731
    click(200, 15, "right")
    assert count() == [3, 3]
    page.keyboard.press("Control+z")
    assert count() == [3, 4] and page.evaluate("P.clicks['1:6'].neg.some(([y, x]) => Math.abs(y - 200) < 3 && Math.abs(x - 15) < 3)")
    page.keyboard.press("Control+z")
    page.wait_for_timeout(300)
    assert count() == [3, 3]
    # Another slice is segmented with the clicked slice's clicks
    page.evaluate("P.z = 3; document.getElementById('slice').value = 3; loadSlice()")
    page.wait_for_function("P.mask !== null && document.getElementById('numbers').textContent.includes('slice 7')", timeout=30000)
    # Whole stack, then both downloads work
    page.click("#stackBtn")
    page.wait_for_selector("#downloads a", timeout=60000)
    assert "12" in page.inner_text("#stackNumbers")
    for href in page.eval_on_selector_all("#downloads a", "as => as.map(a => a.href)"):
        assert urllib.request.urlopen(href).status == 200
    # The full app finds the same clicks; the phone layout has no sideways scrolling
    page.set_viewport_size({"width": 390, "height": 800})
    assert not page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
    page.set_viewport_size({"width": 1440, "height": 900})
    page.wait_for_timeout(600)   # The clicks are saved 400 ms after the last change
    page.goto(url + "/")
    page.wait_for_function("S.ds && S.base", timeout=20000)
    assert page.evaluate("S.points['1:6'].pos.length") == 3
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
    assert page.is_hidden("#jobBox") and page.is_hidden("#sidePanel")
    page.evaluate("document.getElementById('sideImg').parentElement.click()")
    assert not page.errors, page.errors


def test_bugfixes_switching_pages_keeps_the_image(page, server):
    url, data = server
    page.goto(url + "/simple")
    page.click("#demoBtn")
    page.wait_for_function("P.ds && P.img", timeout=20000)
    ds_id = page.evaluate("P.ds.id")
    page.evaluate("P.z = 4; document.getElementById('slice').value = 4; loadSlice()")
    page.wait_for_timeout(400)
    # BUG-08: "Full app" opens the same image, channel and slice
    page.click("#fullLink")
    page.wait_for_function("S.ds && S.base", timeout=20000)
    assert page.evaluate("[S.ds.id, S.c, S.z]") == [ds_id, page.evaluate("S.c"), 4] and page.evaluate("S.ds.id") == ds_id
    # And back: simple mode opens it too, also on a plain reload
    page.evaluate("S.z = 7; loadPlane()")
    page.wait_for_timeout(300)
    page.click("#simpleLink")
    page.wait_for_function("P.ds && P.img", timeout=20000)
    assert page.evaluate("[P.ds.id, P.z]") == [ds_id, 7]
    page.goto(url + "/simple")
    page.wait_for_function("P.ds && P.img", timeout=20000)
    assert page.evaluate("P.ds.id") == ds_id
    assert not page.errors, page.errors

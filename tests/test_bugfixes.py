"""Regression tests for the bugs found by the bug-hunting run of 10 October 2026 (numbers as in its report).
Front-end bugs (lost clicks, undo, slice races, page switching) are tested in tests/e2e/test_browser.py."""
import io
import time

import numpy as np
import pytest
import torch
from PIL import Image

from tests.test_api import SETTINGS, bg_points, client, upload_stack  # noqa: F401  (client is a fixture)


def wait(client, job, statuses=("done", "failed", "cancelled")):
    for _ in range(400):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job still {job['status']}")


def test_bug03_region_after_histomorphometry(client):
    # A histomorphometry table among the results used to break drawing a region (HTTP 500)
    did = client.post("/api/datasets/demo").json()["id"]
    s = {"backbone": "classic", "vit_size": 252}
    for c in (0, 1):
        assert client.post(f"/api/datasets/{did}/segment", json={"channel": c, "z": 6, "pos": [[100, 100]], "neg": [[10, 10]], "settings": s}).status_code == 200
    assert client.post(f"/api/datasets/{did}/histomorphometry", json={"z": 6, "bone": {"channel": 0}, "cells": {"channel": 1}, "settings": s}).status_code == 200
    assert client.put(f"/api/datasets/{did}/roi", json={"polygon": [[0, 0], [0, 200], [200, 200], [200, 0]]}).status_code == 200
    assert client.put(f"/api/datasets/{did}/roi", json={"polygon": None}).status_code == 200


def test_bug09_cancelled_work_is_not_done(tmp_path):
    from boneseg.store import Store

    store = Store(tmp_path)
    assert store.record_job("stack", {}, lambda d: {"x": 1}, cancelled=lambda: True).status == "cancelled"
    assert store.record_job("stack", {}, lambda d: {"x": 1}, cancelled=lambda: False).status == "done"


def test_bug13_batch_runs_single_channel_files(client):
    a, gt, centers = upload_stack(client, n_z=1, with_reference=False)   # One channel, never chosen in the app
    prof = client.post("/api/profiles", json={"name": "p", "dataset_id": a["id"], "channel": 0, "z": 0, "pos": [list(x) for x in centers],
                                              "neg": bg_points(gt), "settings": SETTINGS}).json()
    job = wait(client, client.post("/api/batch", json={"profile_id": prof["id"], "channel": None, "dataset_ids": [a["id"]], "settings": SETTINGS}).json())
    assert [s["status"] for s in job["result"]["samples"]] == ["done"]


def test_bug14_running_jobs_listed_and_one_stack_run_at_a_time(client, monkeypatch):
    import boneseg.api.segmentation as seg

    ds, gt, centers = upload_stack(client, n_z=4)
    did = ds["id"]
    gate = {"open": False}
    real = seg.run_stack

    def slow(*a, **k):   # Holds the job open until the test lets it finish
        while not gate["open"]:
            time.sleep(0.02)
        return real(*a, **k)
    monkeypatch.setattr(seg, "run_stack", slow)
    body = {"channel": 0, "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS}
    job = client.post(f"/api/datasets/{did}/stack", json=body).json()
    running = client.get(f"/api/datasets/{did}/jobs", params={"status": "running"}).json()
    assert [j["id"] for j in running] == [job["id"]]
    assert client.post(f"/api/datasets/{did}/stack", json=body).status_code == 409
    gate["open"] = True
    assert wait(client, job)["status"] == "done"
    assert client.get(f"/api/datasets/{did}/jobs", params={"status": "running"}).json() == []


def test_bug16_labels_inside_a_region_ignore_the_outside(client):
    ds, gt, centers = upload_stack(client, n_z=3)
    did = ds["id"]
    h, w = gt.shape
    client.put(f"/api/datasets/{did}/roi", json={"polygon": [[0, 0], [0, w // 2], [h, w // 2], [h, 0]]})
    body = {"channel": 0, "z": 1, "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS}
    client.post(f"/api/datasets/{did}/segment", json=body)
    client.post(f"/api/datasets/{did}/labels", json={"channel": 0, "z": 1})
    store = client.app.state.store
    region = store.label_region(did, 0, 1)
    assert region is not None and region[:, : w // 2 - 2].all() and not region[:, w // 2 + 2:].any()
    store.delete_label(did, 0, 1)
    assert store.label_region(did, 0, 1) is None


def test_bug16_training_skips_patches_outside_the_valid_region():
    from boneseg.head import fit
    from boneseg.segment import Embedding

    g = torch.nn.functional.normalize(torch.rand(10, 10, 8), dim=-1)
    emb = Embedding(g, 140, 140)
    mask = np.zeros((140, 140), bool)
    valid = np.zeros((140, 140), bool)
    valid[:, :70] = True                 # Only the left half is labelled
    mask[20:50, 10:40] = True
    # With a valid region only the left half's 50 patches are used; the result must still train
    model, dim = fit([(emb, mask, valid)], "linear", epochs=5)
    assert dim == 8 and model(torch.zeros(1, 8)).shape == (1, 1)


def test_bug19_inputs_are_checked(client):
    ds, gt, centers = upload_stack(client, n_z=1)
    did = ds["id"]
    seg = lambda st: client.post(f"/api/datasets/{did}/segment", json={"channel": 0, "z": 0, "pos": [list(centers[0])], "neg": bg_points(gt), "settings": st})  # noqa: E731
    for bad in ({"vit_size": "abc"}, {"neg_weight": "x"}, {"vit_size": 0}, {"shift_passes": 0}, {"refiner": "nope.pt"}, {"clip_low": 99, "clip_high": 1}):
        r = seg({**SETTINGS, **bad})
        assert r.status_code == 400 and r.headers["content-type"].startswith("application/json"), (bad, r.text)
    assert seg({**SETTINGS, "vit_size": "252"}).status_code == 200   # A number given as text is fine
    assert client.put(f"/api/datasets/{did}/annotations", json={"channel": 0, "z": 99, "pos": [[1, 1]], "neg": []}).status_code == 400
    assert client.put(f"/api/datasets/{did}/annotations", json={"channel": 7, "z": 0, "pos": [[1, 1]], "neg": []}).status_code == 400
    assert client.post(f"/api/datasets/{did}/labels", json={"channel": 0, "z": 42}).status_code == 400
    r = client.post(f"/api/datasets/{did}/labels", json={"channel": 0, "z": 0, "mask_png": "data:image/png;base64,bm90IGEgcG5n"})
    assert r.status_code == 400 and "PNG" in r.json()["detail"]
    assert client.put(f"/api/datasets/{did}/roi", content='{"polygon": [[0, 0], [NaN, 5], [9, 9]]}',
                      headers={"content-type": "application/json"}).status_code in (400, 422)


def test_bug20_downloads_are_named_after_the_image(client):
    ds, gt, centers = upload_stack(client, n_z=2)
    job = client.post(f"/api/datasets/{ds['id']}/stack", json={"channel": 0, "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS}).json()
    wait(client, job)
    r = client.get(f"/api/jobs/{job['id']}/files/slices.csv")
    assert r.status_code == 200 and 'filename="stack_slices.csv"' in r.headers["content-disposition"]


def test_bug21_errors_without_server_paths(client):
    r = client.post("/api/datasets", files={"file": ("broken.png", b"not really a png", "image/png")})
    assert r.status_code == 400 and "broken.png" in r.json()["detail"]
    assert str(client.app.state.store.root) not in r.json()["detail"]
    r = client.post("/api/datasets/from-path", json={"path": "  "})
    assert r.status_code == 400 and "full path" in r.json()["detail"]


def test_bug22_training_as_a_background_job(client):
    ds, gt, centers = upload_stack(client, n_z=4)
    did = ds["id"]
    client.patch(f"/api/datasets/{did}", json={"reference_channel": 1})
    client.post(f"/api/datasets/{did}/labels/from-reference", json={"channel": 0, "n": 2})
    job = client.post(f"/api/datasets/{did}/head", json={"channel": 0, "settings": SETTINGS, "background": True}).json()
    assert job["kind"] == "train"
    job = wait(client, job)
    assert job["status"] == "done" and job["result"]["kind"] in ("linear", "mlp") and "threshold" in job["result"]
    assert client.get(f"/api/datasets/{did}/head", params={"c": 0}).json() is not None


def test_bug23_finetune_cancel_before_the_first_check(monkeypatch):
    import boneseg.finetune as fmod

    class Tiny(torch.nn.Module):   # A stand-in for DINOv2 with the methods SegNet uses
        patch_size, embed_dim = 14, 4

        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
            self.norm = torch.nn.LayerNorm(4)
            self.proj = torch.nn.Conv2d(3, 4, 14, 14)

        def get_intermediate_layers(self, x, n=1, reshape=True, norm=True):
            return [self.proj(x)]

    img = np.random.default_rng(0).random((56, 56)).astype(np.float32)
    gt = img > 0.5
    steps = []
    ft = fmod.finetune([(img, gt)], [(img, gt)], train_blocks=1, steps=500, crop=56, batch=1, pretrained=Tiny(), device=torch.device("cpu"),
                       progress=lambda p, m: steps.append(p), cancelled=lambda: len(steps) >= 3)
    assert ft.info["best_step"] <= 5 and len(steps) <= 5   # Stopped right after cancelling, not at step 100

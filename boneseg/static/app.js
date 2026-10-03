"use strict";
// boneseg front end. Plain JavaScript, no build step.

const $ = (id) => document.getElementById(id);
const S = {
  health: null,
  datasets: [],
  ds: null,
  c: 0,
  z: 0,
  low: 1,
  high: 99.5,
  points: {},          // "c:z" -> {pos: [[y, x]], neg: [[y, x]]} in full-resolution pixels
  history: [],         // Undo stack of {key, kind}
  mode: "pos",
  profiles: [],
  base: null,          // HTMLImageElement of the current plane at display resolution
  layers: {},          // mask, heat, unc, ref -> HTMLImageElement
  result: null,
  view: { scale: 1, ox: 0, oy: 0 },
  seq: 0,
  job: null,
  space: false,
};

// ---------------------------------------------------------------------------------------------
// Small helpers
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {},
    ...opts,
    body: opts.body && !(opts.body instanceof FormData) ? JSON.stringify(opts.body) : opts.body,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) { /* not JSON */ }
    throw new Error(msg);
  }
  return res.headers.get("content-type")?.includes("json") ? res.json() : res;
}

let toastTimer = null;
function toast(msg, isError = false) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.toggle("error", isError);
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), isError ? 6000 : 3000);
}

function status(msg) { $("statusbar").textContent = msg; }
function busy(on, text = "Working…") { $("busy").classList.toggle("hidden", !on); $("busyText").textContent = text; }
const key = () => `${S.c}:${S.z}`;
const pts = () => (S.points[key()] ||= { pos: [], neg: [] });

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not load image"));
    img.src = src;
  });
}

function fmt(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (a >= 1e4) return (v / 1e3).toFixed(1) + "k";
  return v.toFixed(digits);
}

// ---------------------------------------------------------------------------------------------
// Settings
function settings() {
  return {
    backbone: $("backboneSelect").value,
    threshold_mode: $("thrMode").value,
    top_percent: +$("topPercent").value,
    manual_threshold: +$("manualThr").value,
    neg_weight: +$("lambda").value,
    min_object_um2: +$("minObj").value,
    fill_holes_um2: +$("fillHoles").value,
    smooth_px: +$("smooth").value,
    vit_size: +$("vitSize").value,
    layer_from_end: +$("layer").value,
    clip_low: S.low,
    clip_high: S.high,
  };
}

function syncSettingLabels() {
  const mode = $("thrMode").value;
  $("topPercentRow").classList.toggle("hidden", mode !== "top_percent");
  $("manualRow").classList.toggle("hidden", mode !== "manual");
  $("thrHint").classList.toggle("hidden", mode !== "clicks");
  $("topPercentValue").textContent = `${$("topPercent").value}%`;
  $("manualValue").textContent = (+$("manualThr").value).toFixed(2);
  $("lambdaValue").textContent = (+$("lambda").value).toFixed(2);
  $("minObjValue").textContent = `${$("minObj").value} µm²`;
  $("fillValue").textContent = `${$("fillHoles").value} µm²`;
  $("smoothValue").textContent = `${$("smooth").value} px`;
  const bb = S.health?.backbones.find((b) => b.id === $("backboneSelect").value);
  $("backboneHint").textContent = !bb ? "" : bb.id === "classic"
    ? "Hand-made intensity and texture features. Instant, but less accurate than DINOv2."
    : bb.ready ? "Weights are downloaded. The first click on each slice computes features, later clicks are instant."
      : "The weights download on first use, which can take a minute.";
}

// ---------------------------------------------------------------------------------------------
// Datasets
async function refreshDatasets(selectId) {
  S.datasets = await api("/api/datasets");
  const list = $("datasetList");
  list.innerHTML = "";
  for (const d of S.datasets) {
    const el = document.createElement("div");
    el.className = "dataset-item" + (S.ds?.id === d.id ? " active" : "");
    el.innerHTML = `<span class="name" title="${d.name}">${d.name}</span><span class="small muted">${d.n_z}z·${d.n_channels}c</span><button title="Remove from the app">✕</button>`;
    el.onclick = () => openDataset(d.id);
    el.querySelector("button").onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Remove ${d.name} from the app? Uploaded copies are deleted, files opened from disk are not.`)) return;
      await api(`/api/datasets/${d.id}`, { method: "DELETE" });
      if (S.ds?.id === d.id) { S.ds = null; showEmpty(true); }
      refreshDatasets();
    };
    list.appendChild(el);
  }
  if (selectId) openDataset(selectId);
}

function showEmpty(on) {
  $("emptyState").classList.toggle("hidden", !on);
  document.querySelectorAll(".disabled-until-data").forEach((el) => el.classList.toggle("locked", on));
  if (on) {
    $("datasetTitle").textContent = "No image loaded";
    $("resultsSection").classList.add("hidden");
    S.base = null;
    draw();
  }
}

async function openDataset(id) {
  const d = await api(`/api/datasets/${id}`);
  S.ds = d;
  S.points = {};
  S.history = [];
  S.result = null;
  S.c = 0;
  S.z = Math.floor(d.n_z / 2);
  $("datasetTitle").textContent = `${d.name} · ${d.width}×${d.height} px · ${d.n_z} slices · ${d.voxel_size_known ? `${d.voxel_um[2].toFixed(3)} µm/px` : "pixel size unknown"}`;
  const cs = $("channelSelect");
  cs.innerHTML = d.channel_names.map((n, i) => `<option value="${i}">${i}: ${n}</option>`).join("");
  const rs = $("refSelect");
  rs.innerHTML = `<option value="">None</option>` + d.channel_names.map((n, i) => `<option value="${i}">${i}: ${n}</option>`).join("");
  rs.value = d.reference_channel ?? "";
  $("refHint").textContent = d.reference_channel != null
    ? (d.reference_guessed ? "Guessed from the channel name. Every result is scored against it." : "Every result is scored against this expert mask.")
    : "Pick a channel that holds an expert mask to score results with Dice.";
  // Start on a channel that is not the reference mask
  if (d.default_channel != null) S.c = d.default_channel;
  else if (d.reference_channel === 0 && d.n_channels > 1) S.c = 1;
  cs.value = S.c;
  $("zSlider").max = d.n_z - 1;
  $("zStart").max = $("zEnd").max = d.n_z - 1;
  $("zStart").value = 0;
  $("zEnd").value = d.n_z - 1;
  $("zStep").value = Math.max(1, Math.round(d.n_z / 50));
  showEmpty(false);
  $("resultsSection").classList.add("hidden");
  await refreshDatasets();
  await loadPlane(true);
}

async function loadPlane(fit = false) {
  if (!S.ds) return;
  $("zSlider").value = S.z;
  $("zValue").textContent = `${S.z} / ${S.ds.n_z - 1}${S.ds.voxel_size_known ? ` · ${(S.z * S.ds.voxel_um[0]).toFixed(1)} µm` : ""}`;
  $("contrastValue").textContent = `${S.low}–${S.high}%`;
  const url = `/api/datasets/${S.ds.id}/plane?c=${S.c}&z=${S.z}&low=${S.low}&high=${S.high}`;
  busy(true, "Loading slice…");
  try {
    S.base = await loadImage(url);
  } catch (e) { toast(e.message, true); return; } finally { busy(false); }
  S.layers = {};
  S.result = null;
  $("resultsSection").classList.add("hidden");
  await loadReference();
  if (fit) fitView();
  updateCounts();
  draw();
  if (pts().pos.length || $("profileSelect").value) scheduleSegment(0);
}

async function loadReference() {
  if (!S.ds || S.ds.reference_channel == null) { delete S.layers.ref; return; }
  try { S.layers.ref = await loadImage(`/api/datasets/${S.ds.id}/reference?z=${S.z}`); } catch (_) { delete S.layers.ref; }
}

function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const xhr = new XMLHttpRequest();
  $("uploadProgress").classList.remove("hidden");
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    $("uploadProgress").querySelector(".bar span").style.width = `${(100 * e.loaded) / e.total}%`;
    $("uploadText").textContent = `Uploading ${file.name}: ${fmt(e.loaded / 1e6)} of ${fmt(e.total / 1e6)} MB`;
  };
  xhr.onload = () => {
    $("uploadProgress").classList.add("hidden");
    if (xhr.status !== 200) {
      let msg = xhr.statusText;
      try { msg = JSON.parse(xhr.responseText).detail; } catch (_) { /* keep status text */ }
      toast(`Upload failed: ${msg}`, true);
      return;
    }
    const d = JSON.parse(xhr.responseText);
    toast(`Loaded ${d.name}`);
    refreshDatasets(d.id);
  };
  xhr.onerror = () => { $("uploadProgress").classList.add("hidden"); toast("Upload failed", true); };
  $("uploadText").textContent = `Uploading ${file.name}…`;
  xhr.open("POST", "/api/datasets");
  xhr.send(fd);
}

// ---------------------------------------------------------------------------------------------
// Segmentation
let segTimer = null;
function scheduleSegment(delay = 120) {
  clearTimeout(segTimer);
  segTimer = setTimeout(runSegment, delay);
}

async function runSegment() {
  if (!S.ds) return;
  const p = pts();
  const profile = $("profileSelect").value || null;
  if (!p.pos.length && !profile) { toast("Click the structure you want first, or pick a profile"); return; }
  const seq = ++S.seq;
  busy(true, S.result ? "Updating…" : "Computing features…");
  try {
    const out = await api(`/api/datasets/${S.ds.id}/segment`, {
      method: "POST",
      body: { channel: S.c, z: S.z, pos: p.pos, neg: p.neg, profile_id: profile, settings: settings(), uncertainty: $("uncToggle").checked },
    });
    if (seq !== S.seq) return; // A newer request is on its way
    const [mask, heat, unc] = await Promise.all([
      loadImage(out.mask_png), loadImage(out.heat_png), out.uncertainty_png ? loadImage(out.uncertainty_png) : null,
    ]);
    if (seq !== S.seq) return;
    S.layers.mask = mask;
    S.layers.heat = heat;
    if (unc) S.layers.unc = unc; else delete S.layers.unc;
    S.result = out;
    showResults(out);
    draw();
    status(`Segmented slice ${S.z} in ${out.timing.total_s.toFixed(2)} s`);
  } catch (e) {
    if (seq === S.seq) toast(e.message, true);
  } finally {
    if (seq === S.seq) busy(false);
  }
}

function showResults(out) {
  $("resultsSection").classList.remove("hidden");
  const s = out.stats;
  const cards = [
    ["Area", `${fmt(s.area_um2)} µm²`],
    ["Area fraction", `${(100 * s.area_fraction).toFixed(1)}%`],
    ["Objects", `${s.n_objects}`],
    ["Objects per mm²", fmt(s.objects_per_mm2)],
    ["Median object", `${fmt(s.median_object_area_um2)} µm²`],
    ["Signal contrast", s.contrast_ratio ? `${s.contrast_ratio.toFixed(2)}×` : "–"],
  ];
  $("statCards").innerHTML = cards.map(([k, v]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  const ev = out.evaluation;
  $("evalBox").classList.toggle("hidden", !ev);
  if (ev) $("evalBox").innerHTML = `<b>Against the reference mask</b><br>Dice ${ev.dice.toFixed(3)} · IoU ${ev.iou.toFixed(3)} · HD95 ${fmt(ev.hd95_um)} µm`;
  const sug = out.suggestion;
  $("suggestionBox").classList.toggle("hidden", !(out.uncertainty_png));
  if (out.uncertainty_png) {
    $("suggestionBox").innerHTML = sug
      ? `<b>${(100 * out.uncertain_fraction).toFixed(1)}%</b> of the image is uncertain. The dashed ring marks where one more click helps most. Is it part of the structure?
         <div class="row" style="margin-top:6px"><button class="ghost" id="sugPos"><span class="dot pos"></span> Object</button><button class="ghost" id="sugNeg"><span class="dot neg"></span> Background</button></div>`
      : "The mask is stable under resampling of your clicks.";
    if (sug) {
      $("sugPos").onclick = () => addPoint("pos", sug[0], sug[1]);
      $("sugNeg").onclick = () => addPoint("neg", sug[0], sug[1]);
    }
  }
  const src = { clicks: "from your clicks", "carried over": "from the profile", top_percent: "fixed share", manual: "manual", otsu: "Otsu" }[out.threshold_source] || out.threshold_source;
  $("timing").textContent = `Threshold ${out.threshold.toFixed(3)} (${src}) · features ${out.timing.embed_s.toFixed(2)} s · total ${out.timing.total_s.toFixed(2)} s`;
}

function addPoint(kind, y, x) {
  pts()[kind].push([Math.round(y), Math.round(x)]);
  S.history.push({ key: key(), kind });
  updateCounts();
  draw();
  if ($("autoRun").checked) scheduleSegment();
}

function removeNearest(y, x) {
  const p = pts();
  let best = null;
  for (const kind of ["pos", "neg"]) {
    p[kind].forEach(([py, px], i) => {
      const d = (py - y) ** 2 + (px - x) ** 2;
      if (!best || d < best.d) best = { kind, i, d };
    });
  }
  if (!best) return;
  p[best.kind].splice(best.i, 1);
  updateCounts();
  draw();
  if ($("autoRun").checked && (p.pos.length || $("profileSelect").value)) scheduleSegment();
}

function undo() {
  const h = S.history.pop();
  if (!h) return;
  const p = S.points[h.key];
  p?.[h.kind].pop();
  updateCounts();
  draw();
  if ($("autoRun").checked && (pts().pos.length || $("profileSelect").value)) scheduleSegment();
}

function updateCounts() {
  const p = pts();
  $("nPos").textContent = p.pos.length;
  $("nNeg").textContent = p.neg.length;
}

function autoBackground() {
  if (!S.ds) return;
  const { height: h, width: w } = S.ds;
  const m = 0.02;
  for (const [fy, fx] of [[m, m], [m, 0.5], [m, 1 - m], [0.5, m], [0.5, 1 - m], [1 - m, m], [1 - m, 0.5], [1 - m, 1 - m]]) {
    pts().neg.push([Math.round(fy * h), Math.round(fx * w)]);
    S.history.push({ key: key(), kind: "neg" });
  }
  updateCounts();
  draw();
  if ($("autoRun").checked && pts().pos.length) scheduleSegment();
}

// ---------------------------------------------------------------------------------------------
// Viewer: drawing, zoom and pan
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

function fullToDisp() { return S.base ? S.base.naturalWidth / S.ds.width : 1; }

function fitView() {
  if (!S.base) return;
  const r = canvas.getBoundingClientRect();
  const s = Math.min(r.width / S.base.naturalWidth, r.height / S.base.naturalHeight) * 0.96;
  S.view = { scale: s, ox: (r.width - S.base.naturalWidth * s) / 2, oy: (r.height - S.base.naturalHeight * s) / 2 };
  draw();
}

function draw() {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) {
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  if (!S.base) return;
  const { scale, ox, oy } = S.view;
  const w = S.base.naturalWidth * scale;
  const h = S.base.naturalHeight * scale;
  ctx.imageSmoothingEnabled = scale < 2;
  ctx.drawImage(S.base, ox, oy, w, h);
  const op = +$("opacity").value;
  const layer = (img, alpha) => { if (img) { ctx.globalAlpha = alpha; ctx.drawImage(img, ox, oy, w, h); ctx.globalAlpha = 1; } };
  if ($("showHeat").checked) layer(S.layers.heat, op);
  if ($("showUnc").checked) layer(S.layers.unc, Math.min(1, op + 0.2));
  if ($("showMask").checked) layer(S.layers.mask, Math.min(1, op + 0.3));
  if ($("showRef").checked) layer(S.layers.ref, 0.9);
  $("zoomLabel").textContent = `${Math.round(scale * fullToDisp() * 100)}% of full size`;

  const f = fullToDisp();
  const toScreen = ([y, x]) => [ox + x * f * scale, oy + y * f * scale];
  if ($("showPoints").checked) {
    const p = S.points[key()] || { pos: [], neg: [] };
    for (const [kind, color] of [["pos", "#22d27a"], ["neg", "#ff5a6e"]]) {
      for (const pt of p[kind]) {
        const [sx, sy] = toScreen(pt);
        ctx.beginPath();
        ctx.arc(sx, sy, 5.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#0b1220";
        ctx.stroke();
      }
    }
  }
  const sug = S.result?.suggestion;
  if (sug && $("uncToggle").checked) {
    const [sx, sy] = toScreen(sug);
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ffa53a";
    ctx.beginPath();
    ctx.arc(sx, sy, 14, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function screenToFull(ev) {
  const r = canvas.getBoundingClientRect();
  const f = fullToDisp();
  const x = (ev.clientX - r.left - S.view.ox) / S.view.scale / f;
  const y = (ev.clientY - r.top - S.view.oy) / S.view.scale / f;
  return [y, x];
}

function inside([y, x]) { return S.ds && y >= 0 && x >= 0 && y < S.ds.height && x < S.ds.width; }

let drag = null;
canvas.addEventListener("mousedown", (ev) => {
  if (!S.base) return;
  drag = { x: ev.clientX, y: ev.clientY, ox: S.view.ox, oy: S.view.oy, moved: false, button: ev.button, pan: ev.button === 1 || S.space };
  if (drag.pan) $("viewer").classList.add("panning");
});
window.addEventListener("mousemove", (ev) => {
  if (S.base && ev.target === canvas) {
    const [y, x] = screenToFull(ev);
    $("cursorInfo").textContent = inside([y, x])
      ? `x ${Math.round(x)} · y ${Math.round(y)}${S.ds.voxel_size_known ? ` · ${(x * S.ds.voxel_um[2]).toFixed(1)}, ${(y * S.ds.voxel_um[1]).toFixed(1)} µm` : ""}`
      : "";
  }
  if (!drag) return;
  const dx = ev.clientX - drag.x;
  const dy = ev.clientY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 5) drag.moved = true;
  if (drag.moved && (drag.pan || drag.button === 0)) {
    S.view.ox = drag.ox + dx;
    S.view.oy = drag.oy + dy;
    $("viewer").classList.add("panning");
    draw();
  }
});
window.addEventListener("mouseup", (ev) => {
  if (!drag) return;
  const d = drag;
  drag = null;
  $("viewer").classList.remove("panning");
  if (d.moved || d.pan || ev.target !== canvas) return;
  const p = screenToFull(ev);
  if (!inside(p)) return;
  if (d.button === 2) { removeNearest(...p); return; }
  if (d.button !== 0) return;
  const kind = ev.shiftKey ? (S.mode === "pos" ? "neg" : "pos") : S.mode;
  addPoint(kind, ...p);
});
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("wheel", (ev) => {
  if (!S.base) return;
  ev.preventDefault();
  const r = canvas.getBoundingClientRect();
  const mx = ev.clientX - r.left;
  const my = ev.clientY - r.top;
  const factor = Math.exp(-ev.deltaY * 0.0015);
  const ns = Math.min(40, Math.max(0.05, S.view.scale * factor));
  S.view.ox = mx - ((mx - S.view.ox) * ns) / S.view.scale;
  S.view.oy = my - ((my - S.view.oy) * ns) / S.view.scale;
  S.view.scale = ns;
  draw();
}, { passive: false });
window.addEventListener("resize", () => draw());

// ---------------------------------------------------------------------------------------------
// Profiles
async function refreshProfiles(selectId) {
  S.profiles = await api("/api/profiles");
  const sel = $("profileSelect");
  const current = selectId ?? sel.value;
  sel.innerHTML = `<option value="">None, use my clicks</option>` +
    S.profiles.map((p) => `<option value="${p.id}" title="${p.description || p.source}">${p.name} · ${p.backbone}</option>`).join("");
  sel.value = S.profiles.some((p) => p.id === current) ? current : "";
  $("deleteProfileBtn").classList.toggle("hidden", !sel.value);
}

async function saveProfile() {
  const name = $("profileName").value.trim();
  if (!name) { toast("Give the profile a name"); return; }
  const p = pts();
  try {
    const prof = await api("/api/profiles", {
      method: "POST",
      body: { name, description: $("profileDesc").value, dataset_id: S.ds.id, channel: S.c, z: S.z, pos: p.pos, neg: p.neg, settings: settings() },
    });
    $("profileDialog").close();
    toast(`Saved profile ${prof.name}`);
    await refreshProfiles();
  } catch (e) { toast(e.message, true); }
}

// ---------------------------------------------------------------------------------------------
// Stack jobs
async function runStack() {
  const p = pts();
  const profile = $("profileSelect").value || null;
  if (!p.pos.length && !profile) { toast("Click the structure on this slice first, or pick a profile"); return; }
  try {
    const job = await api(`/api/datasets/${S.ds.id}/stack`, {
      method: "POST",
      body: {
        channel: S.c, ref_z: S.z, pos: p.pos, neg: p.neg, profile_id: profile, settings: settings(),
        z_start: +$("zStart").value, z_end: +$("zEnd").value, z_step: +$("zStep").value,
      },
    });
    S.job = job.id;
    $("jobBox").classList.remove("hidden");
    $("jobDownloads").innerHTML = "";
    $("stackChart").innerHTML = "";
    $("chartLegend").textContent = "";
    $("cancelBtn").classList.remove("hidden");
    $("stackBtn").disabled = true;
    pollJob(job.id);
  } catch (e) { toast(e.message, true); }
}

async function pollJob(id) {
  let job;
  try { job = await api(`/api/jobs/${id}`); } catch (e) { toast(e.message, true); return; }
  $("jobBar").style.width = `${100 * job.progress}%`;
  $("jobText").textContent = job.status === "running" || job.status === "queued" ? job.message || "Starting…" : "";
  if (job.status === "running" || job.status === "queued") { setTimeout(() => pollJob(id), 600); return; }
  $("cancelBtn").classList.add("hidden");
  $("stackBtn").disabled = false;
  if (job.status === "failed") { $("jobText").textContent = `Failed: ${job.error}`; return; }
  const s = job.result.summary || {};
  $("jobText").innerHTML = `${job.status === "cancelled" ? "Cancelled after" : "Done:"} ${s.n_slices || 0} slices · volume ${fmt(s.volume_um3)} µm³ · mean area ${(100 * (s.mean_area_fraction || 0)).toFixed(1)}%`
    + (s.n_objects_3d != null ? ` · ${s.n_objects_3d} objects in 3D (${s.n_objects_3d_inside} not cut by the stack ends), median ${fmt(s.median_object_volume_um3)} µm³` : "")
    + (s.mean_dice_vs_reference != null ? ` · mean Dice ${s.mean_dice_vs_reference.toFixed(3)}` : "");
  drawChart(job.result.slices || []);
  $("jobDownloads").innerHTML = ["masks.tif", "slices.csv", "objects_3d.csv", "summary.json"]
    .map((f) => `<a class="small" href="/api/jobs/${id}/files/${f}" download><button class="ghost">${f}</button></a>`).join("");
}

function drawChart(rows) {
  const svg = $("stackChart");
  if (!rows.length) { svg.innerHTML = ""; return; }
  const W = 300, H = 120, pad = 8;
  const zs = rows.map((r) => r.z);
  const z0 = Math.min(...zs), z1 = Math.max(...zs);
  const sx = (z) => pad + ((z - z0) / Math.max(1, z1 - z0)) * (W - 2 * pad);
  const series = [["area_fraction", "#00c8f0", "Area fraction"]];
  if (rows[0].dice != null) series.push(["dice", "#22d27a", "Dice vs reference"]);
  const maxA = Math.max(...rows.map((r) => r.area_fraction), 1e-6);
  let out = "";
  for (const [k, color] of series) {
    const scale = k === "area_fraction" ? maxA : 1;
    const d = rows.map((r, i) => `${i ? "L" : "M"}${sx(r.z).toFixed(1)},${(H - pad - (r[k] / scale) * (H - 2 * pad)).toFixed(1)}`).join("");
    out += `<path d="${d}" fill="none" stroke="${color}" stroke-width="1.6" vector-effect="non-scaling-stroke"/>`;
  }
  out += `<line x1="${sx(S.z)}" x2="${sx(S.z)}" y1="0" y2="${H}" stroke="#ffa53a" stroke-dasharray="3 3" vector-effect="non-scaling-stroke"/>`;
  svg.innerHTML = out;
  svg.onclick = (ev) => {
    const r = svg.getBoundingClientRect();
    const z = z0 + ((ev.clientX - r.left) / r.width) * (z1 - z0);
    const nearest = zs.reduce((a, b) => (Math.abs(b - z) < Math.abs(a - z) ? b : a));
    S.z = nearest;
    loadPlane();
  };
  $("chartLegend").innerHTML = series.map(([, c, label]) => `<span style="color:${c}">━</span> ${label}`).join(" · ")
    + ` · area peaks at ${(100 * maxA).toFixed(1)}% · click the chart to jump to a slice`;
}

// ---------------------------------------------------------------------------------------------
// Wiring
function setMode(m) {
  S.mode = m;
  $("modePos").classList.toggle("active", m === "pos");
  $("modeNeg").classList.toggle("active", m === "neg");
}

function bind() {
  $("fileInput").onchange = (e) => e.target.files[0] && uploadFile(e.target.files[0]);
  const dz = $("dropzone");
  ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", (e) => e.dataTransfer.files[0] && uploadFile(e.dataTransfer.files[0]));
  document.body.addEventListener("dragover", (e) => e.preventDefault());
  document.body.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files[0] && !dz.contains(e.target)) uploadFile(e.dataTransfer.files[0]); });
  $("pathBtn").onclick = async () => {
    const path = $("pathInput").value.trim();
    if (!path) return;
    busy(true, "Opening file…");
    try { const d = await api("/api/datasets/from-path", { method: "POST", body: { path } }); refreshDatasets(d.id); }
    catch (e) { toast(e.message, true); } finally { busy(false); }
  };
  $("demoBtn").onclick = async () => {
    busy(true, "Creating demo image…");
    try { const d = await api("/api/datasets/demo", { method: "POST" }); await refreshDatasets(d.id); toast("Demo loaded. Click a few bright cells, then some background."); }
    catch (e) { toast(e.message, true); } finally { busy(false); }
  };

  $("channelSelect").onchange = (e) => { S.c = +e.target.value; loadPlane(); };
  $("zSlider").oninput = (e) => { S.z = +e.target.value; $("zValue").textContent = S.z; };
  $("zSlider").onchange = () => loadPlane();
  $("zPrev").onclick = () => stepZ(-1);
  $("zNext").onclick = () => stepZ(1);
  const contrast = () => { S.low = +$("lowSlider").value; S.high = +$("highSlider").value; loadPlane(); };
  $("lowSlider").onchange = contrast;
  $("highSlider").onchange = contrast;
  $("refSelect").onchange = async (e) => {
    const v = e.target.value === "" ? null : +e.target.value;
    try {
      S.ds = await api(`/api/datasets/${S.ds.id}`, { method: "PATCH", body: { reference_channel: v } });
      $("refHint").textContent = v == null ? "Pick a channel that holds an expert mask to score results with Dice." : "Every result is scored against this expert mask.";
      await loadReference();
      draw();
      if (S.result) scheduleSegment(0);
    } catch (err) { toast(err.message, true); }
  };

  $("modePos").onclick = () => setMode("pos");
  $("modeNeg").onclick = () => setMode("neg");
  $("undoBtn").onclick = undo;
  $("clearBtn").onclick = () => { S.points[key()] = { pos: [], neg: [] }; S.layers.mask = S.layers.heat = S.layers.unc = null; S.result = null; $("resultsSection").classList.add("hidden"); updateCounts(); draw(); };
  $("autoNegBtn").onclick = autoBackground;
  $("profileSelect").onchange = () => {
    $("deleteProfileBtn").classList.toggle("hidden", !$("profileSelect").value);
    const prof = S.profiles.find((p) => p.id === $("profileSelect").value);
    if (prof && prof.backbone !== $("backboneSelect").value) { $("backboneSelect").value = prof.backbone; syncSettingLabels(); }
    if (prof || pts().pos.length) scheduleSegment(0);
  };
  $("saveProfileBtn").onclick = () => {
    if (!pts().pos.length) { toast("Add object clicks first"); return; }
    $("profileName").value = "";
    $("profileDesc").value = S.ds ? `${S.ds.channel_names[S.c]}` : "";
    $("profileDialog").showModal();
  };
  $("profileSaveConfirm").onclick = saveProfile;
  $("profileCancel").onclick = () => $("profileDialog").close();
  $("deleteProfileBtn").onclick = async () => {
    const id = $("profileSelect").value;
    if (!id || !confirm("Delete this profile?")) return;
    await api(`/api/profiles/${id}`, { method: "DELETE" });
    refreshProfiles("");
  };

  for (const id of ["thrMode", "topPercent", "manualThr", "lambda", "minObj", "fillHoles", "smooth", "vitSize", "layer", "backboneSelect"]) {
    $(id).addEventListener("input", syncSettingLabels);
    $(id).addEventListener("change", () => { if (S.result) scheduleSegment(0); });
  }
  $("uncToggle").onchange = () => { if (S.result) scheduleSegment(0); };
  $("segmentBtn").onclick = () => runSegment();
  for (const id of ["showMask", "showHeat", "showUnc", "showRef", "showPoints", "opacity"]) $(id).addEventListener("input", draw);
  $("fitBtn").onclick = fitView;
  document.querySelectorAll("[data-export]").forEach((b) => {
    b.onclick = () => {
      const kind = b.dataset.export;
      const base = `/api/datasets/${S.ds.id}/export`;
      window.location = kind === "csv" ? `${base}/objects.csv?c=${S.c}&z=${S.z}` : `${base}/mask?c=${S.c}&z=${S.z}&fmt=${kind}`;
    };
  });
  $("stackBtn").onclick = runStack;
  $("cancelBtn").onclick = () => S.job && api(`/api/jobs/${S.job}/cancel`, { method: "POST" });
  $("helpBtn").onclick = () => $("helpDialog").showModal();

  window.addEventListener("keydown", (e) => {
    if (e.target.matches("input[type=text], input[type=number], select, textarea") || document.querySelector("dialog[open]")) return;
    const k = e.key.toLowerCase();
    if (k === " ") { S.space = true; e.preventDefault(); }
    else if (k === "1") setMode("pos");
    else if (k === "2") setMode("neg");
    else if ((e.ctrlKey || e.metaKey) && k === "z") { e.preventDefault(); undo(); }
    else if (k === ",") stepZ(-1);
    else if (k === ".") stepZ(1);
    else if (k === "f") fitView();
    else if (k === "m") toggle("showMask");
    else if (k === "h") toggle("showHeat");
    else if (k === "u") toggle("showUnc");
    else if (k === "?") $("helpDialog").showModal();
  });
  window.addEventListener("keyup", (e) => { if (e.key === " ") S.space = false; });
}

function toggle(id) { $(id).checked = !$(id).checked; draw(); }
function stepZ(d) {
  if (!S.ds) return;
  const z = Math.min(S.ds.n_z - 1, Math.max(0, S.z + d));
  if (z !== S.z) { S.z = z; loadPlane(); }
}

async function init() {
  bind();
  showEmpty(true);
  try {
    S.health = await api("/api/health");
    $("version").textContent = `v${S.health.version}`;
    $("devicePill").textContent = `Runs on ${S.health.device.toUpperCase()}`;
    $("backboneSelect").innerHTML = S.health.backbones.map((b) => `<option value="${b.id}">${b.label}${b.ready ? "" : " · downloads on first use"}</option>`).join("");
    const d = S.health.default_settings;
    $("backboneSelect").value = d.backbone;
    $("thrMode").value = d.threshold_mode;
    $("lambda").value = d.neg_weight;
    syncSettingLabels();
    await Promise.all([refreshDatasets(), refreshProfiles()]);
    if (S.datasets.length) openDataset(S.datasets[0].id);
  } catch (e) {
    toast(`Could not reach the server: ${e.message}`, true);
  }
}

init();

"use strict";
// Simple mode: open an image, click bone and background, run the whole stack, download the results.
// Uses the same web API as the full app; helpers api(), esc(), fmt(), loadImage(), toast() and
// takeTokenFromUrl() come from core.js, which index.html and simple.html both load first.

// ---------------------------------------------------------------------------------------------
// State of the page. Click positions are (y, x) in full-resolution image pixels, as the API expects.
const P = {
  ds: null,            // Dataset info from GET /api/datasets/{id}
  c: 0,                // Channel shown and segmented
  z: 0,                // Slice shown
  img: null,           // The slice as an <img> (at most 1600 px on its longest side)
  mask: null,          // The mask overlay as an <img>, or null
  clicks: {},          // "c:z" -> {pos: [[y, x]], neg: [[y, x]], extra: [...]}, saved on the server
  clicksZ: null,       // Slice whose clicks are used; other slices are segmented with them, like a stack run
  mode: "pos",         // What the next click marks: "pos" (bone) or "neg" (background)
  history: [],         // Undo stack of [key, kind]
  view: { scale: 1, ox: 0, oy: 0 },   // Canvas transform: screen = image * scale + offset
  seq: 0,              // Increases with every request, so a slow old response cannot overwrite a newer one
  job: null,           // Id of the running stack job
};
const canvas = $("canvas");
const ctx2d = canvas.getContext("2d");
const MIN_CLICKS = 3;  // The hint asks for at least this many of each kind

const keyOf = (c, z) => `${c}:${z}`;
// First slice of the current channel that has bone clicks, or null
const clickedSlice = () => {
  const zs = Object.keys(P.clicks).filter((k) => k.startsWith(`${P.c}:`) && P.clicks[k].pos?.length).map((k) => +k.split(":")[1]);
  return zs.length ? Math.min(...zs) : null;
};
const here = () => (P.clicks[keyOf(P.c, P.z)] ||= { pos: [], neg: [] });
// The clicks the mask is made from: this slice's own, or those of the clicked slice
const activeClicks = () => {
  const own = P.clicks[keyOf(P.c, P.z)];
  if (own && own.pos.length) return { ...own, z: P.z };
  const other = P.clicksZ != null ? P.clicks[keyOf(P.c, P.clicksZ)] : null;
  return other && other.pos.length ? { ...other, z: P.clicksZ } : null;
};

// ---------------------------------------------------------------------------------------------
// Step 1: opening an image

// Streams the file as the raw request body (no temporary copy on the server), with a progress bar
function upload(file) {
  const xhr = new XMLHttpRequest();
  $("upload").classList.remove("hidden");
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    $("upload").querySelector(".bar span").style.width = `${(100 * e.loaded) / e.total}%`;
    $("uploadText").textContent = `Uploading ${file.name}: ${fmt(e.loaded / 1e6)} of ${fmt(e.total / 1e6)} MB`;
  };
  xhr.onload = () => {
    $("upload").classList.add("hidden");
    if (xhr.status !== 200) {
      let msg = xhr.statusText;
      try { msg = JSON.parse(xhr.responseText).detail; } catch (_) { /* not JSON, keep the status text */ }
      toast(`Upload failed: ${msg}`, true);
      return;
    }
    openDataset(JSON.parse(xhr.responseText).id);
  };
  xhr.onerror = () => { $("upload").classList.add("hidden"); toast("Upload failed", true); };
  xhr.open("POST", `/api/datasets/stream?filename=${encodeURIComponent(file.name)}`);
  xhr.setRequestHeader("Content-Type", "application/octet-stream");
  xhr.send(file);
}

// Buttons for images opened before, so a reload or a second visit continues where it left off
async function showRecent() {
  const list = await api("/api/datasets");
  $("recent").innerHTML = "";
  for (const d of list.slice(0, 6)) {
    const b = document.createElement("button");
    b.className = "ghost small";
    b.textContent = d.name;   // textContent, not innerHTML: file names are not trusted markup
    b.onclick = () => openDataset(d.id);
    $("recent").appendChild(b);
  }
}

async function openDataset(id) {
  const d = await api(`/api/datasets/${id}`);
  P.ds = d;
  P.history = [];
  P.mask = null;
  // Default channel: the one the app remembered or guessed, never the expert-mask (reference) channel
  P.c = d.default_channel ?? d.rgb_channel ?? 0;
  if (P.c === d.reference_channel) P.c = d.reference_channel === 0 && d.n_channels > 1 ? 1 : 0;
  try { P.clicks = await api(`/api/datasets/${id}/annotations`); } catch (_) { P.clicks = {}; }
  // Start on a slice that already has clicks, else in the middle of the stack
  P.clicksZ = clickedSlice();
  P.z = P.clicksZ ?? Math.floor((d.n_z - 1) / 2);

  $("title").textContent = d.name;
  showOpenControls(false);   // Collapse step 1 to one line
  $("openedName").textContent = d.name;
  const [vz, vy, vx] = d.voxel_um;
  $("info").textContent = `${d.width} × ${d.height} px, ${d.n_z} slice${d.n_z > 1 ? "s" : ""}, ${d.n_channels} channel${d.n_channels > 1 ? "s" : ""}. `
    + (d.voxel_size_known ? `Pixel size ${vx.toFixed(3)} × ${vy.toFixed(3)} µm${d.n_z > 1 ? `, slices ${vz.toFixed(2)} µm apart` : ""}.`
      : "The file has no pixel size, so areas are in pixels. Set it under \"Pixel size\" in the full app.");
  // Channel menu only when there is a choice
  $("channelRow").classList.toggle("hidden", d.n_channels < 2);
  $("channel").innerHTML = d.channel_names.map((n, i) => `<option value="${i}">${i}: ${esc(n)}${i === d.reference_channel ? " (expert mask)" : ""}</option>`).join("");
  $("channel").value = P.c;
  // Slice slider only for stacks
  $("sliceRow").classList.toggle("hidden", d.n_z < 2);
  $("slice").max = d.n_z - 1;
  $("slice").value = P.z;
  $("step2").classList.remove("off");
  $("step3").classList.toggle("off", d.n_z < 2);
  $("stackNumbers").innerHTML = "";
  $("downloads").innerHTML = "";
  await loadSlice(true);
  showRecent();
}

// Step 1 is either the full set of open controls or, with an image open, a single line
function showOpenControls(on) {
  $("openControls").classList.toggle("hidden", !on);
  $("openedRow").classList.toggle("hidden", on);
}

// ---------------------------------------------------------------------------------------------
// Step 2: showing a slice, collecting clicks and segmenting

async function loadSlice(fit = false) {
  const seq = ++P.seq;
  const img = await loadImage(`/api/datasets/${P.ds.id}/plane?c=${P.c}&z=${P.z}&max_side=1600`);
  if (seq !== P.seq) return;   // The user moved on while this slice was loading
  P.img = img;
  P.mask = null;
  if (fit) fitView();
  $("sliceText").textContent = `${P.z + 1} of ${P.ds.n_z}`;
  $("toClicks").classList.toggle("hidden", P.clicksZ == null || P.clicksZ === P.z);
  draw();
  segment();
}

// Full-resolution pixels per displayed pixel of the slice image
const fullPerImg = () => P.ds.width / P.img.naturalWidth;

function fitView() {
  const r = canvas.getBoundingClientRect();
  canvas.width = r.width * devicePixelRatio;
  canvas.height = r.height * devicePixelRatio;
  const s = Math.min(canvas.width / P.img.naturalWidth, canvas.height / P.img.naturalHeight);
  P.view = { scale: s, ox: (canvas.width - P.img.naturalWidth * s) / 2, oy: (canvas.height - P.img.naturalHeight * s) / 2 };
}

// Redraws image, mask and clicks with the current zoom and pan
function draw() {
  ctx2d.setTransform(1, 0, 0, 1, 0, 0);
  ctx2d.fillStyle = "#000";
  ctx2d.fillRect(0, 0, canvas.width, canvas.height);
  if (!P.img) return;
  const { scale, ox, oy } = P.view;
  ctx2d.setTransform(scale, 0, 0, scale, ox, oy);
  ctx2d.imageSmoothingEnabled = scale < 1;
  ctx2d.drawImage(P.img, 0, 0);
  if (P.mask && $("showMask").checked) ctx2d.drawImage(P.mask, 0, 0, P.img.naturalWidth, P.img.naturalHeight);
  // Clicks are only drawn on the slice they belong to
  const p = P.clicks[keyOf(P.c, P.z)];
  if (!p) return;
  const f = fullPerImg();
  const r = 6 / scale;   // 6 screen pixels whatever the zoom
  for (const [kind, color] of [["pos", "#22d27a"], ["neg", "#ff5a6e"]]) {
    for (const [y, x] of p[kind]) {
      ctx2d.beginPath();
      ctx2d.arc(x / f, y / f, r, 0, 2 * Math.PI);
      ctx2d.fillStyle = color;
      ctx2d.fill();
      ctx2d.lineWidth = 2 / scale;
      ctx2d.strokeStyle = "#000";
      ctx2d.stroke();
    }
  }
}

// Screen position of a mouse event -> (y, x) in full-resolution pixels
function toFull(ev) {
  const r = canvas.getBoundingClientRect();
  const sx = (ev.clientX - r.left) * devicePixelRatio, sy = (ev.clientY - r.top) * devicePixelRatio;
  const f = fullPerImg();
  return [((sy - P.view.oy) / P.view.scale) * f, ((sx - P.view.ox) / P.view.scale) * f];
}

function addClick(ev) {
  const [y, x] = toFull(ev);
  if (y < 0 || x < 0 || y >= P.ds.height || x >= P.ds.width) return;   // Outside the image
  here()[P.mode].push([Math.round(y), Math.round(x)]);
  P.history.push([keyOf(P.c, P.z), P.mode]);
  P.clicksZ = P.z;   // Clicking on a slice makes it the clicked slice
  clicksChanged();
}

// Right-click: remove the click nearest to the pointer, if it is within 20 screen pixels
function removeNearest(ev) {
  const p = P.clicks[keyOf(P.c, P.z)];
  if (!p) return;
  const [y, x] = toFull(ev);
  const limit = (20 * devicePixelRatio / P.view.scale) * fullPerImg();
  let best = null;
  for (const kind of ["pos", "neg"]) {
    p[kind].forEach(([py, px], i) => {
      const d = Math.hypot(py - y, px - x);
      if (d < limit && (!best || d < best.d)) best = { kind, i, d };
    });
  }
  if (!best) return;
  p[best.kind].splice(best.i, 1);
  clicksChanged();
}

function undo() {
  const last = P.history.pop();
  if (!last) return;
  const [k, kind] = last;
  P.clicks[k]?.[kind].pop();
  clicksChanged();
}

// After any change to the clicks: save them, redraw, segment again
let saveTimer = null;
function clicksChanged() {
  const p = here(), dsId = P.ds.id, c = P.c, z = P.z;
  clearTimeout(saveTimer);
  // Saved like the full app does, keeping any extra structures the full app added on this slice
  saveTimer = setTimeout(() => api(`/api/datasets/${dsId}/annotations`, { method: "PUT", body: { channel: c, z, pos: p.pos, neg: p.neg, extra: p.extra || [] } })
    .catch((e) => toast(`Could not save clicks: ${e.message}`, true)), 400);
  draw();
  clearTimeout(segTimer);
  segTimer = setTimeout(segment, 150);   // Wait a moment, so quick clicking sends one request, not many
}

// Asks the server for this slice's mask, made with the app's default settings
let segTimer = null;
async function segment() {
  const a = activeClicks();
  updateHint(a);
  if (!a || !a.neg.length) { P.mask = null; $("numbers").innerHTML = ""; draw(); return; }
  const seq = ++P.seq;
  try {
    const out = await api(`/api/datasets/${P.ds.id}/segment`, {
      method: "POST",
      body: { method: "clicks", channel: P.c, z: P.z, pos: a.pos, neg: a.neg, clicks_z: a.z, settings: {}, max_side: 1600 },
    });
    if (seq !== P.seq) return;
    P.mask = await loadImage(out.mask_png);
    if (seq !== P.seq) return;
    const st = out.stats;
    const unit = P.ds.voxel_size_known ? "µm²" : "px²";
    $("numbers").innerHTML = `<span>Bone area <b>${(100 * st.area_fraction).toFixed(1)}%</b> of the slice</span>`
      + `<span><b>${fmt(st.area_um2, 0)}</b> ${unit}</span><span><b>${st.n_objects}</b> separate pieces</span>`
      + (a.z !== P.z ? `<span class="muted">from the clicks on slice ${a.z + 1}</span>` : "");
    draw();
  } catch (e) { toast(e.message, true); }
}

function updateHint(a) {
  const own = P.clicks[keyOf(P.c, P.z)] || { pos: [], neg: [] };
  const np = own.pos.length, nn = own.neg.length;
  let h;
  if (a && a.z !== P.z) h = `This slice is segmented with your clicks on slice ${a.z + 1}. Click here to start new clicks on this slice.`;
  else if (np < MIN_CLICKS) h = `Click on bone (green). ${np} of at least ${MIN_CLICKS} so far.`;
  else if (nn < MIN_CLICKS) h = `Now switch to Background and click on things that are not bone. ${nn} of at least ${MIN_CLICKS}.`;
  else h = "Click where the mask is wrong: Bone where it misses bone, Background where it covers something else.";
  $("hint").textContent = h;
}

function setMode(m) {
  P.mode = m;
  $("posBtn").setAttribute("aria-pressed", String(m === "pos"));
  $("negBtn").setAttribute("aria-pressed", String(m === "neg"));
}

// ---------------------------------------------------------------------------------------------
// Step 3: whole stack and downloads

async function runStack() {
  const a = activeClicks();
  if (!a || a.pos.length < 1 || a.neg.length < 1) { toast("Click bone and background on a slice first"); return; }
  try {
    const job = await api(`/api/datasets/${P.ds.id}/stack`, {
      method: "POST",
      body: { method: "clicks", channel: P.c, ref_z: a.z, pos: a.pos, neg: a.neg, settings: {}, z_start: 0, z_end: P.ds.n_z - 1, z_step: 1 },
    });
    P.job = job.id;
    $("job").classList.remove("hidden");
    $("stackBtn").disabled = true;
    $("cancelBtn").classList.remove("hidden");
    $("downloads").innerHTML = "";
    $("stackNumbers").innerHTML = "";
    pollJob(job.id);
  } catch (e) { toast(e.message, true); }
}

// Checks the job every second until it is done, then shows the numbers and the download buttons
async function pollJob(id) {
  let job;
  try { job = await api(`/api/jobs/${id}`); } catch (e) { toast(e.message, true); return; }
  $("job").querySelector(".bar span").style.width = `${100 * job.progress}%`;
  if (job.status === "queued" || job.status === "running") {
    $("jobText").textContent = job.message || "Starting…";
    setTimeout(() => pollJob(id), 1000);
    return;
  }
  $("stackBtn").disabled = false;
  $("cancelBtn").classList.add("hidden");
  if (job.status === "failed") { $("jobText").textContent = `Failed: ${job.error}`; return; }
  const s = job.result.summary || {};
  const unit = P.ds.voxel_size_known ? "µm³" : "voxels";
  $("jobText").textContent = job.status === "cancelled" ? "Stopped. The files hold the slices done so far." : "Done.";
  $("stackNumbers").innerHTML = `<span><b>${s.n_slices || 0}</b> slices</span><span>Bone volume <b>${fmt(s.volume_um3, 0)}</b> ${unit}</span>`
    + `<span>Mean bone area <b>${(100 * (s.mean_area_fraction || 0)).toFixed(1)}%</b> per slice</span>`;
  const name = P.ds.name.replace(/\.[^.]+$/, "");
  $("downloads").innerHTML = [["masks.tif", "Masks (TIFF stack)"], ["slices.csv", "Measurements per slice (CSV)"]]
    .map(([f, label]) => `<a href="/api/jobs/${id}/files/${f}" download="${esc(name)}_${f}"><button>${label}</button></a>`).join("");
}

// ---------------------------------------------------------------------------------------------
// Wiring: buttons, keys and the mouse

function bind() {
  $("drop").addEventListener("dragover", (e) => { e.preventDefault(); $("drop").classList.add("over"); });
  $("drop").addEventListener("dragleave", () => $("drop").classList.remove("over"));
  $("drop").addEventListener("drop", (e) => { e.preventDefault(); $("drop").classList.remove("over"); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });
  $("file").onchange = (e) => { if (e.target.files[0]) upload(e.target.files[0]); e.target.value = ""; };
  $("pathBtn").onclick = async () => {
    try { openDataset((await api("/api/datasets/from-path", { method: "POST", body: { path: $("path").value.trim() } })).id); } catch (e) { toast(e.message, true); }
  };
  $("demoBtn").onclick = async () => { try { openDataset((await api("/api/datasets/demo", { method: "POST" })).id); } catch (e) { toast(e.message, true); } };
  $("changeBtn").onclick = () => showOpenControls(true);
  $("posBtn").onclick = () => setMode("pos");
  $("negBtn").onclick = () => setMode("neg");
  $("undoBtn").onclick = undo;
  $("clearBtn").onclick = () => { const p = here(); p.pos = []; p.neg = []; clicksChanged(); };
  $("showMask").onchange = draw;
  $("channel").onchange = () => { P.c = +$("channel").value; P.clicksZ = clickedSlice(); loadSlice(); };
  $("toClicks").onclick = () => { P.z = P.clicksZ; $("slice").value = P.z; loadSlice(); };
  let sliceTimer = null;
  $("slice").oninput = () => { P.z = +$("slice").value; $("sliceText").textContent = `${P.z + 1} of ${P.ds.n_z}`; clearTimeout(sliceTimer); sliceTimer = setTimeout(loadSlice, 120); };
  $("stackBtn").onclick = runStack;
  $("cancelBtn").onclick = () => P.job && api(`/api/jobs/${P.job}/cancel`, { method: "POST" }).catch(() => {});

  // Mouse: a click without movement adds a click; a drag pans; the wheel zooms around the pointer
  let down = null;
  canvas.addEventListener("pointerdown", (e) => { if (e.button === 0) down = { x: e.clientX, y: e.clientY, ox: P.view.ox, oy: P.view.oy, moved: false }; });
  canvas.addEventListener("pointermove", (e) => {
    if (!down) return;
    const dx = e.clientX - down.x, dy = e.clientY - down.y;
    if (Math.hypot(dx, dy) > 4) down.moved = true;
    if (down.moved) { P.view.ox = down.ox + dx * devicePixelRatio; P.view.oy = down.oy + dy * devicePixelRatio; draw(); }
  });
  canvas.addEventListener("pointerup", (e) => { if (down && !down.moved && P.ds) addClick(e); down = null; });
  canvas.addEventListener("contextmenu", (e) => { e.preventDefault(); if (P.ds) removeNearest(e); });
  canvas.addEventListener("dblclick", () => { if (P.img) { fitView(); draw(); } });
  canvas.addEventListener("wheel", (e) => {
    if (!P.img) return;
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    const sx = (e.clientX - r.left) * devicePixelRatio, sy = (e.clientY - r.top) * devicePixelRatio;
    const k = Math.exp(-e.deltaY * 0.0015);
    P.view.ox = sx - (sx - P.view.ox) * k;   // Keep the image point under the pointer in place
    P.view.oy = sy - (sy - P.view.oy) * k;
    P.view.scale *= k;
    draw();
  }, { passive: false });
  window.addEventListener("resize", () => { if (P.img) { fitView(); draw(); } });

  // Keys: 1 and 2 switch the click kind, Ctrl/Cmd+Z undoes (not while typing in a text field)
  window.addEventListener("keydown", (e) => {
    if (e.target.matches("input[type=text], select")) return;
    if (e.key === "1") setMode("pos");
    else if (e.key === "2") setMode("neg");
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); }
  });
}

async function init() {
  takeTokenFromUrl();
  try { if (localStorage.getItem("boneseg-theme") === "light") document.documentElement.dataset.theme = "light"; } catch (_) { /* storage blocked */ }
  bind();
  setMode("pos");
  try {
    const health = await api("/api/health");
    $("pathRow").classList.toggle("hidden", !health.allow_paths);   // Opening by path is off on shared servers
    await showRecent();
  } catch (e) { toast(`Could not reach the server: ${e.message}`, true); }
}

init();

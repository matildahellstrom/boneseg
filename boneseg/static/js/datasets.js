"use strict";
// boneseg front end: Loading, uploading and browsing datasets.
// Plain scripts sharing globals, loaded in order by index.html. No build step.

// ---------------------------------------------------------------------------------------------
// Datasets
async function refreshDatasets(selectId) {
  S.datasets = await api("/api/datasets");
  const list = $("datasetList");
  list.innerHTML = "";
  for (const d of S.datasets) {
    const el = document.createElement("div");
    el.className = "dataset-item" + (S.ds?.id === d.id ? " active" : "");
    el.innerHTML = `<span class="name" title="${esc(d.name)}">${esc(d.name)}</span><span class="small muted">${d.n_z}z·${d.n_channels}c</span><button title="Remove from the app">✕</button>`;
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
    // Panels that belonged to the closed image
    for (const el of ["sidePanel", "jobBox", "histoCards", "histoDownloads"]) $(el)?.classList.add("hidden");
    S.lastJob = null;
    S.job = null;
    S.base = null;
    draw();
  }
}

async function openDataset(id, view = {}) {
  await flushSaves();   // Clicks on the previous dataset are saved before its state is replaced
  const d = await api(`/api/datasets/${id}`);
  S.ds = d;
  S.points = {};
  S.history = [];
  S.result = null;
  S.c = 0;
  S.z = Math.floor(d.n_z / 2);
  $("datasetTitle").textContent = `${d.name} · ${d.width}×${d.height} px · ${d.n_z} slices · ${d.voxel_size_known ? `${d.voxel_um[2].toFixed(3)} µm/px` : "pixel size unknown"}`;
  const cs = $("channelSelect");
  cs.innerHTML = d.channel_names.map((n, i) => `<option value="${i}">${i}: ${esc(n)}</option>`).join("");
  const rs = $("refSelect");
  rs.innerHTML = `<option value="">None</option>` + d.channel_names.map((n, i) => `<option value="${i}">${i}: ${esc(n)}</option>`).join("");
  rs.value = d.reference_channel ?? "";
  $("notesInput").value = d.notes || "";
  $("overlaySelect").innerHTML = `<option value="">None</option>` + d.channel_names.map((n, i) => `<option value="${i}">${i}: ${esc(n)}</option>`).join("");
  $("refHint").textContent = d.reference_channel != null
    ? (d.reference_guessed ? "Guessed from the channel name. Every result is scored against it." : "Every result is scored against this expert mask.")
    : "Pick a channel that holds an expert mask to score results with Dice.";
  // Start on a channel that is not the reference mask
  if (d.default_channel != null) S.c = d.default_channel;
  else if (d.rgb_channel != null) S.c = d.rgb_channel;   // Colour images start in colour, which DINOv2 reads best
  else if (d.reference_channel === 0 && d.n_channels > 1) S.c = 1;
  cs.value = S.c;
  $("zSlider").max = d.n_z - 1;
  $("zStart").max = $("zEnd").max = d.n_z - 1;
  $("zStart").value = 0;
  $("zEnd").value = d.n_z - 1;
  $("zStep").value = Math.max(1, Math.round(d.n_z / 50));
  showEmpty(false);
  $("resultsSection").classList.add("hidden");
  stopEditing();
  try { S.points = await api(`/api/datasets/${id}/annotations`); } catch (_) { S.points = {}; }
  // Channel and slice: from the link if given, else the first slice that has clicks, else the middle
  if (view.c != null && view.c >= 0 && view.c < d.n_channels) { S.c = view.c; cs.value = S.c; }
  const clicked = firstClickedSlice(S.points, S.c, d.n_z);
  S.z = view.z != null && view.z >= 0 && view.z < d.n_z ? view.z : clicked ?? Math.floor(d.n_z / 2);
  rememberDataset(id);
  $("jobBox").classList.add("hidden");   // A stack run of the previous dataset is not this one's
  S.job = null;
  applySettings(d.settings);
  S.structures = structuresFromAnnotations(S.points);
  S.active = 0;
  renderStructures();
  await refreshDatasets();
  await refreshLabels();
  await refreshHead();
  fillHistoControls();
  histoDefaults();
  cancelRoi();
  showVoxel();
  S.sideY = null;
  S.lastJob = null;
  if ($("showSide").checked) loadSide();
  $("histoCards").classList.add("hidden");
  $("histoDownloads").classList.add("hidden");
  await loadPlane(true);
  resumeJobs(id);
}

// After a reload or in a second tab, a stack run or fine-tune still going on this image is shown again,
// with its progress and Cancel button
async function resumeJobs(id) {
  let running = [];
  try { running = await api(`/api/datasets/${id}/jobs?status=running`); } catch (_) { return; }
  if (S.ds?.id !== id) return;
  const stack = running.find((j) => j.kind === "stack" && !j.meta.batch);
  if (stack) {
    S.job = stack.id;
    $("jobBox").classList.remove("hidden");
    $("cancelBtn").classList.remove("hidden");
    $("stackBtn").disabled = true;
    pollJob(stack.id);
  }
  const ft = running.find((j) => j.kind === "finetune");
  if (ft) {
    S.finetuneJob = ft.id;
    $("finetuneBox").open = true;
    $("finetuneJob").classList.remove("hidden");
    $("finetuneBtn").disabled = true;
    $("finetuneCancel").classList.remove("hidden");
    pollFinetune(ft.id);
  }
}

// Each slice request gets a number; a response that is not the newest is dropped, so stepping quickly through
// slices cannot leave an older slice on screen (cached images can arrive before newer, slower ones)
let planeSeq = 0;
async function loadPlane(fit = false) {
  if (!S.ds) return;
  const seq = ++planeSeq;
  $("zSlider").value = S.z;
  $("zValue").textContent = `${S.z} / ${S.ds.n_z - 1}${S.ds.voxel_size_known ? ` · ${(S.z * S.ds.voxel_um[0]).toFixed(1)} µm` : ""}`;
  $("contrastValue").textContent = `${S.low}–${S.high}%`;
  const ov = $("overlaySelect").value;
  const url = `/api/datasets/${S.ds.id}/plane?c=${S.c}&z=${S.z}&low=${S.low}&high=${S.high}`
    + (ov !== "" && +ov !== S.c ? `&overlay=${ov}&color=${$("overlayColor").value.slice(1)}` : "");
  canvas.setAttribute("aria-label", `${S.ds.name}, channel ${S.c} (${S.ds.channel_names[S.c]}), slice ${S.z} of ${S.ds.n_z - 1}. Click to add points.`);
  busy(true, "Loading slice…");
  let img;
  try {
    img = await loadImage(url);
  } catch (e) { if (seq === planeSeq) toast(e.message, true); return; } finally { if (seq === planeSeq) busy(false); }
  if (seq !== planeSeq) return;
  S.base = img;
  S.layers = {};
  S.result = null;
  $("resultsSection").classList.add("hidden");
  await loadReference();
  await loadLabelLayer();
  if (seq !== planeSeq) return;   // A newer slice was requested while the overlays loaded
  stopEditing();
  if (fit) fitView();
  updateCounts();
  renderLabels();
  draw();
  if (S.method === "learned" || pts().pos.length || $("profileSelect").value) scheduleSegment(0);
  placeSideZ();
}

async function loadReference() {
  if (!S.ds || S.ds.reference_channel == null) { delete S.layers.ref; return; }
  const z = S.z, ds = S.ds.id;
  let img = null;
  try { img = await loadImage(`/api/datasets/${ds}/reference?z=${z}`); } catch (_) { /* no reference on this slice */ }
  if (S.z !== z || S.ds?.id !== ds) return;   // The user moved to another slice meanwhile
  if (img) S.layers.ref = img; else delete S.layers.ref;
}

function uploadFile(file) {
  // The raw file is the request body, so the server writes it once without a temporary copy
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
  xhr.open("POST", `/api/datasets/stream?filename=${encodeURIComponent(file.name)}`);
  xhr.setRequestHeader("Content-Type", "application/octet-stream");
  xhr.send(file);
}

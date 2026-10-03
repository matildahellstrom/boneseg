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
}

async function loadPlane(fit = false) {
  if (!S.ds) return;
  $("zSlider").value = S.z;
  $("zValue").textContent = `${S.z} / ${S.ds.n_z - 1}${S.ds.voxel_size_known ? ` · ${(S.z * S.ds.voxel_um[0]).toFixed(1)} µm` : ""}`;
  $("contrastValue").textContent = `${S.low}–${S.high}%`;
  const ov = $("overlaySelect").value;
  const url = `/api/datasets/${S.ds.id}/plane?c=${S.c}&z=${S.z}&low=${S.low}&high=${S.high}`
    + (ov !== "" && +ov !== S.c ? `&overlay=${ov}&color=${$("overlayColor").value.slice(1)}` : "");
  canvas.setAttribute("aria-label", `${S.ds.name}, channel ${S.c} (${S.ds.channel_names[S.c]}), slice ${S.z} of ${S.ds.n_z - 1}. Click to add points.`);
  busy(true, "Loading slice…");
  try {
    S.base = await loadImage(url);
  } catch (e) { toast(e.message, true); return; } finally { busy(false); }
  S.layers = {};
  S.result = null;
  $("resultsSection").classList.add("hidden");
  await loadReference();
  await loadLabelLayer();
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
  try { S.layers.ref = await loadImage(`/api/datasets/${S.ds.id}/reference?z=${S.z}`); } catch (_) { delete S.layers.ref; }
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

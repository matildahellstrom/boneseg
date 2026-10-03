"use strict";
// boneseg front end: Wiring controls and starting the app.
// Plain scripts sharing globals, loaded in order by index.html. No build step.

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

  $("channelSelect").onchange = async (e) => { S.c = +e.target.value; stopEditing(); renderLabels(); await refreshHead(); loadPlane(); if ($("showSide").checked) loadSide(); };
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
      fillHistoControls();
      draw();
      if (S.result) scheduleSegment(0);
    } catch (err) { toast(err.message, true); }
  };

  $("modePos").onclick = () => setMode("pos");
  $("modeNeg").onclick = () => setMode("neg");
  $("undoBtn").onclick = undo;
  $("clearBtn").onclick = () => { S.points[key()] = { pos: [], neg: [] }; persistClicks(); S.layers.mask = S.layers.heat = S.layers.unc = null; S.result = null; $("resultsSection").classList.add("hidden"); updateCounts(); draw(); };
  $("autoNegBtn").onclick = autoBackground;
  $("profileSelect").onchange = () => {
    $("deleteProfileBtn").classList.toggle("hidden", !$("profileSelect").value);
    $("downloadProfileBtn").classList.toggle("hidden", !$("profileSelect").value);
    const prof = S.profiles.find((p) => p.id === $("profileSelect").value);
    if (prof && prof.backbone !== $("backboneSelect").value) { $("backboneSelect").value = prof.backbone; syncSettingLabels(); }
    if (prof?.kind === "learned" && prof.vit_size) $("vitSize").value = prof.vit_size;
    if (prof || pts().pos.length) scheduleSegment(0);
  };
  $("exportHeadBtn").onclick = () => {
    S.profileFromHead = true;
    $("profileName").value = "";
    $("profileDesc").value = S.ds ? `Learned from ${S.head.trained_on.length} labelled slices of ${S.ds.name}` : "";
    $("profileDialog").showModal();
  };
  $("saveProfileBtn").onclick = () => {
    S.profileFromHead = false;
    if (!pts().pos.length) { toast("Add object clicks first"); return; }
    $("profileName").value = "";
    $("profileDesc").value = S.ds ? `${S.ds.channel_names[S.c]}` : "";
    $("profileDialog").showModal();
  };
  $("profileSaveConfirm").onclick = saveProfile;
  $("downloadProfileBtn").onclick = () => { window.location = `/api/profiles/${$("profileSelect").value}/download`; };
  $("importProfile").onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    try {
      const prof = await api("/api/profiles/import", { method: "POST", body: fd });
      await refreshProfiles(prof.id);
      toast(`Imported profile ${prof.name}`);
      $("profileSelect").dispatchEvent(new Event("change"));
    } catch (err) { toast(err.message, true); }
    e.target.value = "";
  };
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
  for (const id of ["showMask", "showHeat", "showUnc", "showRef", "showErr", "showPoints", "opacity"]) $(id).addEventListener("input", draw);
  $("fitBtn").onclick = fitView;
  document.querySelectorAll("[data-export]").forEach((b) => {
    b.onclick = () => {
      const kind = b.dataset.export;
      const base = `/api/datasets/${S.ds.id}/export`;
      window.location = kind === "csv" ? `${base}/objects.csv?c=${S.c}&z=${S.z}` : `${base}/mask?c=${S.c}&z=${S.z}&fmt=${kind}`;
    };
  });
  $("stackBtn").onclick = runStack;
  $("editBtn").onclick = () => (S.editing ? stopEditing() : startEditing());
  $("saveLabelBtn").onclick = () => saveLabel(S.editing);
  $("editSave").onclick = () => saveLabel(true);
  $("editCancel").onclick = stopEditing;
  $("trainBtn").onclick = trainHead;
  $("methodClicks").onclick = () => { setMethod("clicks"); if (pts().pos.length || $("profileSelect").value) scheduleSegment(0); };
  $("methodLearned").onclick = () => { setMethod("learned"); scheduleSegment(0); };
  $("showLabel").addEventListener("input", draw);
  $("showHisto").addEventListener("input", draw);
  $("showSide").addEventListener("input", loadSide);
  $("sideImg").onload = placeSideZ;
  $("sideImg").parentElement.onclick = (ev) => {
    const r = $("sideImg").getBoundingClientRect();
    const z = Math.floor(((ev.clientY - r.top) / r.height) * S.ds.n_z);
    S.z = Math.max(0, Math.min(S.ds.n_z - 1, z));
    loadPlane();
  };
  $("histoBtn").onclick = runHisto;
  $("roiBtn").onclick = () => (S.roiDraft ? finishRoi() : startRoi());
  $("roiClear").onclick = () => saveRoi(null);
  $("voxelSave").onclick = saveVoxel;
  $("studyBtn").onclick = openStudy;
  $("studyMetric").onchange = loadStudy;
  $("reportBtn").onclick = () => window.open(`/api/datasets/${S.ds.id}/report?c=${S.c}&z=${S.z}`, "_blank");
  $("hContact").oninput = () => { $("hContactValue").textContent = `${$("hContact").value} µm`; };
  $("histoCsv").onclick = () => { window.location = `/api/datasets/${S.ds.id}/histomorphometry/cells.csv?z=${S.histoZ ?? S.z}`; };
  $("cancelBtn").onclick = () => S.job && api(`/api/jobs/${S.job}/cancel`, { method: "POST" });
  $("helpBtn").onclick = () => $("helpDialog").showModal();

  window.addEventListener("keydown", (e) => {
    if (e.target.matches("input[type=text], input[type=number], select, textarea") || document.querySelector("dialog[open]")) return;
    const k = e.key.toLowerCase();
    if (k === " ") { S.space = true; e.preventDefault(); }
    else if (k === "1") setMode("pos");
    else if (k === "2") setMode("neg");
    else if ((e.ctrlKey || e.metaKey) && k === "z") { e.preventDefault(); S.editing ? undoStroke() : undo(); }
    else if (S.editing && (k === "[" || k === "]")) {
      const b = $("brush");
      b.value = Math.max(+b.min, Math.min(+b.max, +b.value * (k === "]" ? 1.25 : 0.8)));
      draw();
    }
    else if (k === ",") stepZ(-1);
    else if (k === ".") stepZ(1);
    else if (k === "f") fitView();
    else if (k === "m") toggle("showMask");
    else if (k === "h") toggle("showHeat");
    else if (k === "u") toggle("showUnc");
    else if (k === "?") $("helpDialog").showModal();
    else if (k === "e") (S.editing ? stopEditing() : startEditing());
    else if (k === "escape" && S.editing) stopEditing();
    else if (k === "escape" && S.roiDraft) cancelRoi();
    else if (k === "enter" && S.roiDraft) finishRoi();
    else if (k === "r") (S.roiDraft ? finishRoi() : startRoi());
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
    if (!S.health.allow_paths) $("pathInput").closest("details").classList.add("hidden");
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

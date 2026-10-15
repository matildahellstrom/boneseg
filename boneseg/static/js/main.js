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

  $("channelSelect").onchange = async (e) => {
    S.c = +e.target.value;
    if (S.ds) api(`/api/datasets/${S.ds.id}`, { method: "PATCH", body: { default_channel: S.c } }).then((d) => { S.ds.default_channel = d.default_channel; }).catch(() => {}); stopEditing(); renderLabels(); await refreshHead(); loadPlane(); };
  $("zSlider").oninput = (e) => { S.z = +e.target.value; $("zValue").textContent = S.z; };
  $("zSlider").onchange = () => loadPlane();
  $("zPrev").onclick = () => stepZ(-1);
  $("zNext").onclick = () => stepZ(1);

  $("modePos").onclick = () => setMode("pos");
  $("modeNeg").onclick = () => setMode("neg");
  $("undoBtn").onclick = undo;
  $("clearBtn").onclick = () => { S.points[key()] = { pos: [], neg: [] }; S.history = forgetHistory(S.history, key()); persistClicks(); S.layers.mask = S.layers.heat = S.layers.unc = S.layers.err = null; S.result = null; $("resultsSection").classList.add("hidden"); updateCounts(); draw(); };
  $("autoNegBtn").onclick = autoBackground;

  $("minObj").addEventListener("input", syncSettingLabels);
  $("minObj").addEventListener("change", () => { rememberSettings(); if (S.result) scheduleSegment(0); });
  for (const id of ["showMask", "showRef", "showErr"]) $(id).addEventListener("input", draw);
  $("fitBtn").onclick = fitView;
  document.querySelectorAll("[data-export]").forEach((b) => {
    b.onclick = () => {
      const kind = b.dataset.export;
      const base = `/api/datasets/${S.ds.id}/export`;
      window.location = `${base}/mask?c=${S.c}&z=${S.z}&fmt=${kind}`;
    };
  });
  $("stackBtn").onclick = runStack;
  $("editBtn").onclick = () => (S.editing ? stopEditing() : startEditing());
  $("saveLabelBtn").onclick = () => saveLabel(S.editing);
  $("editSave").onclick = () => saveLabel(true);
  $("editCancel").onclick = stopEditing;
  $("trainBtn").onclick = trainHead;
  $("refLabelsBtn").onclick = async () => {
    busy(true, "Saving labels from the reference…");
    try {
      const out = await api(`/api/datasets/${S.ds.id}/labels/from-reference`, { method: "POST", body: { channel: S.c, n: 5 } });
      S.labels = out.labels;
      renderLabels(); await loadLabelLayer(); draw();
      toast(`Labelled slices ${out.saved.join(", ")} from the reference. Train the model next.`);
    } catch (e) { toast(e.message, true); } finally { busy(false); }
  };
  $("methodClicks").onclick = () => { setMethod("clicks"); if (pts().pos.length) scheduleSegment(0); };
  $("methodLearned").onclick = () => { setMethod("learned"); scheduleSegment(0); };
  $("projectBtn").onclick = () => { window.location = `/api/datasets/${S.ds.id}/export/project.zip`; };
  $("themeBtn").onclick = () => {
    const light = document.documentElement.dataset.theme !== "light";
    document.documentElement.dataset.theme = light ? "light" : "dark";
    try { localStorage.setItem("boneseg-theme", light ? "light" : "dark"); } catch (_) { /* storage unavailable */ }
  };
  $("cancelBtn").onclick = () => S.job && api(`/api/jobs/${S.job}/cancel`, { method: "POST" });
  $("helpBtn").onclick = () => $("helpDialog").showModal();

  window.addEventListener("keydown", (e) => {
    if (e.target.matches("input[type=text], input[type=number], select, textarea") || document.querySelector("dialog[open]")) return;  // Typing in a field never triggers shortcuts
    const k = e.key.toLowerCase();
    if (k === " ") {
      // Space pans the image, except on a focused button, checkbox, link or section header, where it must press it
      if (e.target.closest("button, a, summary, input, [role=button], [tabindex]")) return;
      S.space = true; e.preventDefault();
    }
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
    else if (k === "?") $("helpDialog").showModal();
    else if (k === "e") (S.editing ? stopEditing() : startEditing());
    else if (k === "escape" && S.editing) stopEditing();
  });
  window.addEventListener("keyup", (e) => { if (e.key === " ") S.space = false; });
}

function toggle(id) { $(id).checked = !$(id).checked; draw(); }
function stepZ(d) {
  if (!S.ds) return;
  const z = Math.min(S.ds.n_z - 1, Math.max(0, S.z + d));
  if (z !== S.z) { S.z = z; loadPlane(); }
}

// Panel sections fold away when their heading is clicked; the choice is remembered per browser
function bindCollapsibleSections() {
  let folded = [];
  try { folded = JSON.parse(localStorage.getItem("boneseg-folded") || "[]"); } catch (_) { /* storage unavailable */ }
  document.querySelectorAll(".panel section[id]").forEach((sec) => {
    const h = sec.querySelector(":scope > h2");
    if (!h) return;
    h.setAttribute("role", "button");
    h.tabIndex = 0;
    const apply = () => h.setAttribute("aria-expanded", String(!sec.classList.contains("collapsed")));
    if (folded.includes(sec.id)) sec.classList.add("collapsed");
    apply();
    const toggleSec = () => {
      sec.classList.toggle("collapsed");
      apply();
      const now = [...document.querySelectorAll(".panel section.collapsed[id]")].map((x) => x.id);
      try { localStorage.setItem("boneseg-folded", JSON.stringify(now)); } catch (_) { /* storage unavailable */ }
    };
    h.addEventListener("click", toggleSec);
    h.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggleSec(); } });
  });
}

// A shared access link carries ?token=…; it is kept as a cookie for the API and removed from the address bar
async function init() {
  takeTokenFromUrl();
  bindCollapsibleSections();
  try { if (localStorage.getItem("boneseg-theme") === "light") document.documentElement.dataset.theme = "light"; } catch (_) { /* storage unavailable */ }
  bind();
  showEmpty(true);
  try {
    S.health = await api("/api/health");
    $("version").textContent = `v${S.health.version}`;
    if (!S.health.allow_paths) $("pathInput").closest("details").classList.add("hidden");
    S.backbone = S.health.default_settings.backbone;
    syncSettingLabels();
    await refreshDatasets();
    const view = wantedView(S.datasets);
    if (view) openDataset(view.id, view);
    else if (S.datasets.length) openDataset(S.datasets[0].id);
  } catch (e) {
    toast(`Could not reach the server: ${e.message}`, true);
  }
}

init();

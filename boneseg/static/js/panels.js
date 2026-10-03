"use strict";
// boneseg front end: Side view, sample comparison, pixel size, region of interest and histomorphometry.
// Plain scripts sharing globals, loaded in order by index.html. No build step.

// ---------------------------------------------------------------------------------------------
// Side view
let sideSeq = 0;
async function loadSide() {
  const on = $("showSide").checked && S.ds && S.ds.n_z > 1;
  $("sidePanel").classList.toggle("hidden", !on);
  if (!on) { draw(); return; }
  if (S.sideY == null) S.sideY = Math.round(S.ds.height / 2);
  const seq = ++sideSeq;
  const job = S.lastJob && S.lastJob.ds === S.ds.id && S.lastJob.c === S.c ? `&job_id=${S.lastJob.id}` : "";
  try {
    const img = await loadImage(`/api/datasets/${S.ds.id}/xz?c=${S.c}&y=${S.sideY}&low=${S.low}&high=${S.high}${job}`);
    if (seq !== sideSeq) return;
    $("sideImg").src = img.src;
    $("sideLabel").textContent = `Side view at row ${S.sideY}${job ? " · cyan: latest stack run" : " · run the stack to see its mask here"}`;
    placeSideZ();
  } catch (e) { toast(e.message, true); }
  draw();
}

function placeSideZ() {
  const img = $("sideImg");
  if (!img.complete || !S.ds) return;
  $("sideZ").style.top = `${((S.z + 0.5) / S.ds.n_z) * img.clientHeight - 1}px`;
}

// ---------------------------------------------------------------------------------------------
// Comparing samples between groups
const GROUP_COLORS = ["#00c8f0", "#ffa53a", "#22d27a", "#c78bff", "#ff5a6e"];

async function openStudy() {
  await loadStudy();
  $("studyDialog").showModal();
}

async function loadStudy() {
  const metric = $("studyMetric").value || "mean_area_fraction";
  let out;
  try { out = await api(`/api/study?metric=${metric}`); } catch (e) { toast(e.message, true); return; }
  if (!$("studyMetric").options.length) {
    $("studyMetric").innerHTML = Object.entries(out.metrics).map(([k, l]) => `<option value="${k}">${l}</option>`).join("");
    $("studyMetric").value = metric;
  }
  const unit = out.unit ? ` (${out.unit})` : "";
  $("studyValueHead").textContent = out.label + unit;
  const val = (v) => (v == null ? "–" : fmt(v * out.scale, Number.isInteger(v * out.scale) ? 0 : 2));
  const groups = [...new Set(out.rows.map((r) => r.group).filter(Boolean))];
  $("groupNames").innerHTML = groups.map((g) => `<option value="${esc(g)}">`).join("");
  $("studyRows").innerHTML = out.rows.map((r) => `<tr><td title="${esc(r.name)}">${esc(r.name.length > 34 ? r.name.slice(0, 32) + "…" : r.name)}</td>
      <td><input type="text" list="groupNames" value="${esc(r.group)}" data-id="${esc(r.dataset_id)}" placeholder="group" aria-label="Group of ${esc(r.name)}"></td>
      <td class="num">${r.has_stack_run ? val(r[metric]) : '<span class="muted">no stack run</span>'}</td></tr>`).join("");
  $("studyRows").querySelectorAll("input").forEach((inp) => {
    inp.onchange = async () => {
      try { await api(`/api/datasets/${inp.dataset.id}`, { method: "PATCH", body: { group: inp.value.trim() || null } }); loadStudy(); }
      catch (e) { toast(e.message, true); }
    };
  });
  drawStudyPlot(out, groups, metric);
  const c = out.comparison;
  $("studyTest").innerHTML = c.test
    ? `<b>${c.test}</b>: p = ${c.p_value < 0.001 ? c.p_value.toExponential(1) : c.p_value.toFixed(3)}${c.effect ? ` · ${c.effect}` : ""}. ${c.note || ""}`
    : c.note;
}

function drawStudyPlot(out, groups, metric) {
  const svg = $("studyPlot");
  const pts = out.rows.filter((r) => r.group && r[metric] != null);
  if (!groups.length || !pts.length) { svg.innerHTML = `<text x="160" y="110" text-anchor="middle" fill="#8fa0bd" font-size="12">Assign groups to see the plot</text>`; return; }
  const W = 320, H = 220, padL = 46, padB = 28, padT = 12;
  const vals = pts.map((r) => r[metric] * out.scale);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi === lo) { hi += 1; lo -= 1; }
  const span = hi - lo; lo -= span * 0.1; hi += span * 0.1;
  const sy = (v) => padT + (1 - (v - lo) / (hi - lo)) * (H - padT - padB);
  const bw = (W - padL) / groups.length;
  let g = `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" stroke="#22314f"/>`;
  for (const t of [lo + 0.1 * (hi - lo), (lo + hi) / 2, hi - 0.1 * (hi - lo)]) {
    g += `<text x="${padL - 4}" y="${sy(t) + 3}" text-anchor="end" font-size="10" fill="#8fa0bd">${fmt(t, 2)}</text><line x1="${padL}" x2="${W}" y1="${sy(t)}" y2="${sy(t)}" stroke="#16223a"/>`;
  }
  groups.forEach((name, i) => {
    const cx = padL + bw * (i + 0.5);
    const color = GROUP_COLORS[i % GROUP_COLORS.length];
    const gv = pts.filter((r) => r.group === name).map((r) => r[metric] * out.scale);
    gv.forEach((v, j) => { g += `<circle cx="${cx + (j - (gv.length - 1) / 2) * 7}" cy="${sy(v)}" r="4.5" fill="${color}" fill-opacity=".85"><title>${fmt(v, 3)}</title></circle>`; });
    const med = out.comparison.groups[name]?.median;
    if (med != null) g += `<line x1="${cx - bw * 0.28}" x2="${cx + bw * 0.28}" y1="${sy(med * out.scale)}" y2="${sy(med * out.scale)}" stroke="#e6edf7" stroke-width="2"/>`;
    g += `<text x="${cx}" y="${H - 10}" text-anchor="middle" font-size="11" fill="#e6edf7">${esc(name)} (n=${gv.length})</text>`;
  });
  svg.innerHTML = g;
}

// ---------------------------------------------------------------------------------------------
// Pixel size
function showVoxel() {
  const d = S.ds;
  $("pxInput").value = d.voxel_um[2];
  $("pzInput").value = d.voxel_um[0];
  $("voxelSummary").textContent = d.voxel_size_known ? `Pixel size ${d.voxel_um[2].toFixed(3)} µm` : "Pixel size unknown, set it for measurements in µm";
  $("voxelSummary").style.color = d.voxel_size_known ? "" : "var(--warn)";
  $("voxelHint").textContent = d.voxel_um_override ? "Set by you." : d.voxel_size_known ? "Read from the file." : "The file has no pixel size, so measurements are in pixels until you set it.";
  $("datasetTitle").textContent = `${d.name} · ${d.width}×${d.height} px · ${d.n_z} slices · ${d.voxel_size_known ? `${d.voxel_um[2].toFixed(3)} µm/px` : "pixel size unknown"}`;
}

async function saveVoxel() {
  const px = +$("pxInput").value, pz = +$("pzInput").value || 1;
  if (!(px > 0)) { toast("Enter a positive pixel size"); return; }
  try {
    S.ds = await api(`/api/datasets/${S.ds.id}`, { method: "PATCH", body: { voxel_um_override: [pz, px, px] } });
    showVoxel();
    toast(`Pixel size set to ${px} µm`);
    if (S.result) scheduleSegment(0);
  } catch (e) { toast(e.message, true); }
}

// ---------------------------------------------------------------------------------------------
// Region of interest
function startRoi() {
  if (!S.base) return;
  stopEditing();
  S.roiDraft = [];
  $("roiBtn").textContent = "Finish region";
  $("roiHint").textContent = "Click the corners. Double-click or press Enter to finish, Esc to cancel.";
  draw();
}

function cancelRoi() {
  S.roiDraft = null;
  S.roiCursor = null;
  $("roiBtn").textContent = "Draw region";
  updateRoiHint();
  draw();
}

async function finishRoi() {
  // A double-click also adds two corners at the same spot, so drop near-duplicates
  const pts = S.roiDraft.filter((p, i, a) => i === 0 || Math.hypot(p[0] - a[i - 1][0], p[1] - a[i - 1][1]) > 2);
  if (pts.length < 3) { toast("A region needs at least three corners"); return; }
  await saveRoi(pts);
  cancelRoi();
}

async function saveRoi(polygon) {
  try {
    const out = await api(`/api/datasets/${S.ds.id}/roi`, { method: "PUT", body: { polygon } });
    S.ds.roi = out.roi;
    S.roiArea = out.area_um2;
    updateRoiHint();
    draw();
    if (S.method === "learned" || pts().pos.length || $("profileSelect").value) scheduleSegment(0);
  } catch (e) { toast(e.message, true); }
}

function updateRoiHint() {
  const has = S.ds?.roi && S.ds.roi.length >= 3;
  $("roiClear").classList.toggle("hidden", !has);
  $("roiHint").textContent = has ? `Measurements, masks and exports are limited to the region${S.roiArea ? ` (${fmt(S.roiArea / 1e6, 3)} mm²)` : ""}.` : "Measurements cover the whole image.";
}

// ---------------------------------------------------------------------------------------------
// Histomorphometry
function fillHistoControls() {
  if (!S.ds) return;
  const chans = S.ds.channel_names.map((n, i) => `<option value="${i}">${i}: ${esc(n)}</option>`).join("");
  for (const id of ["hBoneC", "hCellC"]) {
    const cur = $(id).value;
    $(id).innerHTML = chans;
    if (cur !== "" && +cur < S.ds.n_channels) $(id).value = cur;
  }
  const structs = S.structures.length > 1 ? S.structures.map((st, k) => `<option value="structure:${k}">Structure: ${esc(st.name)}</option>`).join("") : "";
  const sources = `<option value="current">Current result</option>${structs}<option value="learned">Learned model</option><option value="label">Saved label</option>`
    + (S.ds.reference_channel != null ? `<option value="reference">Reference channel</option>` : "")
    + S.profiles.map((p) => `<option value="profile:${esc(p.id)}">Profile: ${esc(p.name)}</option>`).join("");
  for (const id of ["hBoneSrc", "hCellSrc"]) { const cur = $(id).value; $(id).innerHTML = sources; if ([...$(id).options].some((o) => o.value === cur)) $(id).value = cur; }
}

function histoDefaults() {
  // Guess bone and cell channels from their names
  const names = S.ds.channel_names.map((n) => n.toLowerCase());
  const bone = names.findIndex((n) => /bone|matrix|col|autofl/.test(n));
  const cell = names.findIndex((n) => /trap|osteoclast|ctsk|cell/.test(n));
  $("hBoneC").value = bone >= 0 ? bone : 0;
  $("hCellC").value = cell >= 0 ? cell : S.c;
}

function histoFromStructures() {
  // With several structures on this channel, use the one named like bone for bone and the first other one for cells
  if (S.structures.length < 2) return;
  fillHistoControls();
  const k = S.structures.findIndex((st) => /bone|matrix/i.test(st.name));
  const boneK = k >= 0 ? k : 1;
  const cellK = boneK === 0 ? 1 : 0;
  $("hBoneC").value = S.c; $("hCellC").value = S.c;
  $("hBoneSrc").value = `structure:${boneK}`; $("hCellSrc").value = `structure:${cellK}`;
}

async function runHisto() {
  const spec = (c, src) => src.startsWith("profile:") ? { channel: +$(c).value, source: "profile", profile_id: src.slice(8) } : { channel: +$(c).value, source: src };
  busy(true, "Measuring…");
  try {
    const out = await api(`/api/datasets/${S.ds.id}/histomorphometry`, {
      method: "POST",
      body: { z: S.z, bone: spec("hBoneC", $("hBoneSrc").value), cells: spec("hCellC", $("hCellSrc").value), contact_um: +$("hContact").value, settings: settings() },
    });
    S.layers.histo = await loadImage(out.overlay_png);
    S.histoZ = S.z;
    const m = out.summary;
    const cards = [
      ["B.Ar/T.Ar", `${m["B.Ar/T.Ar_%"].toFixed(1)}%`],
      ["B.Pm", `${m["B.Pm_mm"].toFixed(2)} mm`],
      ["Oc.Pm/B.Pm", `${m["Oc.Pm/B.Pm_%"].toFixed(1)}%`],
      ["N.Oc/B.Pm", `${m["N.Oc/B.Pm_per_mm"].toFixed(1)} /mm`],
      ["Cells on bone", `${m["N.Oc"]} of ${m.cells_total}`],
      ["Median distance", `${fmt(m.median_distance_to_bone_um)} µm`],
    ];
    $("histoCards").innerHTML = cards.map(([k, v]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
    $("histoCards").classList.remove("hidden");
    $("histoDownloads").classList.remove("hidden");
    $("showHisto").checked = true;
    draw();
    status("Histomorphometry overlay: white is bone surface, magenta is surface covered by cells, cyan are the cells");
  } catch (e) { toast(e.message, true); } finally { busy(false); }
}

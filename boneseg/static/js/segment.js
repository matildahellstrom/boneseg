"use strict";
// boneseg front end: Segmentation requests, results and clicks.
// Plain scripts sharing globals, loaded in order by index.html. No build step.

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
  if (S.method === "clicks" && !profile && isMulti()) { runMulti(); return; }
  if (S.method === "clicks" && !p.pos.length && !profile) { toast("Click the structure you want first, or pick a profile"); return; }
  const seq = ++S.seq;
  busy(true, S.result ? "Updating…" : "Computing features…");
  try {
    const out = await api(`/api/datasets/${S.ds.id}/segment`, {
      method: "POST",
      body: { method: S.method, channel: S.c, z: S.z, pos: p.pos, neg: p.neg, profile_id: profile, settings: settings(), uncertainty: $("uncToggle").checked },
    });
    if (seq !== S.seq) return; // A newer request is on its way
    const [mask, heat, unc, err] = await Promise.all([
      loadImage(out.mask_png), loadImage(out.heat_png), out.uncertainty_png ? loadImage(out.uncertainty_png) : null,
      out.error_png ? loadImage(out.error_png) : null,
    ]);
    if (seq !== S.seq) return;
    S.layers.mask = mask;
    S.layers.heat = heat;
    if (unc) S.layers.unc = unc; else delete S.layers.unc;
    if (err) S.layers.err = err; else delete S.layers.err;
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
  $("labelsExport").classList.add("hidden");
  $("editBtn").classList.remove("hidden");
  $("saveLabelBtn").classList.remove("hidden");
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
  if (ev) $("evalBox").innerHTML = `<b>Against ${ev.against === "your saved label" ? "your saved label" : "the reference mask"}</b><br>Dice ${ev.dice.toFixed(3)} · IoU ${ev.iou.toFixed(3)} · HD95 ${fmt(ev.hd95_um)} µm`
    + `<br><span class="small">Extra ${fmt(ev.false_positive_um2)} µm² · missed ${fmt(ev.false_negative_um2)} µm² · <a href="#" id="errLink">show errors</a></span>`;
  if (ev) $("errLink").onclick = (e) => { e.preventDefault(); $("showErr").checked = !$("showErr").checked; draw(); };
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
  const src = { clicks: "from your clicks", "carried over": "from the profile", "learned (probability 0.5)": "learned model, probability 0.5", top_percent: "fixed share", manual: "manual", otsu: "Otsu" }[out.threshold_source] || out.threshold_source;
  updateHint();
  $("timing").textContent = `Threshold ${out.threshold.toFixed(3)} (${src}) · features ${out.timing.embed_s.toFixed(2)} s · total ${out.timing.total_s.toFixed(2)} s`;
}

async function runMulti() {
  const p = pts();
  const structures = S.structures.map((st, k) => ({ name: st.name, color: st.color, pos: posList(p, k) }));
  if (!structures.some((st) => st.pos.length)) { toast("Click examples of at least one structure"); return; }
  const seq = ++S.seq;
  busy(true, "Segmenting structures…");
  try {
    const out = await api(`/api/datasets/${S.ds.id}/segment_multi`, {
      method: "POST", body: { channel: S.c, z: S.z, structures, neg: p.neg, settings: settings() },
    });
    if (seq !== S.seq) return;
    S.layers.mask = await loadImage(out.labels_png);
    delete S.layers.heat; delete S.layers.unc; delete S.layers.err;
    S.result = { multi: true, ...out };
    showMultiResults(out);
    draw();
    status(`Segmented ${out.structures.length} structures on slice ${S.z} in ${out.timing.total_s.toFixed(2)} s`);
  } catch (e) {
    if (seq === S.seq) toast(e.message, true);
  } finally {
    if (seq === S.seq) busy(false);
  }
}

function showMultiResults(out) {
  $("resultsSection").classList.remove("hidden");
  $("statCards").innerHTML = out.structures.map((st) => `<div class="card" style="border-left:3px solid ${st.color}"><div class="k">${st.name}</div>
    <div class="v">${(100 * st.area_fraction).toFixed(1)}%</div><div class="k">${st.n_objects} objects · ${fmt(st.area_um2)} µm²</div></div>`).join("");
  $("evalBox").classList.add("hidden");
  $("suggestionBox").classList.add("hidden");
  $("labelsExport").classList.remove("hidden");
  // Brush corrections and labels work on a single mask, so they are hidden for several structures
  $("editBtn").classList.add("hidden");
  $("saveLabelBtn").classList.add("hidden");
  histoFromStructures();
  $("timing").textContent = "Each pixel goes to the structure it resembles most, if it clears that structure's threshold. Stack runs, labels and histomorphometry use the first structure.";
  updateHint();
}

function addPoint(kind, y, x) {
  const list = kind === "pos" ? posList(pts(), S.active) : pts().neg;
  list.push([Math.round(y), Math.round(x)]);
  S.history.push({ key: key(), kind, struct: kind === "pos" ? S.active : null });
  persistClicks();
  updateCounts();
  draw();
  if ($("autoRun").checked) scheduleSegment();
}

function removeNearest(y, x) {
  const p = pts();
  let best = null;
  const lists = [p.neg, ...S.structures.map((_, k) => posList(p, k))];
  lists.forEach((list) => list.forEach(([py, px], i) => {
    const d = (py - y) ** 2 + (px - x) ** 2;
    if (!best || d < best.d) best = { list, i, d };
  }));
  if (!best) return;
  best.list.splice(best.i, 1);
  persistClicks();
  updateCounts();
  draw();
  if ($("autoRun").checked && (p.pos.length || $("profileSelect").value)) scheduleSegment();
}

function undo() {
  const h = S.history.pop();
  if (!h) return;
  const p = S.points[h.key];
  if (p) (h.kind === "pos" ? posList(p, h.struct || 0) : p.neg).pop();
  persistClicks(h.key);
  updateCounts();
  draw();
  if ($("autoRun").checked && (pts().pos.length || $("profileSelect").value)) scheduleSegment();
}

function updateCounts() {
  const p = pts();
  $("nPos").textContent = posList(p, S.active).length;
  $("nNeg").textContent = p.neg.length;
  updateHint();
}

// One short suggestion for what to do next, based on where the user is
function updateHint() {
  const p = pts();
  const profile = $("profileSelect").value;
  let hint = "";
  if (!S.ds) hint = "";
  else if (S.method === "learned") hint = "The learned model segments every slice you open. Correct and save more labels to improve it.";
  else if (!p.pos.length && !profile) hint = "Click two or three examples of the structure you want to segment.";
  else if (!p.neg.length && !profile) hint = "Now Shift-click a few background spots. The threshold is placed between your object and background clicks.";
  else if (S.result && S.result.uncertain_fraction > 0.05) hint = "Orange areas depend on single clicks. Click inside the dashed ring to settle the most uncertain one.";
  else if (S.result && !S.result.evaluation) hint = "Looks stable. Run the whole stack, save the clicks as a profile, or press E to correct the mask.";
  else if (S.result) hint = "Turn on Errors to see where the mask disagrees with the reference.";
  $("nextHint").textContent = hint;
}

function autoBackground() {
  if (!S.ds) return;
  const { height: h, width: w } = S.ds;
  const m = 0.02;
  for (const [fy, fx] of [[m, m], [m, 0.5], [m, 1 - m], [0.5, m], [0.5, 1 - m], [1 - m, m], [1 - m, 0.5], [1 - m, 1 - m]]) {
    pts().neg.push([Math.round(fy * h), Math.round(fx * w)]);
    S.history.push({ key: key(), kind: "neg" });
  }
  persistClicks();
  updateCounts();
  draw();
  if ($("autoRun").checked && pts().pos.length) scheduleSegment();
}

// ---------------------------------------------------------------------------------------------
// Structures: several things to segment on the same slice
function renderStructures() {
  const box = $("structList");
  box.innerHTML = "";
  S.structures.forEach((st, k) => {
    const b = document.createElement("button");
    b.className = "struct" + (k === S.active ? " active" : "");
    b.title = k === S.active ? "Object clicks go to this structure" : "Click to send object clicks to this structure";
    b.innerHTML = `<span class="sw" style="background:${st.color}"></span>${st.name}${k > 0 ? '<span class="x" title="Remove this structure">✕</span>' : ""}`;
    b.onclick = (e) => {
      if (e.target.classList.contains("x")) { removeStructure(k); return; }
      S.active = k; renderStructures(); updateCounts();
    };
    box.appendChild(b);
  });
  const add = document.createElement("button");
  add.className = "struct add";
  add.textContent = "+ Structure";
  add.title = "Segment another structure on the same slice, such as a second cell type";
  add.onclick = () => {
    const name = prompt("Name of the new structure", `Structure ${S.structures.length + 1}`);
    if (!name) return;
    S.structures.push({ name: name.trim().slice(0, 40), color: STRUCT_COLORS[S.structures.length % STRUCT_COLORS.length] });
    S.active = S.structures.length - 1;
    renderStructures(); updateCounts(); fillHistoControls();
  };
  box.appendChild(add);
  $("modePos").lastChild.textContent = isMulti() ? S.structures[S.active].name : "Object";
  $("modePos").querySelector(".dot").style.background = S.structures[S.active].color;
}

function removeStructure(k) {
  if (!confirm(`Remove ${S.structures[k].name} and its clicks on every slice?`)) return;
  S.structures.splice(k, 1);
  for (const [kk, p] of Object.entries(S.points)) {
    if (p.extra && p.extra.length >= k) { p.extra.splice(k - 1, 1); persistClicks(kk); }
  }
  S.active = Math.min(S.active, S.structures.length - 1);
  renderStructures(); updateCounts(); draw();
  if (S.result) scheduleSegment(0);
}

function structuresFromAnnotations(points) {
  // Rebuild the structure list from saved clicks, in the order they were added
  const out = [{ name: "Object", color: STRUCT_COLORS[0] }];
  for (const p of Object.values(points)) {
    (p.extra || []).forEach((e, i) => { if (!out[i + 1] && e.name) out[i + 1] = { name: e.name, color: e.color || STRUCT_COLORS[(i + 1) % STRUCT_COLORS.length] }; });
  }
  return out.filter(Boolean);
}

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

function addPoint(kind, y, x) {
  pts()[kind].push([Math.round(y), Math.round(x)]);
  S.history.push({ key: key(), kind });
  persistClicks();
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
  persistClicks();
  updateCounts();
  draw();
  if ($("autoRun").checked && (p.pos.length || $("profileSelect").value)) scheduleSegment();
}

function undo() {
  const h = S.history.pop();
  if (!h) return;
  const p = S.points[h.key];
  p?.[h.kind].pop();
  persistClicks(h.key);
  updateCounts();
  draw();
  if ($("autoRun").checked && (pts().pos.length || $("profileSelect").value)) scheduleSegment();
}

function updateCounts() {
  const p = pts();
  $("nPos").textContent = p.pos.length;
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

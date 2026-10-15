"use strict";
// boneseg front end: side view and sample comparison.
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
  const job = S.lastJob && S.lastJob.ds === S.ds.id && S.lastJob.c === S.c
    ? `&job_id=${S.lastJob.id}${isMulti() ? `&colors=${encodeURIComponent(S.structures.map((st) => st.color).join(","))}` : ""}` : "";
  try {
    const img = await loadImage(`/api/datasets/${S.ds.id}/xz?c=${S.c}&y=${S.sideY}&low=${S.low}&high=${S.high}${job}`);
    if (seq !== sideSeq) return;
    $("sideImg").src = img.src;
    $("sideLabel").textContent = `Side view at row ${S.sideY}${job ? (isMulti() ? " · latest stack run, in the structures' colours" : " · cyan: latest stack run") : " · run the stack to see its mask here"}`;
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
  $("batchProfile").innerHTML = S.profiles.length
    ? S.profiles.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join("")
    : `<option value="">Save a profile first</option>`;
  $("batchChannel").value = S.c;
  await loadStudy();
  $("studyDialog").showModal();
}

async function runBatch() {
  const pid = $("batchProfile").value;
  if (!pid) { toast("Save a profile first"); return; }
  $("batchRun").disabled = true;
  try {
    const channel = $("batchChannelMode").value === "own" ? null : +$("batchChannel").value;
    const job = await api("/api/batch", { method: "POST", body: { profile_id: pid, channel, z_step: +$("batchStep").value, settings: settings() } });
    const poll = async () => {
      const j = await api(`/api/jobs/${job.id}`);
      $("batchBar").style.width = `${100 * j.progress}%`;
      $("batchText").textContent = j.status === "running" || j.status === "queued" ? `Running ${j.message || "…"}` : "";
      if (j.status === "running" || j.status === "queued") { setTimeout(poll, 800); return; }
      $("batchRun").disabled = false;
      if (j.status === "failed") { $("batchText").textContent = `Failed: ${j.error}`; return; }
      const samples = j.result.samples || [];
      const failed = samples.filter((s) => s.status === "failed");
      const skipped = samples.filter((s) => s.status === "skipped");
      const stopped = samples.filter((s) => s.status === "cancelled");
      $("batchText").textContent = `${j.status === "cancelled" ? "Cancelled" : "Done"}: ${samples.filter((s) => s.status === "done").length} samples`
        + (stopped.length ? `, ${stopped.length} stopped part-way (not used in the comparison)` : "")
        + (skipped.length ? `, ${skipped.length} skipped (${skipped.map((s) => s.error).filter((v, i, a) => a.indexOf(v) === i).join(", ")})` : "")
        + (failed.length ? `, ${failed.length} failed (${failed.map((f) => f.name).join(", ")})` : "");
      loadStudy();
    };
    poll();
  } catch (e) { $("batchText").textContent = e.message; toast(e.message, true); $("batchRun").disabled = false; }
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
      <td class="num">${r[metric] != null ? val(r[metric]) : `<span class="muted">${r.has_stack_run ? "–" : "no stack run"}</span>`}</td></tr>`).join("");
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

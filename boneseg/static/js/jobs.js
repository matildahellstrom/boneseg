"use strict";
// boneseg front end: Profiles and whole-stack runs.
// Plain scripts sharing globals, loaded in order by index.html. No build step.

// ---------------------------------------------------------------------------------------------
// Profiles
async function refreshProfiles(selectId) {
  S.profiles = await api("/api/profiles");
  const sel = $("profileSelect");
  const current = selectId ?? sel.value;
  sel.innerHTML = `<option value="">None, use my clicks</option>` +
    S.profiles.map((p) => `<option value="${esc(p.id)}" title="${esc(p.description || p.source)}">${esc(p.name)} · ${p.kind === "learned" ? "learned model" : esc(p.backbone)}</option>`).join("");
  sel.value = S.profiles.some((p) => p.id === current) ? current : "";
  $("deleteProfileBtn").classList.toggle("hidden", !sel.value);
  $("downloadProfileBtn").classList.toggle("hidden", !sel.value);
  fillHistoControls();
}

async function saveProfile() {
  const name = $("profileName").value.trim();
  if (!name) { toast("Give the profile a name"); return; }
  const p = pts();
  try {
    const prof = S.profileFromHead
      ? await api(`/api/datasets/${S.ds.id}/head/export`, { method: "POST", body: { channel: S.c, name, description: $("profileDesc").value } })
      : await api("/api/profiles", {
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
  const multi = S.method === "clicks" && !profile && isMulti();
  const structures = multi ? S.structures.map((st, k) => ({ name: st.name, color: st.color, pos: posList(p, k) })) : [];
  if (S.method === "clicks" && !p.pos.length && !profile && !multi) { toast("Click the structure on this slice first, or pick a profile"); return; }
  try {
    const job = await api(`/api/datasets/${S.ds.id}/stack`, {
      method: "POST",
      body: {
        method: S.method, channel: S.c, ref_z: S.z, pos: p.pos, neg: p.neg, profile_id: profile, settings: settings(),
        z_start: +$("zStart").value, z_end: +$("zEnd").value, z_step: +$("zStep").value, structures,
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
  if (s.structures) {
    // Several structures: one line per structure
    const colorOf = (name) => safeColor(S.structures.find((st) => st.name === name)?.color);
    $("jobText").innerHTML = `${job.status === "cancelled" ? "Cancelled after" : "Done:"} ${s.n_slices || 0} slices`
      + Object.entries(s.structures).map(([name, st]) => `<br><span style="color:${colorOf(name)}">●</span> ${esc(name)}: volume ${fmt(st.volume_um3)} µm³ · mean area ${(100 * (st.mean_area_fraction || 0)).toFixed(1)}% · ${st.n_objects_3d} objects in 3D`).join("");
    drawChart(job.result.slices || [], colorOf);
    if (job.status === "done") { S.lastJob = { id, ds: job.meta.dataset_id, c: job.meta.channel }; if ($("showSide").checked) loadSide(); }
    $("jobDownloads").innerHTML = ["labels.tif", "masks.tif", "slices.csv", "objects_3d.csv", "summary.json"]
      .map((f) => `<a class="small" href="/api/jobs/${id}/files/${f}" download><button class="ghost">${f}</button></a>`).join("");
    return;
  }
  $("jobText").innerHTML = `${job.status === "cancelled" ? "Cancelled after" : "Done:"} ${s.n_slices || 0} slices · volume ${fmt(s.volume_um3)} µm³ · mean area ${(100 * (s.mean_area_fraction || 0)).toFixed(1)}%`
    + (s.n_objects_3d != null ? ` · ${s.n_objects_3d} objects in 3D (${s.n_objects_3d_inside} not cut by the stack ends), median ${fmt(s.median_object_volume_um3)} µm³` : "")
    + (s.mean_dice_vs_reference != null ? ` · mean Dice ${s.mean_dice_vs_reference.toFixed(3)}` : "");
  drawChart(job.result.slices || []);
  if (job.status === "done") { S.lastJob = { id, ds: job.meta.dataset_id, c: job.meta.channel }; if ($("showSide").checked) loadSide(); }
  $("jobDownloads").innerHTML = ["masks.tif", "labels_3d.tif", "slices.csv", "objects_3d.csv", "summary.json"]
    .map((f) => `<a class="small" href="/api/jobs/${id}/files/${f}" download><button class="ghost">${f}</button></a>`).join("");
}

function drawChart(rows, colorOf = null) {
  const svg = $("stackChart");
  if (!rows.length) { svg.innerHTML = ""; return; }
  const W = 300, H = 120, pad = 8;
  const zs = [...new Set(rows.map((r) => r.z))];
  const z0 = Math.min(...zs), z1 = Math.max(...zs);
  const sx = (z) => pad + ((z - z0) / Math.max(1, z1 - z0)) * (W - 2 * pad);
  const maxA = Math.max(...rows.map((r) => r.area_fraction), 1e-6);
  // Each series: [rows, value key, colour, label, scale]
  let series;
  if (colorOf) {
    const names = [...new Set(rows.map((r) => r.structure))];
    series = names.map((n) => [rows.filter((r) => r.structure === n), "area_fraction", colorOf(n), esc(n), maxA]);
  } else {
    series = [[rows, "area_fraction", "#00c8f0", "Area fraction", maxA]];
    if (rows[0].dice != null) series.push([rows, "dice", "#22d27a", "Dice vs reference", 1]);
  }
  let out = "";
  for (const [rs, k, color, , scale] of series) {
    const d = rs.map((r, i) => `${i ? "L" : "M"}${sx(r.z).toFixed(1)},${(H - pad - (r[k] / scale) * (H - 2 * pad)).toFixed(1)}`).join("");
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
  $("chartLegend").innerHTML = series.map(([, , c, label]) => `<span style="color:${c}">━</span> ${label}`).join(" · ")
    + ` · area peaks at ${(100 * maxA).toFixed(1)}% · click the chart to jump to a slice`;
}

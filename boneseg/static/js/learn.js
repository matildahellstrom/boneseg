"use strict";
// boneseg front end: Brush corrections, labels and the learned model.
// Plain scripts sharing globals, loaded in order by index.html. No build step.

// ---------------------------------------------------------------------------------------------
// Correcting masks with a brush, labels and the learned model
function startEditing() {
  if (!S.base) return;
  const w = S.base.naturalWidth, h = S.base.naturalHeight;
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const g = c.getContext("2d");
  // Start from the current mask, or the saved label if there is no result yet
  const src = S.layers.mask || S.layers.labelFill;
  const multi = isMulti() && S.result?.multi;
  if (src) {
    g.drawImage(src, 0, 0, w, h);
    const d = g.getImageData(0, 0, w, h);
    for (let i = 0; i < d.data.length; i += 4) {
      const on = d.data[i + 3] > 0;
      // With several structures each keeps its colour; a single mask is painted in yellow
      if (!multi) { d.data[i] = 255; d.data[i + 1] = 216; d.data[i + 2] = 74; }
      d.data[i + 3] = on ? 255 : 0;
    }
    g.putImageData(d, 0, 0);
  }
  S.edit = c;
  S.editHistory = [];
  S.editing = true;
  $("editBar").classList.remove("hidden");
  $("viewer").classList.add("editing");
  draw();
}

function undoStroke() {
  const snap = S.editHistory?.pop();
  if (!snap || !S.edit) return;
  S.edit.getContext("2d").putImageData(snap, 0, 0);
  draw();
}

function stopEditing() {
  S.editing = false;
  S.edit = null;
  S.brushAt = null;
  $("editBar")?.classList.add("hidden");
  $("viewer")?.classList.remove("editing");
  draw();
}

function paintAt(ev, erase, last) {
  const r = canvas.getBoundingClientRect();
  const toImg = (e) => [(e.clientX - r.left - S.view.ox) / S.view.scale, (e.clientY - r.top - S.view.oy) / S.view.scale];
  const g = S.edit.getContext("2d");
  g.globalCompositeOperation = erase ? "destination-out" : "source-over";
  g.strokeStyle = g.fillStyle = isMulti() && S.result?.multi ? safeColor(S.structures[S.active].color) : "rgb(255,216,74)";
  g.lineCap = "round";
  g.lineWidth = +$("brush").value;
  const [x, y] = toImg(ev);
  g.beginPath();
  if (last) { const [lx, ly] = toImg(last); g.moveTo(lx, ly); g.lineTo(x, y); g.stroke(); }
  else { g.arc(x, y, g.lineWidth / 2, 0, Math.PI * 2); g.fill(); }
  g.globalCompositeOperation = "source-over";
  draw();
}

// A painted canvas with structure colours becomes an index map: red channel = structure number
function editToIndexPng() {
  const w = S.edit.width, h = S.edit.height;
  const src = S.edit.getContext("2d").getImageData(0, 0, w, h).data;
  const cols = S.structures.map((st) => { const c = safeColor(st.color).slice(1); const f = c.length === 3 ? c.split("").map((x) => x + x).join("") : c; return [0, 2, 4].map((i) => parseInt(f.slice(i, i + 2), 16)); });
  const out = document.createElement("canvas");
  out.width = w; out.height = h;
  const og = out.getContext("2d");
  const img = og.createImageData(w, h);
  for (let i = 0; i < src.length; i += 4) {
    if (src[i + 3] < 128) continue;
    let best = 0, bd = Infinity;
    cols.forEach(([r, g, b], k) => { const d = (src[i] - r) ** 2 + (src[i + 1] - g) ** 2 + (src[i + 2] - b) ** 2; if (d < bd) { bd = d; best = k; } });
    img.data[i] = img.data[i + 1] = img.data[i + 2] = best + 1;
    img.data[i + 3] = 255;
  }
  og.putImageData(img, 0, 0);
  return out.toDataURL("image/png");
}

async function saveLabel(fromEdit) {
  try {
    const body = { channel: S.c, z: S.z };
    const multi = isMulti() && S.result?.multi;
    if (multi) body.structures = S.structures.map((st) => st.name);
    if (fromEdit && S.edit) body.mask_png = multi ? editToIndexPng() : S.edit.toDataURL("image/png");
    else if (!S.result) { toast("Segment the slice or correct a mask first"); return; }
    const out = await api(`/api/datasets/${S.ds.id}/labels`, { method: "POST", body });
    S.labels = out.labels;
    stopEditing();
    await loadLabelLayer();
    renderLabels();
    draw();
    toast(`Saved label for slice ${S.z}. ${S.labels.filter((l) => l.channel === S.c).length} labelled on this channel.`
      + (fromEdit ? "" : " Labels teach best when corrected with the brush (E) first."));
  } catch (e) { toast(e.message, true); }
}

async function loadLabelLayer() {
  delete S.layers.label;
  delete S.layers.labelFill;
  if (!S.ds || !S.labels.some((l) => l.channel === S.c && l.z === S.z)) return;
  try { S.layers.label = await loadImage(`/api/datasets/${S.ds.id}/labels/png?c=${S.c}&z=${S.z}&t=${Date.now()}`); S.layers.labelFill = S.layers.label; } catch (_) { /* none */ }
}

async function refreshLabels() {
  if (!S.ds) return;
  try { S.labels = await api(`/api/datasets/${S.ds.id}/labels`); } catch (_) { S.labels = []; }
  renderLabels();
}

function renderLabels() {
  const mine = S.labels.filter((l) => l.channel === S.c);
  $("refLabelsBtn").classList.toggle("hidden", !(S.ds && S.ds.reference_channel != null && S.ds.reference_channel !== S.c));
  const box = $("labelList");
  box.innerHTML = mine.length ? "" : `<span class="muted small">No labels on this channel yet.</span>`;
  for (const l of mine) {
    const b = document.createElement("button");
    b.className = "chip" + (l.z === S.z ? " active" : "");
    b.innerHTML = `z ${l.z}<span class="x" title="Delete label">✕</span>`;
    b.onclick = async (e) => {
      if (e.target.classList.contains("x")) {
        if (!confirm(`Delete the label on slice ${l.z}?`)) return;
        S.labels = (await api(`/api/datasets/${S.ds.id}/labels?c=${S.c}&z=${l.z}`, { method: "DELETE" })).labels;
        await loadLabelLayer(); renderLabels(); draw();
        return;
      }
      S.z = l.z; loadPlane();
    };
    box.appendChild(b);
  }
  $("trainBtn").disabled = !mine.length;
  $("trainBtn").textContent = mine.length ? `Train model on ${mine.length} label${mine.length > 1 ? "s" : ""}` : "Train model";
}

async function refreshHead() {
  S.head = null;
  if (S.ds) { try { S.head = (await api(`/api/datasets/${S.ds.id}/head?c=${S.c}`)) || null; } catch (_) { S.head = null; } }
  $("methodLearned").disabled = !S.head;
  if (!S.head && S.method === "learned") setMethod("clicks");
  showHeadInfo();
}

function showHeadInfo() {
  const h = S.head;
  if (h?.names?.length) {
    // A model of several structures brings its structures along, keeping colours for names already present
    const prev = Object.fromEntries(S.structures.map((st) => [st.name, st.color]));
    S.structures = h.names.map((n, k) => ({ name: n, color: safeColor(prev[n] || STRUCT_COLORS[k % STRUCT_COLORS.length]) }));
    S.active = Math.min(S.active, S.structures.length - 1);
    renderStructures();
  }
  $("exportHeadBtn").classList.toggle("hidden", !h);
  if (!h) { $("trainResult").textContent = ""; return; }
  const cv = h.cv[h.cv.chosen];
  $("trainResult").innerHTML = `Model trained on ${h.trained_on.length} slice${h.trained_on.length > 1 ? "s" : ""} (${h.kind === "mlp" ? "small neural network" : "linear"}${h.context > 1 ? ", with neighbourhood features" : ""}).`
    + (cv ? ` Dice on held-out labelled slices: <b>${cv.mean_dice.toFixed(3)}</b>, measured against your labels.` : " Label a second slice to estimate how well it generalizes.")
    + (h.reference_check ? ` Against the reference mask on ${h.reference_check.n_slices} unlabelled slices: <b>${h.reference_check.mean_dice.toFixed(3)}</b>.` : "");
}

async function trainHead() {
  $("trainBtn").disabled = true;
  busy(true, "Training…");
  try {
    S.head = await api(`/api/datasets/${S.ds.id}/head`, { method: "POST", body: { channel: S.c, settings: settings() } });
    $("methodLearned").disabled = false;
    showHeadInfo();
    toast(`Trained in ${S.head.seconds} s`);
    setMethod("learned");
    scheduleSegment(0);
  } catch (e) { toast(e.message, true); } finally { busy(false); renderLabels(); }
}

function setMethod(m) {
  S.method = m;
  $("methodClicks").classList.toggle("active", m === "clicks");
  $("methodLearned").classList.toggle("active", m === "learned");
  updateHint();
}

// ---------------------------------------------------------------------------------------------
// Fine-tuning DINOv2 on the labels, as a background job; the result becomes a backbone
function fillBackbones() {
  const sel = $("backboneSelect"), current = sel.value;
  sel.innerHTML = S.health.backbones.map((b) => `<option value="${esc(b.id)}">${esc(b.label)}${b.ready ? "" : " · downloads on first use"}</option>`).join("");
  if (S.health.backbones.some((b) => b.id === current)) sel.value = current;
}

async function startFinetune() {
  const n = S.labels.filter((l) => l.channel === S.c).length;
  if (n < 2) { toast("Save at least 2 labels on this channel first (5 or more work best)"); return; }
  try {
    const job = await api(`/api/datasets/${S.ds.id}/finetune`, { method: "POST", body: { channel: S.c, settings: settings() } });
    S.finetuneJob = job.id;
    $("finetuneJob").classList.remove("hidden");
    $("finetuneBtn").disabled = true;
    $("finetuneCancel").classList.remove("hidden");
    pollFinetune(job.id);
  } catch (e) { toast(e.message, true); }
}

async function pollFinetune(id) {
  let job;
  try { job = await api(`/api/jobs/${id}`); } catch (e) { toast(e.message, true); return; }
  $("finetuneBar").style.width = `${100 * job.progress}%`;
  if (job.status === "running" || job.status === "queued") {
    $("finetuneText").textContent = job.message || "Starting…";
    setTimeout(() => pollFinetune(id), 1500);
    return;
  }
  $("finetuneBtn").disabled = false;
  $("finetuneCancel").classList.add("hidden");
  if (job.status === "failed") { $("finetuneText").textContent = `Failed: ${job.error}`; return; }
  const r = job.result;
  const better = r.val_dice_finetuned > r.val_dice_frozen;
  $("finetuneText").innerHTML = `${job.status === "cancelled" ? "Stopped early; kept the best step so far." : "Done."} Trained on ${r.n_train} label${r.n_train > 1 ? "s" : ""}, checked on ${r.n_val}.`
    + ` Dice on the check slices: fine-tuned <b>${r.val_dice_finetuned.toFixed(3)}</b>, learned model on the unchanged DINOv2 <b>${r.val_dice_frozen.toFixed(3)}</b>.`
    + (better ? " Fine-tuning helped." : " Fine-tuning did not help here; more labels may change that.")
    + ` <button class="ghost" id="useFinetuned">Use it as the backbone</button>`
    + `<br><span class="muted">Saved as ${esc(r.file)}. Learned models and profiles made with another backbone need to be trained or made again.</span>`;
  S.health = await api("/api/health");
  fillBackbones();
  $("useFinetuned").onclick = () => {
    $("backboneSelect").value = r.backbone;
    syncSettingLabels(); rememberSettings();
    toast("Switched to the fine-tuned backbone. The first click on each slice computes its features again.");
    if (S.result) scheduleSegment(0);
  };
}

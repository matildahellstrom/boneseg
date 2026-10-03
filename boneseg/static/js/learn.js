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
  if (src) {
    g.drawImage(src, 0, 0, w, h);
    const d = g.getImageData(0, 0, w, h);
    for (let i = 0; i < d.data.length; i += 4) {
      const on = d.data[i + 3] > 0;
      d.data[i] = 255; d.data[i + 1] = 216; d.data[i + 2] = 74; d.data[i + 3] = on ? 255 : 0;
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
  g.strokeStyle = g.fillStyle = "rgb(255,216,74)";
  g.lineCap = "round";
  g.lineWidth = +$("brush").value;
  const [x, y] = toImg(ev);
  g.beginPath();
  if (last) { const [lx, ly] = toImg(last); g.moveTo(lx, ly); g.lineTo(x, y); g.stroke(); }
  else { g.arc(x, y, g.lineWidth / 2, 0, Math.PI * 2); g.fill(); }
  g.globalCompositeOperation = "source-over";
  draw();
}

async function saveLabel(fromEdit) {
  try {
    const body = { channel: S.c, z: S.z };
    if (fromEdit && S.edit) body.mask_png = S.edit.toDataURL("image/png");
    else if (!S.result) { toast("Segment the slice or correct a mask first"); return; }
    const out = await api(`/api/datasets/${S.ds.id}/labels`, { method: "POST", body });
    S.labels = out.labels;
    stopEditing();
    await loadLabelLayer();
    renderLabels();
    draw();
    toast(`Saved label for slice ${S.z}. ${S.labels.filter((l) => l.channel === S.c).length} labelled on this channel.`);
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
  $("exportHeadBtn").classList.toggle("hidden", !h);
  if (!h) { $("trainResult").textContent = ""; return; }
  const cv = h.cv[h.cv.chosen];
  $("trainResult").innerHTML = `Model trained on ${h.trained_on.length} slice${h.trained_on.length > 1 ? "s" : ""} (${h.kind === "mlp" ? "small neural network" : "linear"}).`
    + (cv ? ` Estimated Dice on unseen slices: <b>${cv.mean_dice.toFixed(3)}</b>.` : " Label a second slice to estimate how well it generalizes.");
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
}

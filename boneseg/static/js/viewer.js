"use strict";
// boneseg front end: Drawing the image and overlays, zoom, pan and mouse input.
// Plain scripts sharing globals, loaded in order by index.html. No build step.

// ---------------------------------------------------------------------------------------------
// Viewer: drawing, zoom and pan
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

function fullToDisp() { return S.base ? S.base.naturalWidth / S.ds.width : 1; }

function fitView() {
  if (!S.base) return;
  const r = canvas.getBoundingClientRect();
  const s = Math.min(r.width / S.base.naturalWidth, r.height / S.base.naturalHeight) * 0.96;
  S.view = { scale: s, ox: (r.width - S.base.naturalWidth * s) / 2, oy: (r.height - S.base.naturalHeight * s) / 2 };
  S.fitted = true;  // Stays fitted on window resizes until the user zooms or pans
  draw();
}

function draw() {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) {
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  if (!S.base) return;
  const { scale, ox, oy } = S.view;
  const w = S.base.naturalWidth * scale;
  const h = S.base.naturalHeight * scale;
  ctx.imageSmoothingEnabled = scale < 2;
  ctx.drawImage(S.base, ox, oy, w, h);
  const op = +$("opacity").value;
  const layer = (img, alpha) => { if (img) { ctx.globalAlpha = alpha; ctx.drawImage(img, ox, oy, w, h); ctx.globalAlpha = 1; } };
  if ($("showHeat").checked) layer(S.layers.heat, op);
  if ($("showUnc").checked) layer(S.layers.unc, Math.min(1, op + 0.2));
  if (S.editing && S.edit) layer(S.edit, 0.5);
  else if ($("showErr").checked && S.layers.err) layer(S.layers.err, Math.min(1, op + 0.3));
  else if ($("showMask").checked) layer(S.layers.mask, Math.min(1, op + 0.3));
  if ($("showRef").checked) layer(S.layers.ref, 0.9);
  if ($("showLabel").checked && !S.editing) layer(S.layers.label, 0.95);
  if ($("showHisto").checked && !S.editing && S.histoZ === S.z) layer(S.layers.histo, 1);
  $("zoomLabel").textContent = `${Math.round(scale * fullToDisp() * 100)}%`;

  const f = fullToDisp();
  const toScreen = ([y, x]) => [ox + x * f * scale, oy + y * f * scale];
  const roi = S.ds?.roi;
  if (roi && roi.length >= 3) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, oy, w, h);
    roi.forEach((p, i) => { const [sx, sy] = toScreen(p); i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy); });
    ctx.closePath();
    ctx.fillStyle = "rgba(5, 9, 16, 0.55)";
    ctx.fill("evenodd");
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "#e6edf7";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    roi.forEach((p, i) => { const [sx, sy] = toScreen(p); i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy); });
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }
  if (S.roiDraft) {
    ctx.save();
    ctx.strokeStyle = "#ffd84a";
    ctx.fillStyle = "#ffd84a";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    S.roiDraft.forEach((p, i) => { const [sx, sy] = toScreen(p); i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy); });
    if (S.roiCursor) ctx.lineTo(S.roiCursor[0], S.roiCursor[1]);
    ctx.stroke();
    for (const p of S.roiDraft) { const [sx, sy] = toScreen(p); ctx.fillRect(sx - 3, sy - 3, 6, 6); }
    ctx.restore();
  }
  if (S.editing && S.brushAt) {
    ctx.beginPath();
    ctx.arc(S.brushAt[0], S.brushAt[1], (+$("brush").value / 2) * scale, 0, Math.PI * 2);
    ctx.strokeStyle = S.brushErase ? "#ff5a6e" : "#ffd84a";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  if ($("showPoints").checked && !S.editing) {
    const p = S.points[key()] || { pos: [], neg: [] };
    const groups = [[p.neg, "#ff5a6e"], ...S.structures.map((st, k) => [posList(p, k), st.color])];
    for (const [list, color] of groups) {
      for (const pt of list) {
        const [sx, sy] = toScreen(pt);
        ctx.beginPath();
        ctx.arc(sx, sy, 5.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#0b1220";
        ctx.stroke();
      }
    }
  }
  if ($("showSide").checked && S.sideY != null) {
    const [, sy] = toScreen([S.sideY, 0]);
    ctx.save();
    ctx.setLineDash([8, 6]);
    ctx.strokeStyle = "#ffa53a";
    ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(ox, sy); ctx.lineTo(ox + w, sy); ctx.stroke();
    ctx.restore();
  }
  const sug = S.result?.suggestion;
  if (sug && $("uncToggle").checked) {
    const [sx, sy] = toScreen(sug);
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ffa53a";
    ctx.beginPath();
    ctx.arc(sx, sy, 14, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function screenToFull(ev) {
  const r = canvas.getBoundingClientRect();
  const f = fullToDisp();
  const x = (ev.clientX - r.left - S.view.ox) / S.view.scale / f;
  const y = (ev.clientY - r.top - S.view.oy) / S.view.scale / f;
  return [y, x];
}

function inside([y, x]) { return S.ds && y >= 0 && x >= 0 && y < S.ds.height && x < S.ds.width; }

let drag = null;
canvas.addEventListener("mousedown", (ev) => {
  if (!S.base) return;
  drag = { x: ev.clientX, y: ev.clientY, ox: S.view.ox, oy: S.view.oy, moved: false, button: ev.button, pan: ev.button === 1 || S.space };
  if (drag.pan) $("viewer").classList.add("panning");
  if (S.editing && !drag.pan) {
    // Snapshot for undo, at most 30 strokes back
    const g = S.edit.getContext("2d");
    S.editHistory.push(g.getImageData(0, 0, S.edit.width, S.edit.height));
    if (S.editHistory.length > 30) S.editHistory.shift();
    drag.paint = true; drag.erase = ev.shiftKey || ev.button === 2; paintAt(ev, drag.erase, null); drag.last = ev;
  }
});
window.addEventListener("mousemove", (ev) => {
  if (S.base && ev.target === canvas) {
    const [y, x] = screenToFull(ev);
    $("cursorInfo").textContent = inside([y, x])
      ? `x ${Math.round(x)} · y ${Math.round(y)}${S.ds.voxel_size_known ? ` · ${(x * S.ds.voxel_um[2]).toFixed(1)}, ${(y * S.ds.voxel_um[1]).toFixed(1)} µm` : ""}`
      : "";
  }
  if (S.roiDraft && ev.target === canvas) {
    const r = canvas.getBoundingClientRect();
    S.roiCursor = [ev.clientX - r.left, ev.clientY - r.top];
    if (!drag) draw();
  }
  if (S.editing && ev.target === canvas) {
    const r = canvas.getBoundingClientRect();
    S.brushAt = [ev.clientX - r.left, ev.clientY - r.top];
    S.brushErase = ev.shiftKey;
    if (!drag) draw();
  }
  if (!drag) return;
  if (drag.paint) { paintAt(ev, drag.erase, drag.last); drag.last = ev; return; }
  const dx = ev.clientX - drag.x;
  const dy = ev.clientY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 5) drag.moved = true;
  if (drag.moved && (drag.pan || drag.button === 0)) {
    S.view.ox = drag.ox + dx;
    S.view.oy = drag.oy + dy;
    S.fitted = false;
    $("viewer").classList.add("panning");
    draw();
  }
});
window.addEventListener("mouseup", (ev) => {
  if (!drag) return;
  const d = drag;
  drag = null;
  $("viewer").classList.remove("panning");
  if (d.paint) return;
  if (d.moved || d.pan || ev.target !== canvas) return;
  const p = screenToFull(ev);
  if (!inside(p)) return;
  if (ev.altKey && d.button === 0) { S.sideY = Math.round(p[0]); $("showSide").checked = true; loadSide(); draw(); return; }
  if (S.roiDraft) {
    if (d.button === 0) { S.roiDraft.push([Math.round(p[0]), Math.round(p[1])]); draw(); }
    return;
  }
  if (d.button === 2) { removeNearest(...p); return; }
  if (d.button !== 0) return;
  const kind = ev.shiftKey ? (S.mode === "pos" ? "neg" : "pos") : S.mode;
  addPoint(kind, ...p);
});
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("dblclick", () => { if (S.roiDraft) finishRoi(); });
canvas.addEventListener("wheel", (ev) => {
  if (!S.base) return;
  ev.preventDefault();
  const r = canvas.getBoundingClientRect();
  const mx = ev.clientX - r.left;
  const my = ev.clientY - r.top;
  const factor = Math.exp(-ev.deltaY * 0.0015);
  const ns = Math.min(40, Math.max(0.05, S.view.scale * factor));
  S.view.ox = mx - ((mx - S.view.ox) * ns) / S.view.scale;
  S.view.oy = my - ((my - S.view.oy) * ns) / S.view.scale;
  S.view.scale = ns;
  S.fitted = false;
  draw();
}, { passive: false });
window.addEventListener("resize", () => { if (S.fitted) fitView(); else draw(); placeSideZ(); });

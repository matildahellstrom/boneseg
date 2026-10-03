"use strict";
// boneseg front end: Shared state, helpers, settings and saved clicks.
// Plain scripts sharing globals, loaded in order by index.html. No build step.

const $ = (id) => document.getElementById(id);
const S = {
  health: null,
  datasets: [],
  ds: null,
  c: 0,
  z: 0,
  low: 1,
  high: 99.5,
  points: {},          // "c:z" -> {pos: [[y, x]], neg: [[y, x]]} in full-resolution pixels
  history: [],         // Undo stack of {key, kind}
  mode: "pos",
  profiles: [],
  base: null,          // HTMLImageElement of the current plane at display resolution
  layers: {},          // mask, heat, unc, ref -> HTMLImageElement
  result: null,
  view: { scale: 1, ox: 0, oy: 0 },
  seq: 0,
  job: null,
  space: false,
  method: "clicks",    // "clicks" or "learned"
  labels: [],          // [{channel, z}] slices with a saved corrected mask
  head: null,          // Info about the learned model for the current channel
  structures: [{ name: "Object", color: "#22d27a" }],
  active: 0,           // Index of the structure that object clicks go to
  roiDraft: null,      // Corners of a region being drawn, in full-resolution pixels
  sideY: null,         // Row of the side view, in full-resolution pixels
  editing: false,
  edit: null,          // Offscreen canvas with the mask being corrected, at display resolution
};

// ---------------------------------------------------------------------------------------------
// Small helpers
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {},
    ...opts,
    body: opts.body && !(opts.body instanceof FormData) ? JSON.stringify(opts.body) : opts.body,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) { /* not JSON */ }
    throw new Error(msg);
  }
  return res.headers.get("content-type")?.includes("json") ? res.json() : res;
}

let toastTimer = null;
function toast(msg, isError = false) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.toggle("error", isError);
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), isError ? 6000 : 3000);
}

function status(msg) { $("statusbar").textContent = msg; }
function busy(on, text = "Working…") { $("busy").classList.toggle("hidden", !on); $("busyText").textContent = text; }
const key = () => `${S.c}:${S.z}`;
const pts = () => (S.points[key()] ||= { pos: [], neg: [] });

// Structures: the first one uses p.pos, further ones live in p.extra[k - 1].pos
const STRUCT_COLORS = ["#22d27a", "#ffa53a", "#c78bff", "#ff5ad2", "#ffe14a", "#5ad1ff"];
function posList(p, k) {
  if (!k) return p.pos;
  p.extra ||= [];
  p.extra[k - 1] ||= { pos: [] };
  return p.extra[k - 1].pos;
}
const isMulti = () => S.structures.length > 1;

// Clicks are saved on the server per slice, so a reload keeps them
const saveTimers = {};
function persistClicks(k = key()) {
  if (!S.ds) return;
  clearTimeout(saveTimers[k]);
  const dsId = S.ds.id;
  saveTimers[k] = setTimeout(() => {
    const [c, z] = k.split(":").map(Number);
    const p = S.points[k] || { pos: [], neg: [] };
    const extra = S.structures.slice(1).map((st, i) => ({ name: st.name, color: st.color, pos: p.extra?.[i]?.pos || [] }));
    api(`/api/datasets/${dsId}/annotations`, { method: "PUT", body: { channel: c, z, pos: p.pos, neg: p.neg, extra } }).catch((e) => toast(`Could not save clicks: ${e.message}`, true));
  }, 400);
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not load image"));
    img.src = src;
  });
}

function fmt(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (a >= 1e4) return (v / 1e3).toFixed(1) + "k";
  return v.toFixed(digits);
}

// ---------------------------------------------------------------------------------------------
// Settings
function settings() {
  return {
    backbone: $("backboneSelect").value,
    threshold_mode: $("thrMode").value,
    top_percent: +$("topPercent").value,
    manual_threshold: +$("manualThr").value,
    neg_weight: +$("lambda").value,
    min_object_um2: +$("minObj").value,
    fill_holes_um2: +$("fillHoles").value,
    smooth_px: +$("smooth").value,
    vit_size: +$("vitSize").value,
    layer_from_end: +$("layer").value,
    clip_low: S.low,
    clip_high: S.high,
  };
}

function syncSettingLabels() {
  const mode = $("thrMode").value;
  $("topPercentRow").classList.toggle("hidden", mode !== "top_percent");
  $("manualRow").classList.toggle("hidden", mode !== "manual");
  $("thrHint").classList.toggle("hidden", mode !== "clicks");
  $("topPercentValue").textContent = `${$("topPercent").value}%`;
  $("manualValue").textContent = (+$("manualThr").value).toFixed(2);
  $("lambdaValue").textContent = (+$("lambda").value).toFixed(2);
  $("minObjValue").textContent = `${$("minObj").value} µm²`;
  $("fillValue").textContent = `${$("fillHoles").value} µm²`;
  $("smoothValue").textContent = `${$("smooth").value} px`;
  const bb = S.health?.backbones.find((b) => b.id === $("backboneSelect").value);
  $("backboneHint").textContent = !bb ? "" : bb.id === "classic"
    ? "Hand-made intensity and texture features. Instant, but less accurate than DINOv2."
    : bb.ready ? "Weights are downloaded. The first click on each slice computes features, later clicks are instant."
      : "The weights download on first use, which can take a minute.";
}

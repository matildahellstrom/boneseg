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

// An access link's ?token=... becomes a cookie, so every later API call carries it; shared by both pages
function takeTokenFromUrl() {
  const u = new URL(window.location.href);
  const t = u.searchParams.get("token");
  if (!t) return;
  document.cookie = `boneseg_token=${encodeURIComponent(t)}; path=/; SameSite=Strict; max-age=${60 * 60 * 24 * 30}`;
  u.searchParams.delete("token");
  window.history.replaceState(null, "", u.pathname + (u.search || "") + u.hash);
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

// Text from files, profiles or other users goes through esc() before it becomes HTML, so a shared
// profile or a channel name cannot inject markup or script into the page
function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}
const safeColor = (c) => (/^#[0-9a-f]{3,8}$/i.test(c || "") ? c : "#00c8f0");

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
function clickBody(k) {
  const [c, z] = k.split(":").map(Number);
  const p = S.points[k] || { pos: [], neg: [] };
  const extra = S.structures.slice(1).map((st, i) => ({ name: st.name, color: st.color, pos: p.extra?.[i]?.pos || [] }));
  return { channel: c, z, pos: p.pos, neg: p.neg, extra };
}

function persistClicks(k = key()) {
  if (S.ds) scheduleSave(S.ds.id, clickBody(k));
}

// Saving is delayed by 400 ms so quick clicking sends one request. Each dataset and slice has its own timer, and the
// request body is copied now: switching dataset or slice before the timer fires cannot change what gets saved.
// Both pages (index.html and simple.html) use these helpers.
function scheduleSave(dsId, body) {
  const id = `${dsId}|${body.channel}:${body.z}`;
  clearTimeout(saveTimers[id]?.timer);
  saveTimers[id] = { dsId, body: JSON.parse(JSON.stringify(body)), timer: setTimeout(() => sendSave(id), 400) };
}

function sendSave(id, beacon = false) {
  const s = saveTimers[id];
  if (!s) return Promise.resolve();
  clearTimeout(s.timer);
  delete saveTimers[id];
  const url = `/api/datasets/${s.dsId}/annotations`;
  if (beacon) return Promise.resolve(navigator.sendBeacon(url, new Blob([JSON.stringify(s.body)], { type: "application/json" })));
  return api(url, { method: "PUT", body: s.body }).catch((e) => toast(`Could not save clicks: ${e.message}`, true));
}

// Sends every waiting save now, for example before opening another dataset
function flushSaves() { return Promise.all(Object.keys(saveTimers).map((id) => sendSave(id))); }

// When the page closes or navigates away, waiting saves go out as beacons, which the browser delivers after unload
window.addEventListener("pagehide", () => { for (const id of Object.keys(saveTimers)) sendSave(id, true); });

// Reverses one undo-history entry on the click lists p. An added point is removed (the last copy of exactly that
// point, wherever it is now), a removed point goes back where it was. Shared by both pages.
function undoEntry(p, h, listOf) {
  const list = listOf(p);
  if (h.op === "remove") { list.splice(Math.min(h.index, list.length), 0, h.pt); return; }
  for (let i = list.length - 1; i >= 0; i--) {
    if (list[i][0] === h.pt[0] && list[i][1] === h.pt[1]) { list.splice(i, 1); return; }
  }
}

// Drops undo entries for one slice, after its clicks were replaced wholesale (Clear, Copy to next slice)
function forgetHistory(history, k) { return history.filter((h) => h.key !== k); }

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not load image"));
    img.src = src;
  });
}

// Which image to open when a page loads: the one named in the link (?ds=, from the other page's switch button),
// else the last one opened in this browser, else none. Both pages remember the image they open.
function rememberDataset(id) { try { localStorage.setItem("boneseg-last-ds", id); } catch (_) { /* storage blocked */ } }
function wantedView(datasets) {
  const q = new URL(window.location.href).searchParams;
  let id = q.get("ds");
  try { if (!id) id = localStorage.getItem("boneseg-last-ds"); } catch (_) { /* storage blocked */ }
  if (!datasets.some((d) => d.id === id)) return null;
  const num = (k) => (q.get(k) == null || q.get(k) === "" ? null : +q.get(k));
  return { id, c: num("c"), z: num("z") };
}
// The first slice of channel c with object clicks, or null
function firstClickedSlice(points, c, nZ = Infinity) {
  const zs = Object.keys(points).filter((k) => k.startsWith(`${c}:`) && points[k].pos?.length).map((k) => +k.split(":")[1]).filter((z) => z < nZ);
  return zs.length ? Math.min(...zs) : null;
}

// Units for measurements: files without a pixel size are measured with a 1 um placeholder, so they are pixels
const areaUnit = () => (S.ds?.voxel_size_known === false ? "px²" : "µm²");
const lengthUnit = () => (S.ds?.voxel_size_known === false ? "px" : "µm");
const volumeUnit = () => (S.ds?.voxel_size_known === false ? "voxels" : "µm³");
const perArea = () => (S.ds?.voxel_size_known === false ? "per Mpx" : "per mm²");

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
    threshold_position: $("strict").value === "" ? -1 : +$("strict").value,
    edge_refine: $("edgeRefine").value,
    shift_passes: +$("shiftPasses").value,
    refiner: $("refiner").value,
    suggest_missed: $("suggestMissed").checked,
    sam_refine: $("samRefine").value,
    clip_low: S.low,
    clip_high: S.high,
  };
}

// Settings are remembered per dataset, so reopening an image restores them
let settingsTimer = null;
function rememberSettings() {
  if (!S.ds) return;
  clearTimeout(settingsTimer);
  const dsId = S.ds.id;
  settingsTimer = setTimeout(() => api(`/api/datasets/${dsId}`, { method: "PATCH", body: { settings: settings() } }).catch(() => {}), 800);
}

function applySettings(st) {
  if (!st) return;
  const set = (id, v) => { if (v !== undefined && v !== null && $(id)) $(id).value = v; };
  set("backboneSelect", st.backbone); set("thrMode", st.threshold_mode); set("topPercent", st.top_percent);
  set("manualThr", st.manual_threshold); set("lambda", st.neg_weight); set("minObj", st.min_object_um2);
  set("fillHoles", st.fill_holes_um2); set("smooth", st.smooth_px); set("vitSize", st.vit_size); set("layer", st.layer_from_end);
  set("strict", st.threshold_position); set("edgeRefine", st.edge_refine); set("shiftPasses", st.shift_passes); set("refiner", st.refiner);
  set("samRefine", st.sam_refine);
  if (st.suggest_missed != null) $("suggestMissed").checked = !!st.suggest_missed;
  if (st.clip_low != null) { S.low = st.clip_low; $("lowSlider").value = st.clip_low; }
  if (st.clip_high != null) { S.high = st.clip_high; $("highSlider").value = st.clip_high; }
  syncSettingLabels();
}

function syncSettingLabels() {
  const mode = $("thrMode").value;
  $("topPercentRow").classList.toggle("hidden", mode !== "top_percent");
  $("manualRow").classList.toggle("hidden", mode !== "manual");
  $("thrHint").classList.toggle("hidden", mode !== "clicks");
  $("topPercentValue").textContent = `${$("topPercent").value}%`;
  $("manualValue").textContent = (+$("manualThr").value).toFixed(2);
  $("lambdaValue").textContent = (+$("lambda").value).toFixed(2);
  $("minObjValue").textContent = `${$("minObj").value} ${areaUnit()}`;
  $("fillValue").textContent = `${$("fillHoles").value} ${areaUnit()}`;
  $("smoothValue").textContent = `${$("smooth").value} px`;
  const sam = S.health?.sam;
  $("samHint").classList.toggle("hidden", !sam || $("samRefine").value === "off");
  if (sam) $("samHint").textContent = !sam.available ? "Needs the segment-anything package: pip install segment-anything"
    : (sam.weights_cached ? "" : "The first use downloads SAM's weights, 375 MB. ") + "Adds a few seconds on each new slice. Best with many clicks; ignored for the learned model, several structures and stack runs.";
  const bb = S.health?.backbones.find((b) => b.id === $("backboneSelect").value);
  $("backboneHint").textContent = !bb ? "" : bb.id === "classic"
    ? "Hand-made intensity and texture features. Instant, but less accurate than DINOv2."
    : bb.ready ? "Weights are downloaded. The first click on each slice computes features, later clicks are instant."
      : "The weights download on first use, which can take a minute.";
}

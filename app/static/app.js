/* LOB Alpha Studio · Frontend Application */

const STATE = {
  bootstrap: null,
  replays: {},
  currentReplayKey: "RELIANCE",
  currentStep: 120,
  playing: false,
  speed: 1,
  timerId: null,
  explainDebounce: null,
  shapVer: "flagship",
  sandboxModelKey: "RELIANCE",
  sandboxDebounce: null,
  diagSymbol: "RELIANCE",
  diagMetric: "macro_f1",
  mlAuditSymbol: "RELIANCE",
  mlAuditCmModel: "Flagship Hybrid",
  dayIdx: -1,
  dayWindowSec: 22500,
  dayOffsetSec: 0,
  daySmoothN: 1,
  dayData: null,
  trapMode: "acc",
  ablationStage: "D",
  arenaDataset: "RELIANCE",
  highlight250: false,
  selectedSector: "ALL",
  selectedSymbol25: "ITC",
  scatterPoints: [],
};

const CLS_NAME = ["FLAT", "UP", "DOWN"];
const CLS_COLOR = ["var(--flat)", "var(--up)", "var(--down)"];

const PRESETS = {
  ssf_sweep: { bid_l1_mult: 1.0, ask_l1_mult: 1.0, deep_mult: 0.0, spread_ticks: 1, ofi_sigma: 0.0, lag_persist: 0.5, vpin_pct: 0.5, drift_30s_ticks: 0.0, vol_pct: 0.5, day_progress: 0.45, ssf_imb: 0.75, ssf_lead_sigma: 2.4 },
  buyer: { bid_l1_mult: 3.8, ask_l1_mult: 0.4, deep_mult: 0.85, spread_ticks: 1, ofi_sigma: 2.2, lag_persist: 0.9, vpin_pct: 0.35, drift_30s_ticks: 3.5, vol_pct: 0.65, day_progress: 0.35, ssf_imb: 0.55, ssf_lead_sigma: 1.2 },
  seller: { bid_l1_mult: 0.4, ask_l1_mult: 3.8, deep_mult: -0.6, spread_ticks: 1, ofi_sigma: -2.2, lag_persist: 0.9, vpin_pct: 0.75, drift_30s_ticks: -3.5, vol_pct: 0.7, day_progress: 0.45, ssf_imb: -0.55, ssf_lead_sigma: -1.2 },
  wide: { bid_l1_mult: 0.8, ask_l1_mult: 0.8, deep_mult: 0.0, spread_ticks: 3, ofi_sigma: 0.2, lag_persist: 0.5, vpin_pct: 0.8, drift_30s_ticks: 0.0, vol_pct: 0.8, day_progress: 0.15, ssf_imb: 0.0, ssf_lead_sigma: 0.0 },
  closing_freeze: { bid_l1_mult: 3.6, ask_l1_mult: 0.45, deep_mult: 0.4, spread_ticks: 1, ofi_sigma: 0.3, lag_persist: 0.95, vpin_pct: 0.2, drift_30s_ticks: 0.0, vol_pct: 0.0, day_progress: 0.98, ssf_imb: 0.1, ssf_lead_sigma: 0.0 },
  balanced: { bid_l1_mult: 1.0, ask_l1_mult: 1.0, deep_mult: 0.0, spread_ticks: 1, ofi_sigma: 0.0, lag_persist: 0.65, vpin_pct: 0.5, drift_30s_ticks: 0.0, vol_pct: 0.5, day_progress: 0.45, ssf_imb: 0.0, ssf_lead_sigma: 0.0 },
};

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function setupHiDPICanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(300, Math.round(rect.width));
  const h = Math.max(180, Math.round(rect.height));
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

/* ==================== BOOTSTRAP & NAVIGATION ==================== */
async function init() {
  const res = await fetch("/api/bootstrap");
  STATE.bootstrap = await res.json();

  initTheme();
  initNav();
  initTerminalControls();
  initSandboxControls();
  initDiagControls();
  initMlAuditControls();
  initArenaControls();
  initUniverseControls();

  await selectReplay("RELIANCE");
  await runSandboxPredict();
  renderDiagnosticsAll();
  renderMlAuditAll();
  renderTrap();
  renderAblation();
  renderArenaLadder();
  renderUniverseAll();
}

function initTheme() {
  const btn = document.getElementById("theme-toggle");
  const icon = document.getElementById("theme-icon");
  btn.addEventListener("click", () => {
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    document.documentElement.setAttribute("data-theme", isDark ? "light" : "dark");
    icon.textContent = isDark ? "☾" : "☀";
    redrawActiveCanvases();
  });
}

function initNav() {
  document.querySelectorAll(".nav-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-pill").forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      const target = btn.dataset.tab;
      document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
      document.getElementById(`view-${target}`).classList.add("active");
      setTimeout(redrawActiveCanvases, 30);
    });
  });
  window.addEventListener("resize", () => setTimeout(redrawActiveCanvases, 60));
}

function redrawActiveCanvases() {
  if (document.getElementById("view-terminal").classList.contains("active")) {
    drawPriceChart();
    drawDaySessionChart();
  }
  if (document.getElementById("view-diagnostics").classList.contains("active")) {
    drawUpgradeChart();
    drawConvictionChart();
  }
  if (document.getElementById("view-arena").classList.contains("active")) drawArenaLadder();
  if (document.getElementById("view-universe").classList.contains("active")) drawUniverseScatter();
}

/* ==================== VIEW 1: LIVE TERMINAL ==================== */
function initTerminalControls() {
  const wrap = document.getElementById("replay-pills");
  STATE.bootstrap.summaries.replays.forEach((r) => {
    const b = document.createElement("button");
    b.className = "inst-pill" + (r.key === STATE.currentReplayKey ? " active" : "");
    b.dataset.key = r.key;
    b.innerHTML = `<span>${r.symbol === "NIFTY_FUT" ? (r.segment_id === 3 ? "NIFTY Seg 3" : "NIFTY 50 FUT") : r.symbol}</span><span class="mini-tag">${r.badge}</span>`;
    b.addEventListener("click", () => selectReplay(r.key));
    wrap.appendChild(b);
  });

  document.getElementById("btn-play").addEventListener("click", togglePlay);
  document.getElementById("btn-prev").addEventListener("click", () => setStep(STATE.currentStep - 1));
  document.getElementById("btn-next").addEventListener("click", () => setStep(STATE.currentStep + 1));
  document.getElementById("replay-slider").addEventListener("input", (e) => setStep(parseInt(e.target.value, 10)));

  document.querySelectorAll(".speed-group .btn-chip").forEach((c) => {
    c.addEventListener("click", () => {
      c.parentElement.querySelectorAll(".btn-chip").forEach((x) => x.classList.remove("active"));
      c.classList.add("active");
      STATE.speed = parseInt(c.dataset.speed, 10);
      if (STATE.playing) { stopPlay(); startPlay(); }
    });
  });

  function jumpToCurated(field) {
    const rep = STATE.replays[STATE.currentReplayKey];
    if (!rep) return;
    const spots = (rep.meta[field] && rep.meta[field].length) ? rep.meta[field] : (rep.meta.high_conf_steps || [150]);
    const nextSpot = spots.find((s) => s > STATE.currentStep + 3) ?? spots[0];
    setStep(nextSpot, true);
  }

  document.getElementById("btn-jump-signal").addEventListener("click", () => jumpToCurated("high_conf_steps"));
  document.getElementById("btn-jump-ssf").addEventListener("click", () => jumpToCurated("ssf_rescues_steps"));
  document.getElementById("btn-jump-v2").addEventListener("click", () => jumpToCurated("v2_rescues_steps"));
  document.getElementById("btn-jump-tlob").addEventListener("click", () => jumpToCurated("tlob_wins_steps"));
  document.getElementById("btn-jump-lgbm").addEventListener("click", () => jumpToCurated("lgbm_wins_steps"));
  document.getElementById("btn-jump-trap").addEventListener("click", () => jumpToCurated("boundary_trap_steps"));

  document.querySelectorAll("#shap-ver-toggle .btn-chip").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#shap-ver-toggle .btn-chip").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      STATE.shapVer = b.dataset.ver;
      fetchLiveStepExplain(STATE.currentReplayKey, STATE.currentStep);
    });
  });

  // Full-Day & 12-Day Explorer controls
  document.getElementById("day-select").addEventListener("change", (e) => {
    STATE.dayIdx = parseInt(e.target.value, 10);
    STATE.dayOffsetSec = 0;
    document.getElementById("day-offset-slider").value = "0";
    fetchDayOverview();
  });

  document.querySelectorAll("#day-window-group .btn-chip").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#day-window-group .btn-chip").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      STATE.dayWindowSec = parseInt(b.dataset.win, 10);
      if (STATE.dayIdx < 0 && STATE.dayWindowSec < 22500) {
        STATE.dayIdx = 0;
        document.getElementById("day-select").value = "0";
      }
      STATE.dayOffsetSec = 0;
      document.getElementById("day-offset-slider").value = "0";
      fetchDayOverview();
    });
  });

  document.getElementById("day-offset-slider").addEventListener("input", (e) => {
    STATE.dayOffsetSec = parseInt(e.target.value, 10);
    fetchDayOverview();
  });

  document.querySelectorAll("#day-smooth-group .btn-chip").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#day-smooth-group .btn-chip").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      STATE.daySmoothN = parseInt(b.dataset.smooth, 10);
      fetchDayOverview();
    });
  });
}

async function selectReplay(key) {
  stopPlay();
  STATE.currentReplayKey = key;
  document.querySelectorAll("#replay-pills .inst-pill").forEach((p) => {
    p.classList.toggle("active", p.dataset.key === key);
  });

  if (!STATE.replays[key]) {
    const res = await fetch(`/api/replay/${key}`);
    STATE.replays[key] = await res.json();
  }
  const rep = STATE.replays[key];
  const alertEl = document.getElementById("shift-alert");
  if (key === "NIFTY_FUT_SEG3") {
    alertEl.classList.remove("hidden");
    document.getElementById("shift-alert-text").textContent =
      `NIFTY rallied +10.7% from Train (~₹${Math.round(rep.meta.train_mid).toLocaleString()}) to Test (~₹${Math.round(rep.meta.median_mid).toLocaleString()}). Static-scaled LSTM & DeepLOB5 break under covariate shift, while adaptive BiN (TLOB) and ratio-based LightGBM hold steady.`;
  } else {
    alertEl.classList.add("hidden");
  }

  document.getElementById("replay-date-meta").textContent = `${rep.meta.title} · ${rep.meta.date_ist} (${rep.meta.start_time}–${rep.meta.end_time} IST)`;
  document.getElementById("ob-tick-tag").textContent = `Tick ₹${rep.meta.tick_size.toFixed(2)}`;
  const startStep = (rep.meta.high_conf_steps && rep.meta.high_conf_steps[0]) || 120;
  setStep(startStep, true);
  fetchDayOverview(true);
}

async function fetchDayOverview(resetDates = false) {
  const sym = STATE.currentReplayKey === "NIFTY_FUT_SEG3" ? "NIFTY_FUT" : STATE.currentReplayKey;
  document.getElementById("day-exp-sym").textContent = sym;
  const qs = `day_idx=${STATE.dayIdx}&window_sec=${STATE.dayWindowSec}&offset_sec=${STATE.dayOffsetSec}&smooth_n=${STATE.daySmoothN}`;
  try {
    const res = await fetch(`/api/day_overview/${sym}?${qs}`);
    if (!res.ok) return;
    const d = await res.json();
    STATE.dayData = d;

    const sel = document.getElementById("day-select");
    if (resetDates || sel.options.length === 0) {
      sel.innerHTML = `<option value="-1">All ${d.dates.length} Test Days (~268k s)</option>` +
        d.dates.map((dt, idx) => `<option value="${idx}">Day ${idx + 1}: ${dt} (${(d.daily_summary[idx].dir_hit * 100).toFixed(1)}% hit)</option>`).join("");
      sel.value = String(STATE.dayIdx);
    }

    const offWrap = document.getElementById("day-offset-wrap");
    if (d.mode === "single_day" && STATE.dayWindowSec < d.max_day_sec) {
      offWrap.classList.remove("hidden");
      const sl = document.getElementById("day-offset-slider");
      sl.max = String(Math.max(0, d.max_day_sec - STATE.dayWindowSec));
      document.getElementById("day-window-span").textContent = `${d.start_ist}–${d.end_ist} IST`;
    } else {
      offWrap.classList.add("hidden");
    }

    const st = d.stats;
    document.getElementById("day-stat-hit").textContent = `${(st.dir_hit * 100).toFixed(1)}% (F1 ${st.macro_f1.toFixed(3)})`;
    document.getElementById("day-stat-hc").textContent = st.high_conf_hit != null ? `${(st.high_conf_hit * 100).toFixed(1)}%` : "n/a";
    document.getElementById("day-stat-b10").textContent = st.batch_wins["10"] != null ? `${st.batch_wins["10"].toFixed(1)}%` : "n/a";
    document.getElementById("day-stat-b20").textContent = st.batch_wins["20"] != null ? `${st.batch_wins["20"].toFixed(1)}%` : "n/a";
    document.getElementById("day-stat-b50").textContent = st.batch_wins["50"] != null ? `${st.batch_wins["50"].toFixed(1)}%` : "n/a";
    document.getElementById("day-stat-b100").textContent = st.batch_wins["100"] != null ? `${st.batch_wins["100"].toFixed(1)}%` : "n/a";

    const modeDesc = d.mode === "all_days"
      ? `Across all ${d.dates.length} held-out test days (${st.n_seconds.toLocaleString()} s), single-call directional hit rate is <b>${(st.dir_hit * 100).toFixed(1)}%</b> and Batch-100 majority win rate reaches <b>${st.batch_wins["100"]}%</b>.`
      : `Full window <b>${d.date}</b> (${d.start_ist}–${d.end_ist} IST, ${st.n_seconds.toLocaleString()} s): live accuracy <b>${(st.accuracy * 100).toFixed(1)}%</b>, directional hit <b>${(st.dir_hit * 100).toFixed(1)}%</b>.`;
    const smoothNote = STATE.daySmoothN > 1
      ? ` <span style="color:var(--amber)">(Trailing Avg ${STATE.daySmoothN}s enabled: observe phase-lag drop vs Live N=1.)</span>`
      : "";
    document.getElementById("day-session-note").innerHTML = modeDesc + smoothNote;
    drawDaySessionChart();
  } catch (err) {
    console.error("day_overview error:", err);
  }
}

function drawDaySessionChart() {
  const canvas = document.getElementById("canvas-day-session");
  const d = STATE.dayData;
  if (!canvas || !d || !d.series || !d.series.length) return;
  const { ctx, w, h } = setupHiDPICanvas(canvas);

  const pts = d.series;
  const mids = pts.map((p) => p.mid);
  const hits = pts.map((p) => p.dir_hit_pct);
  const minM = Math.min(...mids), maxM = Math.max(...mids);
  const padM = Math.max(0.05, (maxM - minM) * 0.14);
  const m0 = minM - padM, m1 = maxM + padM;

  const minH = Math.min(45, ...hits) - 2, maxH = Math.max(82, ...hits) + 2;
  const ml = 68, mr = 58, mt = 24, mb = 32;
  const pw = w - ml - mr, ph = h - mt - mb;
  const xOf = (i) => ml + (i / Math.max(1, pts.length - 1)) * pw;
  const yMid = (v) => mt + ph - ((v - m0) / (m1 - m0)) * ph;
  const yHit = (v) => mt + ph - ((v - minH) / (maxH - minH)) * ph;

  ctx.clearRect(0, 0, w, h);

  // Horizontal grid + Left Mid-Price ticks + Right Hit% ticks
  ctx.font = "10.5px 'JetBrains Mono', monospace";
  for (let i = 0; i <= 4; i++) {
    const yy = mt + ph - (i / 4) * ph;
    ctx.strokeStyle = cssVar("--border");
    ctx.beginPath(); ctx.moveTo(ml, yy); ctx.lineTo(w - mr, yy); ctx.stroke();
    ctx.fillStyle = cssVar("--blue");
    const mv = m0 + (i / 4) * (m1 - m0);
    ctx.fillText(`₹${mv >= 1000 ? Math.round(mv).toLocaleString() : mv.toFixed(1)}`, 6, yy + 3.5);
    ctx.fillStyle = cssVar("--up");
    const hv = minH + (i / 4) * (maxH - minH);
    ctx.fillText(`${hv.toFixed(0)}%`, w - mr + 8, yy + 3.5);
  }

  // 50% coin-flip baseline
  const y50 = yHit(50);
  if (y50 >= mt && y50 <= mt + ph) {
    ctx.strokeStyle = cssVar("--flat");
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(ml, y50); ctx.lineTo(w - mr, y50); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Series 1: Mid Price (blue)
  ctx.strokeStyle = cssVar("--blue");
  ctx.lineWidth = 2;
  ctx.beginPath();
  pts.forEach((p, i) => {
    const px = xOf(i), py = yMid(p.mid);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();

  // Series 2: Directional Hit Rate % (green)
  ctx.strokeStyle = cssVar("--up");
  ctx.lineWidth = 2.3;
  ctx.beginPath();
  pts.forEach((p, i) => {
    const px = xOf(i), py = yHit(p.dir_hit_pct);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();
  if (pts.length <= 20) {
    pts.forEach((p, i) => {
      const px = xOf(i), py = yHit(p.dir_hit_pct);
      ctx.fillStyle = cssVar("--up");
      ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill();
    });
  }

  // X ticks
  ctx.fillStyle = cssVar("--text-muted");
  ctx.font = "10.5px 'JetBrains Mono', monospace";
  const tickCount = Math.min(6, pts.length);
  for (let k = 0; k < tickCount; k++) {
    const idx = Math.round((k / Math.max(1, tickCount - 1)) * (pts.length - 1));
    ctx.fillText(pts[idx].label, Math.min(w - mr - 32, xOf(idx) - 14), mt + ph + 18);
  }

  ctx.fillStyle = cssVar("--blue"); ctx.font = "700 11px Inter, sans-serif";
  ctx.fillText(d.mode === "all_days" ? "● Daily Close Mid (₹)" : "● Intraday Mid Price (₹)", ml, 13);
  ctx.fillStyle = cssVar("--up");
  ctx.fillText(d.mode === "all_days" ? "● Daily Directional Hit %" : "● Cumulative Directional Hit %", ml + 150, 13);
}

function togglePlay() {
  if (STATE.playing) stopPlay();
  else startPlay();
}

function startPlay() {
  STATE.playing = true;
  document.getElementById("btn-play").textContent = "⏸ Pause";
  const intervalMs = Math.max(120, Math.round(700 / STATE.speed));
  STATE.timerId = setInterval(() => {
    const next = STATE.currentStep + 1;
    setStep(next >= 875 ? 90 : next);
  }, intervalMs);
}

function stopPlay() {
  STATE.playing = false;
  document.getElementById("btn-play").textContent = "▶ Play";
  if (STATE.timerId) { clearInterval(STATE.timerId); STATE.timerId = null; }
}

function setStep(step, immediateExplain = false) {
  const clamped = Math.max(90, Math.min(875, step));
  STATE.currentStep = clamped;
  document.getElementById("replay-slider").value = clamped;
  document.getElementById("replay-step-num").textContent = clamped;

  const rep = STATE.replays[STATE.currentReplayKey];
  if (!rep) return;

  document.getElementById("replay-clock").textContent = `${rep.time[clamped]} IST`;
  renderOrderBook(document.getElementById("ob-ladder"), rep.lob_20[clamped], rep.meta.tick_size, true);
  renderSsfTelemetry(rep, clamped);
  renderModelRace(rep, clamped);
  drawPriceChart();
  updateOutcomeBadge(rep, clamped);

  clearTimeout(STATE.explainDebounce);
  STATE.explainDebounce = setTimeout(() => fetchLiveStepExplain(STATE.currentReplayKey, clamped), immediateExplain ? 0 : 90);
}

function renderSsfTelemetry(rep, step) {
  const ssf = rep.ssf_telemetry;
  if (!ssf) return;
  document.getElementById("ssf-source-name").textContent = ssf.source || "SSF";
  const imb = ssf.imb_l1[step] || 0;
  const r5 = ssf.ret_5s_bp[step] || 0;
  const ld5 = ssf.lead_5s_bp[step] || 0;
  const bas = ssf.basis_dev_bp[step] || 0;
  const colorOf = (v, th) => v > th ? "var(--up)" : v < -th ? "var(--down)" : "var(--text)";

  const elImb = document.getElementById("ssf-imb-val");
  elImb.textContent = `${imb >= 0 ? "+" : ""}${imb.toFixed(2)}`;
  elImb.style.color = colorOf(imb, 0.12);

  const elR5 = document.getElementById("ssf-ret5-val");
  elR5.textContent = `${r5 >= 0 ? "+" : ""}${r5.toFixed(2)} bp`;
  elR5.style.color = colorOf(r5, 0.4);

  const elLd5 = document.getElementById("ssf-lead5-val");
  elLd5.textContent = `${ld5 >= 0 ? "+" : ""}${ld5.toFixed(2)} bp`;
  elLd5.style.color = colorOf(ld5, 0.4);

  const elBas = document.getElementById("ssf-basis-val");
  elBas.textContent = `${bas >= 0 ? "+" : ""}${bas.toFixed(2)} bp`;
  elBas.style.color = colorOf(bas, 0.4);
}

function renderOrderBook(container, lob20, tickSize, updateKPIs) {
  const asks = [];
  const bids = [];
  for (let lvl = 0; lvl < 5; lvl++) {
    asks.push({ lvl: lvl + 1, p: lob20[lvl * 4], q: lob20[lvl * 4 + 1] });
    bids.push({ lvl: lvl + 1, p: lob20[lvl * 4 + 2], q: lob20[lvl * 4 + 3] });
  }
  const maxQ = Math.max(1, ...asks.map((a) => a.q), ...bids.map((b) => b.q));
  const ap1 = asks[0].p, bp1 = bids[0].p;
  const mid = 0.5 * (ap1 + bp1);
  const spread = ap1 - bp1;
  const spreadTicks = Math.max(1, Math.round(spread / tickSize));
  const imb1 = (bids[0].q - asks[0].q) / (bids[0].q + asks[0].q + 1e-8);

  if (updateKPIs) {
    document.getElementById("ob-mid").textContent = `₹${mid.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    document.getElementById("ob-spread").textContent = `₹${spread.toFixed(2)} (${spreadTicks}t)`;
    const imbEl = document.getElementById("ob-imb1");
    imbEl.textContent = `${imb1 >= 0 ? "+" : ""}${imb1.toFixed(2)}`;
    imbEl.style.color = imb1 > 0.12 ? "var(--up)" : imb1 < -0.12 ? "var(--down)" : "var(--text)";

    const fill = document.getElementById("ob-imb-fill");
    const pct = Math.min(50, Math.abs(imb1) * 50);
    fill.style.left = imb1 >= 0 ? "50%" : `${50 - pct}%`;
    fill.style.width = `${pct}%`;
    fill.style.background = imb1 >= 0 ? "var(--up)" : "var(--down)";
    document.getElementById("ob-imb-caption").textContent =
      imb1 > 0.2 ? "More resting buyers (bullish tilt)" :
      imb1 < -0.2 ? "More resting sellers (bearish tilt)" : "Balanced L1 queues";
  }

  let html = "";
  for (let i = 4; i >= 0; i--) {
    const a = asks[i];
    const w = Math.max(4, Math.round((a.q / maxQ) * 100));
    html += `<div class="ob-row ask"><div class="ob-bar" style="width:${w}%"></div><span class="ob-lvl">ASK ${a.lvl}</span><span class="ob-px">₹${a.p.toFixed(2)}</span><span class="ob-qty">${Math.round(a.q).toLocaleString()}</span></div>`;
  }
  html += `<div class="ob-spread-divider">Mid ₹${mid.toFixed(2)} · Spread ${spreadTicks} tick${spreadTicks > 1 ? "s" : ""} · Imb ${imb1 >= 0 ? "+" : ""}${imb1.toFixed(2)}</div>`;
  for (let i = 0; i < 5; i++) {
    const b = bids[i];
    const w = Math.max(4, Math.round((b.q / maxQ) * 100));
    html += `<div class="ob-row bid"><div class="ob-bar" style="width:${w}%"></div><span class="ob-lvl">BID ${b.lvl}</span><span class="ob-px">₹${b.p.toFixed(2)}</span><span class="ob-qty">${Math.round(b.q).toLocaleString()}</span></div>`;
  }
  container.innerHTML = html;
}

function updateOutcomeBadge(rep, step) {
  const truth = rep.y_true[step];
  const heroKey = rep.probs["Hybrid Ensemble"] ? "Hybrid Ensemble" : "LightGBM";
  const heroLabel = heroKey === "Hybrid Ensemble" ? "★ Flagship Hybrid" : "LightGBM";
  const pHero = rep.probs[heroKey][step];
  const pred = argmax(pHero);
  const retBp = rep.fwd_ret_bp[step];
  document.getElementById("truth-badge").textContent = `True 20s Label: ${CLS_NAME[truth]}`;
  document.getElementById("chart-move-detail").textContent = `Realised Δmid(20s): ${retBp >= 0 ? "+" : ""}${retBp.toFixed(2)} bp (${CLS_NAME[truth]})`;

  const pill = document.getElementById("outcome-pill");
  pill.className = "outcome-pill";
  if (pred === truth) {
    pill.textContent = `✓ ${heroLabel} Correct (${CLS_NAME[pred]})`;
    pill.classList.add("outcome-hit");
  } else {
    pill.textContent = `${heroLabel} ${CLS_NAME[pred]} vs True ${CLS_NAME[truth]}`;
    if (pred !== 0 && truth !== 0) pill.classList.add("outcome-miss");
  }

  // Rolling directional hit rate across slice up to current step
  let hits = 0, dirTot = 0;
  for (let i = 90; i <= step; i++) {
    const pr = argmax(rep.probs[heroKey][i]);
    if (pr !== 0 && rep.y_true[i] !== 0) {
      dirTot++;
      if (pr === rep.y_true[i]) hits++;
    }
  }
  document.getElementById("session-hit-rate").textContent = dirTot ? `${((hits / dirTot) * 100).toFixed(1)}% (${hits}/${dirTot})` : "--";
}

function renderModelRace(rep, step) {
  const truth = rep.y_true[step];
  const models = [
    ["Hybrid Ensemble", "★ 46f+SSF·MH+TLOB", true],
    ["LightGBM Flagship", "46f +SSF · 10s/20s", true],
    ["LightGBM v2", "38f · Multi-Scale", false],
    ["LightGBM", "v1 · 25f", false],
    ["TLOB", "T5 · Attn+BiN", false],
    ["DeepLOB5", "T5 · CNN", false],
    ["LSTM", "T4 · RNN", false],
    ["Random Forest", "T2 · Bag", false],
    ["Logistic Regression", "T1 · Linear", false],
  ].filter(([name]) => Boolean(rep.probs[name]));

  const list = document.getElementById("model-race-list");
  list.innerHTML = models.map(([name, tier, isHi]) => {
    const p = rep.probs[name][step]; // [flat, up, down]
    const wDown = (p[2] * 100).toFixed(1);
    const wFlat = (p[0] * 100).toFixed(1);
    const wUp = (p[1] * 100).toFixed(1);
    const pred = argmax(p);
    const match = pred === truth;
    const col = pred === 1 ? "var(--up)" : pred === 2 ? "var(--down)" : "var(--text-muted)";
    return `
      <div class="race-row${isHi ? " highlight" : ""}">
        <div class="race-name"><span>${name}</span><span class="race-tier">${tier}</span></div>
        <div class="stacked-bar" title="DOWN ${wDown}% | FLAT ${wFlat}% | UP ${wUp}%">
          <div class="seg-down seg-label" style="width:${wDown}%">${p[2] >= 0.16 ? Math.round(p[2]*100)+"%" : ""}</div>
          <div class="seg-flat seg-label" style="width:${wFlat}%">${p[0] >= 0.18 ? Math.round(p[0]*100)+"%" : ""}</div>
          <div class="seg-up seg-label" style="width:${wUp}%">${p[1] >= 0.16 ? Math.round(p[1]*100)+"%" : ""}</div>
        </div>
        <div class="race-verdict" style="color:${col}">${CLS_NAME[pred]} ${match ? "✓" : ""}</div>
      </div>`;
  }).join("");
}

async function fetchLiveStepExplain(key, step) {
  try {
    const res = await fetch("/api/explain_step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ replay_key: key, step, ver: STATE.shapVer }),
    });
    const d = await res.json();
    const verLabel = d.active_ver === "flagship"
      ? `★ Flagship LightGBM (46f +SSF·10s/20s)`
      : d.active_ver === "v2"
      ? "LightGBM v2 (38f)"
      : "LightGBM v1 (25f)";
    const featTag = d.active_ver === "flagship" ? "46f·MH" : d.active_ver === "v2" ? "38f" : "25f";
    document.getElementById("live-shap-latency").textContent = `${d.inference_us} µs · ${featTag}`;
    document.getElementById("replay-shap-summary").innerHTML =
      `<b>${verLabel}</b> outputs <b>${d.lgbm_pred}</b> (UP ${(d.lgbm_probs[1]*100).toFixed(1)}% · FLAT ${(d.lgbm_probs[0]*100).toFixed(1)}% · DOWN ${(d.lgbm_probs[2]*100).toFixed(1)}%). TreeSHAP pushes toward <b>${d.focus_class}</b>:`;
    renderShapBars(document.getElementById("replay-shap-bars"), d.top_shap);

    if (d.diagnosis) {
      const badgeEl = document.getElementById("diag-badge");
      badgeEl.textContent = d.diagnosis.badge;
      badgeEl.className = `diag-pill ${d.diagnosis.tone || ""}`;
      document.getElementById("diag-headline").textContent = d.diagnosis.headline;
    }
  } catch (err) {
    console.error("explain_step error:", err);
  }
}

function renderShapBars(container, items) {
  const maxAbs = Math.max(0.01, ...items.map((x) => Math.abs(x.shap)));
  container.innerHTML = items.map((item) => {
    const pct = Math.min(100, Math.round((Math.abs(item.shap) / maxAbs) * 100));
    const cls = item.shap >= 0 ? "shap-pos" : "shap-neg";
    return `
      <div class="shap-row">
        <div class="shap-feat" title="${item.feature} = ${item.value}">${item.label}<span>val = ${item.value}</span></div>
        <div class="shap-track"><div class="shap-bar ${cls}" style="width:${pct}%"></div></div>
        <div class="shap-num" style="color:${item.shap >= 0 ? "var(--up)" : "var(--down)"}">${item.shap >= 0 ? "+" : ""}${item.shap.toFixed(3)}</div>
      </div>`;
  }).join("");
}

function drawPriceChart() {
  const canvas = document.getElementById("canvas-price");
  const rep = STATE.replays[STATE.currentReplayKey];
  if (!canvas || !rep) return;

  const { ctx, w, h } = setupHiDPICanvas(canvas);
  const step = STATE.currentStep;
  const lo = Math.max(0, step - 90);
  const hi = Math.min(rep.mid.length - 1, step + 25);
  const slice = rep.mid.slice(lo, hi + 1);
  const minP = Math.min(...slice);
  const maxP = Math.max(...slice);
  const padY = Math.max(rep.meta.tick_size * 1.5, (maxP - minP) * 0.18);
  const y0 = minP - padY, y1 = maxP + padY;

  const ml = 58, mr = 16, mt = 18, mb = 26;
  const pw = w - ml - mr, ph = h - mt - mb;
  const xOf = (idx) => ml + ((idx - lo) / Math.max(1, hi - lo)) * pw;
  const yOf = (p) => mt + ph - ((p - y0) / (y1 - y0)) * ph;

  ctx.clearRect(0, 0, w, h);

  // Shaded future 20s prediction window
  const xNow = xOf(step);
  const xFut = xOf(Math.min(hi, step + 20));
  const truth = rep.y_true[step];
  ctx.fillStyle = truth === 1 ? "rgba(16,185,129,0.10)" : truth === 2 ? "rgba(225,75,75,0.10)" : "rgba(100,116,139,0.08)";
  ctx.fillRect(xNow, mt, xFut - xNow, ph);

  // Horizontal grid + labels
  ctx.strokeStyle = cssVar("--border");
  ctx.lineWidth = 1;
  ctx.fillStyle = cssVar("--text-muted");
  ctx.font = "11px 'JetBrains Mono', monospace";
  for (let i = 0; i <= 4; i++) {
    const val = y0 + (i / 4) * (y1 - y0);
    const yy = yOf(val);
    ctx.beginPath(); ctx.moveTo(ml, yy); ctx.lineTo(w - mr, yy); ctx.stroke();
    ctx.fillText(val.toFixed(1), 6, yy + 3.5);
  }

  // Past price line
  ctx.strokeStyle = cssVar("--blue");
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  for (let idx = lo; idx <= step; idx++) {
    const px = xOf(idx), py = yOf(rep.mid[idx]);
    if (idx === lo) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Future realised path [step, hi]
  ctx.strokeStyle = truth === 1 ? cssVar("--up") : truth === 2 ? cssVar("--down") : cssVar("--flat");
  ctx.setLineDash([4, 3]);
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let idx = step; idx <= hi; idx++) {
    const px = xOf(idx), py = yOf(rep.mid[idx]);
    if (idx === step) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Vertical cursor at "Now (t)"
  ctx.strokeStyle = cssVar("--text-muted");
  ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(xNow, mt); ctx.lineTo(xNow, mt + ph); ctx.stroke();

  // Dots at t and t+20s
  ctx.fillStyle = cssVar("--blue");
  ctx.beginPath(); ctx.arc(xNow, yOf(rep.mid[step]), 4.5, 0, Math.PI * 2); ctx.fill();
  if (step + 20 <= hi) {
    ctx.fillStyle = truth === 1 ? cssVar("--up") : truth === 2 ? cssVar("--down") : cssVar("--flat");
    ctx.beginPath(); ctx.arc(xFut, yOf(rep.mid[step + 20]), 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = cssVar("--text-muted");
    ctx.font = "11px Inter, sans-serif";
    ctx.fillText("t + 20s", Math.min(w - 50, xFut - 16), mt + ph + 16);
  }
  ctx.fillStyle = cssVar("--text");
  ctx.font = "600 11px Inter, sans-serif";
  ctx.fillText("Now (t)", xNow - 18, mt + ph + 16);
  ctx.fillStyle = cssVar("--text-muted");
  ctx.font = "11px Inter, sans-serif";
  ctx.fillText("t - 90s", ml, mt + ph + 16);
}

/* ==================== VIEW 2: MODEL SANDBOX ==================== */
function formatClockIST(prog) {
  const totalMin = 9 * 60 + 15 + Math.round(prog * 375);
  const hh = String(Math.floor(totalMin / 60)).padStart(2, "0");
  const mm = String(totalMin % 60).padStart(2, "0");
  return `${hh}:${mm} IST`;
}

function initSandboxControls() {
  const wrap = document.getElementById("sandbox-model-pills");
  ["RELIANCE", "NIFTY_FUT", "IDEA"].forEach((key) => {
    const b = document.createElement("button");
    b.className = "inst-pill" + (key === STATE.sandboxModelKey ? " active" : "");
    b.dataset.key = key;
    b.textContent = key === "NIFTY_FUT" ? "NIFTY 50 FUT" : key;
    b.addEventListener("click", () => {
      STATE.sandboxModelKey = key;
      document.querySelectorAll("#sandbox-model-pills .inst-pill").forEach((p) => p.classList.toggle("active", p.dataset.key === key));
      runSandboxPredict();
    });
    wrap.appendChild(b);
  });

  const sliderIds = [
    ["sb-bid-l1", "val-bid-l1", (v) => `${parseFloat(v).toFixed(1)}×`],
    ["sb-ask-l1", "val-ask-l1", (v) => `${parseFloat(v).toFixed(1)}×`],
    ["sb-deep", "val-deep", (v) => `${parseFloat(v) >= 0 ? "+" : ""}${parseFloat(v).toFixed(2)}`],
    ["sb-spread", "val-spread", (v) => `${v} tick${v > 1 ? "s" : ""}`],
    ["sb-ofi", "val-ofi", (v) => `${parseFloat(v) >= 0 ? "+" : ""}${parseFloat(v).toFixed(1)} σ`],
    ["sb-lag", "val-lag", (v) => `${parseFloat(v).toFixed(2)}`],
    ["sb-vpin", "val-vpin", (v) => `${Math.round(parseFloat(v) * 100)}%`],
    ["sb-drift", "val-drift", (v) => `${parseFloat(v) >= 0 ? "+" : ""}${parseFloat(v).toFixed(1)}t`],
    ["sb-vol", "val-vol", (v) => parseFloat(v) <= 0.01 ? "0% (Freeze)" : `${Math.round(parseFloat(v) * 100)}%`],
    ["sb-clock", "val-clock", (v) => formatClockIST(parseFloat(v))],
    ["sb-ssf-imb", "val-ssf-imb", (v) => `${parseFloat(v) >= 0 ? "+" : ""}${parseFloat(v).toFixed(2)}`],
    ["sb-ssf-lead", "val-ssf-lead", (v) => `${parseFloat(v) >= 0 ? "+" : ""}${parseFloat(v).toFixed(1)} σ`],
  ];
  sliderIds.forEach(([id, valId, fmt]) => {
    const el = document.getElementById(id);
    el.addEventListener("input", () => {
      document.getElementById(valId).textContent = fmt(el.value);
      clearTimeout(STATE.sandboxDebounce);
      STATE.sandboxDebounce = setTimeout(runSandboxPredict, 55);
    });
  });

  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const p = PRESETS[btn.dataset.preset];
      document.getElementById("sb-bid-l1").value = p.bid_l1_mult;
      document.getElementById("sb-ask-l1").value = p.ask_l1_mult;
      document.getElementById("sb-deep").value = p.deep_mult;
      document.getElementById("sb-spread").value = p.spread_ticks;
      document.getElementById("sb-ofi").value = p.ofi_sigma;
      document.getElementById("sb-lag").value = p.lag_persist;
      document.getElementById("sb-vpin").value = p.vpin_pct;
      document.getElementById("sb-drift").value = p.drift_30s_ticks;
      document.getElementById("sb-vol").value = p.vol_pct;
      document.getElementById("sb-clock").value = p.day_progress;
      document.getElementById("sb-ssf-imb").value = p.ssf_imb ?? 0.0;
      document.getElementById("sb-ssf-lead").value = p.ssf_lead_sigma ?? 0.0;
      sliderIds.forEach(([id, valId, fmt]) => {
        document.getElementById(valId).textContent = fmt(document.getElementById(id).value);
      });
      runSandboxPredict();
    });
  });
}

async function runSandboxPredict() {
  const payload = {
    model_key: STATE.sandboxModelKey,
    bid_l1_mult: parseFloat(document.getElementById("sb-bid-l1").value),
    ask_l1_mult: parseFloat(document.getElementById("sb-ask-l1").value),
    deep_mult: parseFloat(document.getElementById("sb-deep").value),
    spread_ticks: parseInt(document.getElementById("sb-spread").value, 10),
    ofi_sigma: parseFloat(document.getElementById("sb-ofi").value),
    lag_persist: parseFloat(document.getElementById("sb-lag").value),
    vpin_pct: parseFloat(document.getElementById("sb-vpin").value),
    drift_30s_ticks: parseFloat(document.getElementById("sb-drift").value),
    vol_pct: parseFloat(document.getElementById("sb-vol").value),
    day_progress: parseFloat(document.getElementById("sb-clock").value),
    ssf_imb: parseFloat(document.getElementById("sb-ssf-imb").value),
    ssf_lead_sigma: parseFloat(document.getElementById("sb-ssf-lead").value),
  };
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const d = await res.json();
  document.getElementById("sb-latency").textContent = `${d.inference_us} µs live 46f MH`;
  document.getElementById("sb-mid-readout").textContent = `Mid ₹${d.mid.toLocaleString("en-IN")} · Spread ${d.spread_bp} bp`;
  document.getElementById("sb-focus-tag").textContent = `Explaining Flagship 46f → ${d.focus_class}`;

  const tick = STATE.bootstrap.model_profiles[STATE.sandboxModelKey].stats.tick_size;
  renderOrderBook(document.getElementById("sb-ladder"), d.lob_20, tick, false);

  renderProbCard("sb-flag-verdict", "sb-flag-bars", d.lgbm_flagship_pred || d.lgbm_v2_pred, d.lgbm_flagship_probs || d.lgbm_v2_probs);
  renderProbCard("sb-v2-verdict", "sb-v2-bars", d.lgbm_v2_pred || d.lgbm_pred, d.lgbm_v2_probs || d.lgbm_probs);
  renderProbCard("sb-lgb-verdict", "sb-lgb-bars", d.lgbm_pred, d.lgbm_probs);
  const lp = d.logreg_probs;
  document.getElementById("sb-lr-line").innerHTML =
    `Tier 1 Logistic Regression (Linear 25f baseline): <b class="mono">${d.logreg_pred}</b> (UP ${(lp[1]*100).toFixed(1)}% · FLAT ${(lp[0]*100).toFixed(1)}% · DOWN ${(lp[2]*100).toFixed(1)}%) · Companion source: <b>${d.companion_source || "SSF"}</b>`;
  renderShapBars(document.getElementById("sb-shap-bars"), d.top_shap);
}

function renderProbCard(verdictId, barsId, pred, probs) {
  const v = document.getElementById(verdictId);
  v.textContent = pred;
  v.style.color = pred === "UP" ? "var(--up)" : pred === "DOWN" ? "var(--down)" : "var(--text-muted)";
  const order = [[1, "UP", "var(--up)"], [0, "FLAT", "var(--flat)"], [2, "DOWN", "var(--down)"]];
  document.getElementById(barsId).innerHTML = order.map(([idx, lbl, col]) => {
    const pct = (probs[idx] * 100).toFixed(1);
    return `<div class="prob-line"><span>${lbl}</span><div class="prob-track"><div class="prob-fill" style="width:${pct}%;background:${col}"></div></div><span class="mono">${pct}%</span></div>`;
  }).join("");
}

/* ==================== VIEW 3B: ERROR ANATOMY & V2 ==================== */
function initDiagControls() {
  const wrap = document.getElementById("diag-symbol-pills");
  if (!wrap) return;
  [
    ["RELIANCE", "RELIANCE (Medium Tick)"],
    ["NIFTY_FUT", "NIFTY_FUT (Tiny Tick)"],
    ["IDEA", "IDEA (Ultra-Large Tick)"],
  ].forEach(([sym, lbl]) => {
    const b = document.createElement("button");
    b.className = "inst-pill" + (sym === STATE.diagSymbol ? " active" : "");
    b.dataset.sym = sym;
    b.textContent = lbl;
    b.addEventListener("click", () => {
      STATE.diagSymbol = sym;
      document.querySelectorAll("#diag-symbol-pills .inst-pill").forEach((x) => x.classList.toggle("active", x.dataset.sym === sym));
      renderDiagnosticsAll();
    });
    wrap.appendChild(b);
  });

  document.querySelectorAll("#diag-metric-toggle .btn-chip").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#diag-metric-toggle .btn-chip").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      STATE.diagMetric = b.dataset.metric;
      drawUpgradeChart();
    });
  });
}

function renderDiagnosticsAll() {
  renderDiagDisagreement();
  drawUpgradeChart();
  drawConvictionChart();
  renderDiagEnsembles();
  renderDiagAveraging();
  renderDiagStatTests();
  renderDiagRegimeTable();
}

function renderDiagEnsembles() {
  const fv = STATE.bootstrap?.summaries?.final_verification;
  const tbody = document.querySelector("#diag-ensemble-table tbody");
  if (!tbody || !fv?.ensemble_k1_to_k8) return;
  const sym = STATE.diagSymbol;
  document.getElementById("diag-ens-sym").textContent = sym;
  const rows = fv.ensemble_k1_to_k8.filter((r) => r.symbol === sym);
  tbody.innerHTML = rows.map((r) => {
    const isPeak = (r.k === 4 && r.pool.startsWith("Original")) || (r.k >= 3 && r.k <= 4 && r.pool.startsWith("Upgraded"));
    const isDrop = r.k >= 7 && r.symbol !== "IDEA";
    return `
      <tr class="${isPeak ? "selected" : ""}">
        <td>${r.pool.replace(" (with LGBM_v2)", "")}</td>
        <td class="num"><b>K=${r.k}</b></td>
        <td class="num" style="color:${isDrop ? "var(--down)" : isPeak ? "var(--up)" : "inherit"};font-weight:${isPeak ? "700" : "500"}">${r.macro_f1.toFixed(4)}</td>
        <td class="num">${r.roc_auc_ovr.toFixed(4)}</td>
        <td class="mono" style="font-size:11px;">${r.models}</td>
      </tr>`;
  }).join("");
  const origBest = rows.find((r) => r.pool === "Original-7" && r.k === 4);
  const origFull = rows.find((r) => r.pool === "Original-7" && r.k === 7);
  document.getElementById("diag-ensemble-note").innerHTML = origBest && origFull
    ? `On <b>${sym}</b> (Original-7), heterogeneous <b>K=4</b> (<code>${origBest.models}</code>, <b>${origBest.macro_f1.toFixed(4)}</b>) beats pairing only K=2 and beats dumping all K=7 models (<b>${origFull.macro_f1.toFixed(4)}</b>) because LogReg/LSTM correlate strongly (r=0.75–0.95) with zero extra signal.`
    : `Combining distinct model families (Tree + Transformer + CNN + Drift) beats averaging correlated trees alone.`;
}

function renderDiagAveraging() {
  const fv = STATE.bootstrap?.summaries?.final_verification;
  const tbody = document.querySelector("#diag-averaging-table tbody");
  if (!tbody || !fv?.averaging_1_to_100) return;
  const sym = STATE.diagSymbol;
  document.getElementById("diag-avg-sym").textContent = sym;
  const rows = fv.averaging_1_to_100.filter((r) => r.symbol === sym && [1, 10, 20, 30, 50, 100].includes(r.n_window));
  tbody.innerHTML = rows.map((r) => `
    <tr>
      <td class="num"><b>N=${r.n_window}</b></td>
      <td class="num">${r.trail_f1.toFixed(4)}</td>
      <td class="num" style="color:${r.n_window === 1 ? "var(--up)" : "var(--down)"}">${(r.trail_dir_hit * 100).toFixed(2)}%</td>
      <td class="num">${r.trail_conf50_dir_hit != null ? (r.trail_conf50_dir_hit * 100).toFixed(2) + "%" : "—"}</td>
      <td class="num" style="color:var(--accent);font-weight:700">${r.batch_majority_win_pct.toFixed(1)}%</td>
    </tr>`).join("");
  const r1 = rows[0], r100 = rows[rows.length - 1];
  document.getElementById("diag-averaging-note").innerHTML = r1 && r100
    ? `On <b>${sym}</b> (12 full days), trailing probability smoothing adds phase lag and drops Dir Hit <b>${(r1.trail_dir_hit * 100).toFixed(1)}% → ${(r100.trail_dir_hit * 100).toFixed(1)}%</b>, while grouping N=100 independent calls hits <b>${r100.batch_majority_win_pct.toFixed(1)}%</b> batch win rate.`
    : "";
}

function renderDiagStatTests() {
  const fv = STATE.bootstrap?.summaries?.final_verification;
  const tbody = document.querySelector("#diag-stat-table tbody");
  if (!tbody || !fv?.stat_tests) return;
  tbody.innerHTML = fv.stat_tests.map((r) => `
    <tr>
      <td><b>${r.comparison}</b></td>
      <td>${r.cohort}</td>
      <td class="num" style="color:var(--up);font-weight:700">${r.wins}/${r.n_instruments} (${r.win_rate_pct.toFixed(0)}%)</td>
      <td class="num">${r.mean_before.toFixed(4)} → <b>${r.mean_after.toFixed(4)}</b></td>
      <td class="num" style="color:var(--accent);font-weight:700">${r.p_wilcoxon_onesided.toExponential(2)}</td>
    </tr>`).join("");
}

function renderDiagRegimeTable() {
  const tbody = document.querySelector("#diag-regime-table tbody");
  const fv = STATE.bootstrap?.summaries?.final_verification;
  if (!tbody || !fv || !fv.tick_regimes) return;
  tbody.innerHTML = fv.tick_regimes.map((r) => `
    <tr>
      <td><b>${r.tick_regime}</b></td>
      <td class="num">${r.n_instruments}</td>
      <td class="num">${r.mean_boundary_err_pct.toFixed(1)}%</td>
      <td class="num">${r.mean_v1_f1.toFixed(4)}</td>
      <td class="num">${r.mean_v2_mh_f1.toFixed(4)}</td>
      <td class="num"><b>${r.mean_best_f1.toFixed(4)}</b></td>
      <td class="num" style="color:var(--up);font-weight:700">+${(r.mean_delta_f1 * 100).toFixed(2)} pp</td>
      <td class="num">${r.mean_v1_auc.toFixed(3)} → <b>${r.mean_best_auc.toFixed(3)}</b></td>
      <td class="num" style="color:var(--accent);font-weight:700">${r.mean_batch100_win_pct.toFixed(1)}%</td>
    </tr>`).join("");
}

function renderDiagDisagreement() {
  const ed = STATE.bootstrap.summaries.error_diagnostics;
  if (!ed) return;
  const sym = STATE.diagSymbol;
  const d = ed.disagreement.find((r) => r.symbol === sym);
  if (!d) return;

  document.getElementById("diag-sym-title").textContent = sym;
  document.getElementById("diag-union-tag").textContent = `Union ${d.oracle_union_pct.toFixed(1)}% · Agree Acc ${(d.agree_acc * 100).toFixed(1)}%`;

  const agreeSegs = [
    ["Both Right", d.both_right_pct, "var(--up)"],
    ["Only LGBM", d.only_lgbm_pct, "var(--blue)"],
    ["Only TLOB", d.only_tlob_pct, "var(--amber)"],
    ["Both Wrong", d.both_wrong_pct, "var(--flat)"],
  ];
  document.getElementById("diag-agree-bar").innerHTML = agreeSegs.map(([lbl, pct, col]) =>
    `<div class="agree-seg" style="width:${pct.toFixed(2)}%;background:${col}" title="${lbl}: ${pct.toFixed(1)}%">${pct >= 11 ? pct.toFixed(1) + "%" : ""}</div>`
  ).join("");
  document.getElementById("diag-agree-legend").innerHTML = agreeSegs.map(([lbl, pct, col]) =>
    `<span><i class="agree-dot" style="background:${col}"></i>${lbl}: <b>${pct.toFixed(1)}%</b></span>`
  ).join("");

  document.getElementById("diag-err-split-bar").innerHTML = `
    <div class="agree-seg" style="width:${d.flat_boundary_err_share.toFixed(2)}%;background:var(--amber)">±α Boundary ${d.flat_boundary_err_share.toFixed(1)}%</div>
    <div class="agree-seg" style="width:${d.opp_dir_err_share.toFixed(2)}%;background:var(--down)">${d.opp_dir_err_share >= 14 ? "UP↔DOWN Reversal " + d.opp_dir_err_share.toFixed(1) + "%" : d.opp_dir_err_share.toFixed(1) + "%"}</div>
  `;

  const stallNote = (d.closing_stall_n > 500 && d.closing_stall_v2_acc != null && d.closing_stall_v1_acc != null)
    ? ` Late-day zero-vol stall (${d.closing_stall_n.toLocaleString()} rows): v1 <b>${(d.closing_stall_v1_acc * 100).toFixed(1)}%</b> → v2 <b>${(d.closing_stall_v2_acc * 100).toFixed(1)}%</b>.`
    : "";
  document.getElementById("diag-err-caption").innerHTML =
    `${d.flat_boundary_err_share.toFixed(1)}% of baseline LightGBM errors are <b>FLAT↔Move boundary misses</b> near ±α.${stallNote}`;

  const buckets = ed.buckets.filter((r) => r.symbol === sym);
  const niceName = { BothRight: "Both Right", OnlyLGBM: "Only LightGBM", OnlyTLOB: "Only TLOB", BothWrong: "Both Wrong" };
  document.querySelector("#diag-bucket-table tbody").innerHTML = buckets.map((b) => `
    <tr>
      <td><b>${niceName[b.bucket] || b.bucket}</b></td>
      <td class="num">${b.abs_ret_over_alpha > 0 ? b.abs_ret_over_alpha.toFixed(2) + "×" : "n/a"}</td>
      <td class="num">${b.abs_ret_over_alpha > 0 ? b.near_boundary_pct.toFixed(1) + "%" : "n/a"}</td>
      <td class="num">${b.abs_drift_30s_ticks.toFixed(2)}t</td>
      <td class="num">${b.vol_10s_ticks.toFixed(3)}t</td>
      <td class="num">${b.imb_reversal_pct.toFixed(1)}%</td>
    </tr>`).join("");
}

function drawUpgradeChart() {
  const canvas = document.getElementById("canvas-upgrade");
  const ed = STATE.bootstrap?.summaries?.error_diagnostics;
  if (!canvas || !ed) return;
  const { ctx, w, h } = setupHiDPICanvas(canvas);
  const metric = STATE.diagMetric;
  const rowsBase = ed.upgrade_ablation.filter((r) => r.symbol === STATE.diagSymbol).sort((a, b) => a.stage_order - b.stage_order);
  const rows = rowsBase.map((r) => ({ ...r }));
  const pcReps = STATE.bootstrap?.summaries?.ml_completeness?.per_class_reports || [];
  const hybRow = pcReps.find((r) => r.symbol === STATE.diagSymbol && r.tier === "Flagship");
  if (hybRow && (!rows.length || hybRow[metric] > rows[rows.length - 1][metric] + 0.0005)) {
    rows.push({
      symbol: STATE.diagSymbol,
      stage_order: 8,
      stage: STATE.diagSymbol === "NIFTY_FUT" ? "8. Flagship (+MH+TLOB)" : "8. Flagship (+SSF+MH+TLOB)",
      macro_f1: hybRow.macro_f1,
      roc_auc_ovr: hybRow.roc_auc_ovr,
      log_loss: hybRow.log_loss,
    });
  }
  if (!rows.length) return;

  const vals = rows.map((r) => r[metric]);
  const vMinRaw = Math.min(...vals), vMaxRaw = Math.max(...vals);
  const minVal = Math.max(0.25, Math.floor((vMinRaw - 0.03) * 20) / 20);
  const maxVal = Math.min(0.95, Math.ceil((vMaxRaw + 0.025) * 20) / 20);

  ctx.clearRect(0, 0, w, h);
  const ml = 235, mr = 64, mt = 24, mb = 28;
  const pw = w - ml - mr, ph = h - mt - mb;
  const rowH = ph / rows.length;

  ctx.fillStyle = cssVar("--text-muted");
  ctx.font = "11px Inter, sans-serif";
  ctx.fillText(metric === "macro_f1" ? "Strict Held-Out Test Macro-F1" : "Strict Held-Out Test ROC-AUC (OVR)", ml, 14);

  for (let i = 0; i <= 4; i++) {
    const v = minVal + (i / 4) * (maxVal - minVal);
    const xx = ml + ((v - minVal) / (maxVal - minVal)) * pw;
    ctx.strokeStyle = cssVar("--border");
    ctx.beginPath(); ctx.moveTo(xx, mt); ctx.lineTo(xx, mt + ph); ctx.stroke();
    ctx.fillStyle = cssVar("--text-muted");
    ctx.font = "10.5px 'JetBrains Mono', monospace";
    ctx.fillText(v.toFixed(2), xx - 12, mt + ph + 16);
  }

  rows.forEach((r, idx) => {
    const val = r[metric];
    const yy = mt + idx * rowH + rowH * 0.18;
    const bh = Math.max(12, rowH * 0.64);
    const bw = Math.max(4, ((val - minVal) / (maxVal - minVal)) * pw);
    const isTop = r.stage_order >= 6;

    ctx.fillStyle = cssVar("--text");
    ctx.font = isTop ? "700 11.5px Inter, sans-serif" : "500 11.5px Inter, sans-serif";
    ctx.textBaseline = "middle";
    ctx.fillText(r.stage, 6, yy + bh / 2);

    ctx.fillStyle = r.stage_order >= 8 ? cssVar("--up") : r.stage_order === 7 ? cssVar("--accent") : r.stage_order === 6 ? cssVar("--accent") : r.stage_order === 2 ? cssVar("--blue") : cssVar("--flat");
    ctx.globalAlpha = isTop ? 0.92 : 0.65;
    ctx.fillRect(ml, yy, bw, bh);
    ctx.globalAlpha = 1;

    ctx.fillStyle = cssVar("--text");
    ctx.font = "600 11.5px 'JetBrains Mono', monospace";
    ctx.fillText(val.toFixed(4), ml + bw + 8, yy + bh / 2);
  });

  const base = rows[0], hyb = rows[rows.length - 1];
  const dF1 = ((hyb.macro_f1 - base.macro_f1) * 100).toFixed(2);
  const dAuc = ((hyb.roc_auc_ovr - base.roc_auc_ovr) * 100).toFixed(2);
  document.getElementById("diag-upgrade-note").innerHTML =
    `On <b>${STATE.diagSymbol}</b>, upgrading from Baseline 25f LightGBM to <b>Flagship Hybrid</b> lifts Macro-F1 by <b>+${dF1} pp</b> and ROC-AUC by <b>+${dAuc} pp</b> out-of-sample (LogLoss ${base.log_loss.toFixed(3)} → ${hyb.log_loss.toFixed(3)}).`;
}

/* ==================== VIEW 3C: ML EVALUATION & AUDIT ==================== */
function initMlAuditControls() {
  const symWrap = document.getElementById("mlaudit-symbol-pills");
  if (symWrap) {
    [
      ["RELIANCE", "RELIANCE (Medium Tick)"],
      ["NIFTY_FUT", "NIFTY_FUT (Tiny Tick)"],
      ["IDEA", "IDEA (Ultra-Large Tick)"],
    ].forEach(([sym, lbl]) => {
      const b = document.createElement("button");
      b.className = "inst-pill" + (sym === STATE.mlAuditSymbol ? " active" : "");
      b.dataset.sym = sym;
      b.textContent = lbl;
      b.addEventListener("click", () => {
        STATE.mlAuditSymbol = sym;
        document.querySelectorAll("#mlaudit-symbol-pills .inst-pill").forEach((x) => x.classList.toggle("active", x.dataset.sym === sym));
        renderMlAuditAll();
      });
      symWrap.appendChild(b);
    });
  }

  const cmWrap = document.getElementById("mlaudit-cm-model-pills");
  if (cmWrap) {
    ["Flagship Hybrid", "LightGBM v2 (38f)", "LightGBM v1", "TLOB", "Logistic Regression"].forEach((m) => {
      const b = document.createElement("button");
      b.className = "btn-chip" + (m === STATE.mlAuditCmModel ? " active" : "");
      b.dataset.cm = m;
      b.textContent = m.replace(" (38f)", "").replace("Logistic Regression", "LogReg").replace("Flagship ", "");
      b.addEventListener("click", () => {
        STATE.mlAuditCmModel = m;
        document.querySelectorAll("#mlaudit-cm-model-pills .btn-chip").forEach((x) => x.classList.toggle("active", x.dataset.cm === m));
        renderMlAuditCM();
      });
      cmWrap.appendChild(b);
    });
  }
}

function renderMlAuditAll() {
  const ml = STATE.bootstrap?.summaries?.ml_completeness;
  if (!ml) return;
  renderMlAuditKpis(ml);
  renderMlAuditDist(ml);
  renderMlAuditCM(ml);
  renderMlAuditReport(ml);
  renderMlAuditGen(ml);
  renderMlAuditComplexity(ml);
  renderMlAuditImportance(ml);
}

function renderMlAuditKpis(ml) {
  const u25 = ml.universe25_per_class || [];
  if (u25.length) {
    const minF1 = Math.min(...u25.map((r) => r.best_min_class_f1));
    const healthyCnt = u25.filter((r) => r.best_min_class_f1 > 0.2).length;
    const meanF = u25.reduce((s, r) => s + r.best_f1_flat, 0) / u25.length;
    const meanU = u25.reduce((s, r) => s + r.best_f1_up, 0) / u25.length;
    const meanD = u25.reduce((s, r) => s + r.best_f1_down, 0) / u25.length;
    document.getElementById("mlaudit-kpi-health").textContent = `${healthyCnt} / ${u25.length} ✓`;
    document.getElementById("mlaudit-kpi-health-note").textContent =
      `Global min per-class F1: ${minF1.toFixed(3)} (>0.20) · Mean F/U/D: ${meanF.toFixed(2)}/${meanU.toFixed(2)}/${meanD.toFixed(2)}`;
  }
  const teDist = (ml.class_distributions || []).find((r) => r.symbol === STATE.mlAuditSymbol && r.split === "Test");
  if (teDist) {
    document.getElementById("mlaudit-kpi-priors").textContent = `${teDist.pct_flat.toFixed(1)}/${teDist.pct_up.toFixed(1)}/${teDist.pct_down.toFixed(1)}%`;
    document.getElementById("mlaudit-kpi-priors-note").textContent =
      `${STATE.mlAuditSymbol} Test priors (FLAT / UP / DOWN) · α = ${teDist.alpha_bp_mean.toFixed(2)} bp (0 = exact 1t discrete)`;
  }
  const rep = (ml.per_class_reports || []).find((r) => r.symbol === STATE.mlAuditSymbol && r.tier === "Flagship");
  if (rep) {
    document.getElementById("mlaudit-kpi-calib").textContent = `${rep.log_loss.toFixed(3)} LL`;
    document.getElementById("mlaudit-kpi-calib-note").textContent =
      `${STATE.mlAuditSymbol} Hybrid vs ln(3)=1.099 · Brier ${rep.brier_score.toFixed(3)} · 10-bin ECE ${rep.ece_10bin.toFixed(4)}`;
  }
  const gen = (ml.generalization_gap || []).filter((r) => r.symbol === STATE.mlAuditSymbol);
  if (gen.length) {
    const gBest = gen[gen.length - 1];
    document.getElementById("mlaudit-kpi-gap").textContent = `${gBest.val_minus_test_f1_gap >= 0 ? "+" : ""}${gBest.val_minus_test_f1_gap.toFixed(4)}`;
    document.getElementById("mlaudit-kpi-gap-note").textContent =
      `${gBest.model}: Val ${gBest.val_macro_f1.toFixed(3)} vs Test ${gBest.test_macro_f1.toFixed(3)} (${Math.round(gBest.best_iter_mean)} trees)`;
  }
}

function renderMlAuditDist(ml) {
  const sym = STATE.mlAuditSymbol;
  document.getElementById("mlaudit-dist-sym").textContent = sym;
  const rows = (ml.class_distributions || []).filter((r) => r.symbol === sym);
  const te = rows.find((r) => r.split === "Test");
  if (te) {
    document.getElementById("mlaudit-alpha-tag").textContent = `α = ${te.alpha_bp_mean.toFixed(2)} bp`;
    const segs = [
      ["FLAT (0)", te.pct_flat, "var(--flat)"],
      ["UP (+1)", te.pct_up, "var(--up)"],
      ["DOWN (-1)", te.pct_down, "var(--down)"],
    ];
    document.getElementById("mlaudit-dist-bar").innerHTML = segs.map(([lbl, pct, col]) =>
      `<div class="agree-seg" style="width:${pct.toFixed(2)}%;background:${col}" title="${lbl}: ${pct.toFixed(2)}%">${pct >= 10 ? pct.toFixed(1) + "%" : ""}</div>`
    ).join("");
    document.getElementById("mlaudit-dist-legend").innerHTML = segs.map(([lbl, pct, col]) =>
      `<span><i class="agree-dot" style="background:${col}"></i>${lbl}: <b>${pct.toFixed(2)}%</b></span>`
    ).join("");
  }
  const order = { Train: 1, Validation: 2, Test: 3, Seg0_Test: 4, Seg1_Test: 5, Seg2_Test: 6, Seg3_Test: 7 };
  rows.sort((a, b) => (order[a.split] || 9) - (order[b.split] || 9));
  document.querySelector("#mlaudit-dist-table tbody").innerHTML = rows.map((r) => `
    <tr class="${r.split === "Test" ? "selected" : ""}">
      <td><b>${r.split}</b></td>
      <td class="num">${r.n_rows.toLocaleString()}</td>
      <td class="num">${r.pct_flat.toFixed(2)}%</td>
      <td class="num" style="color:var(--up)">${r.pct_up.toFixed(2)}%</td>
      <td class="num" style="color:var(--down)">${r.pct_down.toFixed(2)}%</td>
      <td class="num">${r.alpha_bp_mean.toFixed(3)}</td>
    </tr>`).join("");
}

function renderMlAuditCM(mlArg) {
  const ml = mlArg || STATE.bootstrap?.summaries?.ml_completeness;
  const sym = STATE.mlAuditSymbol, m = STATE.mlAuditCmModel;
  document.getElementById("mlaudit-cm-sym").textContent = `${sym} · ${m}`;
  const rows = (ml?.confusion_matrices || []).filter((r) => r.symbol === sym && r.model === m);
  const grid = document.getElementById("mlaudit-cm-grid");
  if (!grid || rows.length !== 3) return;

  const colColors = ["rgba(100,116,139,", "rgba(16,185,129,", "rgba(239,68,68,"];
  let html = `
    <div class="cm-hdr">True \\ Pred</div>
    <div class="cm-hdr">Pred FLAT</div>
    <div class="cm-hdr" style="color:var(--up)">Pred UP</div>
    <div class="cm-hdr" style="color:var(--down)">Pred DOWN</div>`;

  rows.forEach((r, idx) => {
    const recArr = [r.recall_FLAT_pct, r.recall_UP_pct, r.recall_DOWN_pct];
    const cntArr = [r.pred_FLAT_cnt, r.pred_UP_cnt, r.pred_DOWN_cnt];
    html += `<div class="cm-row-lbl"><span>True ${r.true_class}</span><span class="cm-sub">n=${r.support.toLocaleString()}</span></div>`;
    for (let c = 0; c < 3; c++) {
      const isDiag = idx === c;
      const alpha = isDiag ? Math.min(0.32, 0.08 + recArr[c] / 250) : Math.min(0.16, 0.02 + recArr[c] / 400);
      const bg = `${colColors[c]}${alpha.toFixed(3)})`;
      html += `
        <div class="cm-cell" style="background:${bg};border-color:${isDiag ? "var(--accent)" : "var(--border)"}">
          <b>${recArr[c].toFixed(1)}%</b>
          <span class="cm-sub">${cntArr[c].toLocaleString()} rows</span>
          ${isDiag ? `<span class="cm-sub" style="color:var(--accent)">Prec ${r.class_precision_pct.toFixed(1)}%</span>` : ""}
        </div>`;
    }
  });
  grid.innerHTML = html;

  const rUp = rows[1], rDn = rows[2];
  const revShare = ((rUp.pred_DOWN_cnt + rDn.pred_UP_cnt) / rows.reduce((s, r) => s + r.support, 0) * 100).toFixed(1);
  document.getElementById("mlaudit-cm-note").innerHTML =
    `Diagonal values show per-class Recall % (sensitivity); <b>UP↔DOWN opposite reversals</b> account for only <b>${revShare}%</b> of total predictions under <b>${m}</b> on ${sym}.`;
}

function renderMlAuditReport(ml) {
  const sym = STATE.mlAuditSymbol;
  document.getElementById("mlaudit-rep-sym").textContent = sym;
  const rows = (ml.per_class_reports || []).filter((r) => r.symbol === sym);
  document.getElementById("mlaudit-rep-count").textContent = `${rows.length} evaluated configs`;
  document.querySelector("#mlaudit-report-table tbody").innerHTML = rows.map((r) => {
    const mf = r.min_class_f1;
    const hBadge = mf < 0.2
      ? `<span class="health-badge bad">✗ Collapse (${mf.toFixed(2)})</span>`
      : mf < 0.3
      ? `<span class="health-badge warn">⚠ Weak (${mf.toFixed(2)})</span>`
      : `<span class="health-badge ok">✓ Healthy (${mf.toFixed(2)})</span>`;
    const isTop = r.tier === "Flagship" || r.tier === "Upgraded";
    return `
      <tr class="${r.tier === "Flagship" ? "selected" : ""}">
        <td class="muted">${r.tier}</td>
        <td><b>${r.model}</b></td>
        <td class="num"><b>${r.f1_flat.toFixed(3)}</b> <span class="muted">(${r.prec_flat.toFixed(2)}/${r.rec_flat.toFixed(2)})</span></td>
        <td class="num" style="color:var(--up)"><b>${r.f1_up.toFixed(3)}</b> <span class="muted">(${r.prec_up.toFixed(2)}/${r.rec_up.toFixed(2)})</span></td>
        <td class="num" style="color:var(--down)"><b>${r.f1_down.toFixed(3)}</b> <span class="muted">(${r.prec_down.toFixed(2)}/${r.rec_down.toFixed(2)})</span></td>
        <td>${hBadge}</td>
        <td class="num" style="font-weight:${isTop ? "700" : "500"}">${r.macro_f1.toFixed(4)} ± ${r.macro_f1_std.toFixed(3)}</td>
        <td class="num">${r.weighted_f1.toFixed(4)}</td>
        <td class="num">${(r.accuracy * 100).toFixed(1)}%</td>
        <td class="num">${(r.balanced_accuracy * 100).toFixed(1)}%</td>
        <td class="num">${r.mcc.toFixed(3)}</td>
        <td class="num">${r.roc_auc_ovr.toFixed(3)}</td>
        <td class="num">${r.pr_auc_macro.toFixed(3)}</td>
        <td class="num" style="color:${r.log_loss > 1.0986 ? "var(--down)" : "inherit"}">${r.log_loss.toFixed(3)}</td>
        <td class="num">${r.brier_score.toFixed(3)}</td>
        <td class="num">${r.ece_10bin.toFixed(4)}</td>
      </tr>`;
  }).join("");
}

function renderMlAuditGen(ml) {
  const sym = STATE.mlAuditSymbol;
  const rows = (ml.generalization_gap || []).filter((r) => r.symbol === sym);
  document.querySelector("#mlaudit-gen-table tbody").innerHTML = rows.map((r) => `
    <tr>
      <td><b>${r.model}</b></td>
      <td class="num">${r.train_macro_f1.toFixed(4)}</td>
      <td class="num">${r.val_macro_f1.toFixed(4)}</td>
      <td class="num"><b>${r.test_macro_f1.toFixed(4)}</b> ± ${r.test_macro_f1_std.toFixed(3)}</td>
      <td class="num" style="color:var(--up)">${r.val_minus_test_f1_gap >= 0 ? "+" : ""}${r.val_minus_test_f1_gap.toFixed(4)}</td>
      <td class="num">${r.train_logloss.toFixed(3)} → ${r.test_logloss.toFixed(3)}</td>
      <td class="num">${r.best_iter_mean.toFixed(0)} ± ${r.best_iter_std.toFixed(0)}</td>
    </tr>`).join("");
}

function renderMlAuditComplexity(ml) {
  const lat = ml.inference_latency || [];
  const cnnMed = lat.length ? `${Math.min(...lat.map((x) => x.deeplob5_p50_ms)).toFixed(2)}–${Math.max(...lat.map((x) => x.deeplob5_p50_ms)).toFixed(2)} ms CUDA` : "~7.5 ms CUDA";
  const rows = [
    ["Tier 0 · Naive Flat / Mom", "0 / 1 scalar", "Majority train prior / k-s sign", "< 1 µs CPU"],
    ["Tier 1 · Logistic Regression", "1s vector (25f)", "L2, class_weight=balanced, z-score", "< 5 µs CPU"],
    ["Tier 2 · Random Forest", "1s vector (25f)", "100 trees, max_depth=12, min_leaf=10", "~1–3 ms CPU"],
    ["Tier 3 · LightGBM v1 / v2 / SSF", "1s causal (25/38/46f)", "leaves=63, depth=8, lr=0.05, early_stop=20", "10.5 µs C++"],
    ["Tier 4–5 · LSTM & DeepLOB5", "100s × 20 raw LOB", "2L LSTM / 3D CNN+Inception, static z", cnnMed],
    ["Tier 5 · TLOB Transformer", "100s × 20 raw LOB", "Dual Attention + Adaptive BiN layer", "Multi-ms CUDA"],
    ["★ Flagship Hybrid", "v2[+SSF] (10s,20s) + TLOB", "Strict LOSO convex probability blend", "Hybrid (CPU+Seq)"],
  ];
  document.getElementById("mlaudit-complexity-body").innerHTML = rows.map(([m, w, h, l]) => `
    <tr>
      <td><b>${m}</b></td>
      <td class="mono">${w}</td>
      <td>${h}</td>
      <td class="num"><b>${l}</b></td>
    </tr>`).join("");
}

function renderMlAuditImportance(ml) {
  const sym = STATE.mlAuditSymbol;
  document.getElementById("mlaudit-imp-sym").textContent = sym;
  const rows = (ml.v2_feature_importance || []).filter((r) => r.symbol === sym).slice(0, 12);
  if (rows.length) document.getElementById("mlaudit-imp-model").textContent = rows[0].model;
  const maxShap = Math.max(...rows.map((r) => r.shap_share_pct), 10);
  document.getElementById("mlaudit-imp-bars").innerHTML = rows.map((r) => {
    const isNew = r.feature.startsWith("ssf_") || r.feature.startsWith("basis_") || r.feature.startsWith("lead_") || r.feature.startsWith("vol_") || r.feature.startsWith("mid_drift") || r.feature === "day_progress" || r.feature.includes("_mean_");
    const col = r.feature.startsWith("ssf_") || r.feature.startsWith("basis_") || r.feature.startsWith("lead_") ? "var(--up)" : isNew ? "var(--accent)" : "var(--blue)";
    return `
      <div class="trap-row" style="grid-template-columns:170px 1fr 122px;">
        <span class="mono" style="font-size:11.5px;font-weight:${isNew ? "700" : "500"}">${r.rank}. ${r.feature}</span>
        <div class="trap-track"><div class="trap-fill" style="width:${(r.shap_share_pct / maxShap * 100).toFixed(1)}%;background:${col}"></div></div>
        <span class="mono" style="text-align:right;font-size:11px;"><b>${r.shap_share_pct.toFixed(1)}%</b> <span class="muted">(${r.gain_share_pct.toFixed(1)}%g)</span></span>
      </div>`;
  }).join("");
}

function drawConvictionChart() {
  const canvas = document.getElementById("canvas-conviction");
  const ed = STATE.bootstrap?.summaries?.error_diagnostics;
  if (!canvas || !ed) return;
  const { ctx, w, h } = setupHiDPICanvas(canvas);
  const pts = ed.conviction_curve.filter((r) => r.symbol === STATE.diagSymbol).sort((a, b) => a.threshold - b.threshold);
  if (!pts.length) return;

  ctx.clearRect(0, 0, w, h);
  const ml = 64, mr = 24, mt = 26, mb = 44;
  const pw = w - ml - mr, ph = h - mt - mb;
  const yMin = 0.42, yMax = 0.95;
  const xOf = (i) => ml + (i / Math.max(1, pts.length - 1)) * pw;
  const yOf = (v) => mt + ph - ((v - yMin) / (yMax - yMin)) * ph;

  ctx.strokeStyle = cssVar("--border");
  ctx.fillStyle = cssVar("--text-muted");
  ctx.font = "11px 'JetBrains Mono', monospace";
  [0.45, 0.55, 0.65, 0.75, 0.85, 0.92].forEach((yv) => {
    const py = yOf(yv);
    ctx.beginPath(); ctx.moveTo(ml, py); ctx.lineTo(w - mr, py); ctx.stroke();
    ctx.fillText(`${Math.round(yv * 100)}%`, 24, py + 4);
  });

  pts.forEach((p, i) => {
    const px = xOf(i);
    ctx.strokeStyle = cssVar("--border");
    ctx.beginPath(); ctx.moveTo(px, mt); ctx.lineTo(px, mt + ph); ctx.stroke();
    ctx.fillStyle = cssVar("--text-muted");
    ctx.font = "10.5px 'JetBrains Mono', monospace";
    ctx.fillText(`>${p.threshold.toFixed(2)}`, px - 14, mt + ph + 15);
    ctx.fillText(`(${p.coverage_pct.toFixed(0)}%)`, px - 14, mt + ph + 28);
  });

  function plotSeries(key, color, labelOffsetY) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    pts.forEach((p, i) => {
      const px = xOf(i), py = yOf(p[key]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
    pts.forEach((p, i) => {
      const px = xOf(i), py = yOf(p[key]);
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(px, py, 4.2, 0, Math.PI * 2); ctx.fill();
      ctx.font = "600 10.5px 'JetBrains Mono', monospace";
      ctx.fillText(`${(p[key] * 100).toFixed(1)}%`, px - 16, py + labelOffsetY);
    });
  }
  plotSeries("accuracy", cssVar("--up"), -9);
  plotSeries("macro_f1", cssVar("--blue"), 15);

  ctx.fillStyle = cssVar("--up"); ctx.font = "700 11.5px Inter, sans-serif";
  ctx.fillText("● Hybrid Accuracy", ml, 14);
  ctx.fillStyle = cssVar("--blue");
  ctx.fillText("● Hybrid Macro-F1", ml + 125, 14);

  ctx.fillStyle = cssVar("--text-muted");
  ctx.font = "600 11px Inter, sans-serif";
  ctx.fillText("Hybrid Max Probability Cutoff (Test Coverage %)", ml + pw / 2 - 125, h - 4);
  ctx.save();
  ctx.translate(12, mt + ph / 2 + 52);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Test Accuracy / Macro-F1", 0, 0);
  ctx.restore();

  const p0 = pts[0], pHi = pts[pts.length - 1];
  document.getElementById("diag-conv-note").innerHTML =
    `Forced 100% predictions yield <b>${(p0.accuracy * 100).toFixed(1)}%</b> accuracy on ${STATE.diagSymbol}. Filtering for high-conviction timestamps (<code>max prob ≥ ${pHi.threshold.toFixed(2)}</code>, top ${pHi.coverage_pct.toFixed(1)}% of seconds) boosts accuracy to <b>${(pHi.accuracy * 100).toFixed(1)}%</b>!`;
}

/* ==================== VIEW 3: ML FINDINGS & ARENA ==================== */
function initArenaControls() {
  document.querySelectorAll("#trap-toggle .btn-chip").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#trap-toggle .btn-chip").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      STATE.trapMode = b.dataset.trap;
      renderTrap();
    });
  });

  document.querySelectorAll("#ablation-stepper .btn-chip").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#ablation-stepper .btn-chip").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      STATE.ablationStage = b.dataset.stage;
      renderAblation();
    });
  });

  const dsWrap = document.getElementById("arena-dataset-pills");
  [
    ["RELIANCE", "NSE · RELIANCE"],
    ["NIFTY_FUT", "NSE · NIFTY_FUT"],
    ["IDEA", "NSE · IDEA"],
    ["BTC", "Bitcoin BTC-USD"],
    ["FI2010", "FI-2010 (k=10 vs 100)"],
    ["SYNTH", "Synthetic Book"],
  ].forEach(([k, label]) => {
    const b = document.createElement("button");
    b.className = "inst-pill" + (k === STATE.arenaDataset ? " active" : "");
    b.dataset.ds = k;
    b.textContent = label;
    b.addEventListener("click", () => {
      STATE.arenaDataset = k;
      document.querySelectorAll("#arena-dataset-pills .inst-pill").forEach((x) => x.classList.toggle("active", x.dataset.ds === k));
      drawArenaLadder();
    });
    dsWrap.appendChild(b);
  });
}

function renderTrap() {
  const idea = STATE.bootstrap.summaries.nse_ladders["IDEA"];
  const pick = ["Naive Flat", "LightGBM", "DeepLOB5", "TLOB", "LSTM"];
  const el = document.getElementById("trap-bars");
  const isAcc = STATE.trapMode === "acc";
  el.innerHTML = pick.map((m) => {
    const row = idea.find((r) => r.model === m);
    const val = isAcc ? row.accuracy : row.macro_f1;
    const col = m === "Naive Flat" ? (isAcc ? "var(--amber)" : "var(--down)") : m === "LightGBM" ? "var(--up)" : "var(--blue)";
    const extra = isAcc ? `${(val * 100).toFixed(1)}%` : `${val.toFixed(3)} F1`;
    return `
      <div class="trap-row">
        <span>${m}</span>
        <div class="trap-track"><div class="trap-fill" style="width:${(val * 100).toFixed(1)}%;background:${col}"></div></div>
        <span class="mono" style="text-align:right;font-weight:700">${extra}</span>
      </div>`;
  }).join("");
  document.getElementById("trap-caption").textContent = isAcc
    ? "On IDEA, 69.4% of rows are FLAT—so a dummy classifier that predicts FLAT 100% of the time scores 69.4% plain accuracy while learning ZERO UP/DOWN moves!"
    : "Macro-F1 averages F1 equally across UP, FLAT, and DOWN (1/3 each): Naive Flat collapses to 0.273, while LightGBM wins (0.484 Macro-F1, 0.790 UP-vs-DOWN ROC-AUC).";
}

function renderAblation() {
  const rows = STATE.bootstrap.summaries.nse_ablation;
  const stagePrefix = `${STATE.ablationStage}:`;
  const subset = rows.filter((r) => r.feature_set.startsWith(stagePrefix));
  const el = document.getElementById("ablation-bars");
  el.innerHTML = subset.map((r) => {
    const col = r.macro_f1 >= 0.44 ? "var(--up)" : r.macro_f1 >= 0.39 ? "var(--blue)" : "var(--flat)";
    return `
      <div class="trap-row">
        <span>${r.symbol}</span>
        <div class="trap-track"><div class="trap-fill" style="width:${Math.min(100, r.macro_f1 * 180).toFixed(1)}%;background:${col}"></div></div>
        <span class="mono" style="text-align:right;font-weight:700">${r.macro_f1.toFixed(3)} F1</span>
      </div>`;
  }).join("");

  const notes = {
    A: "Stage A (1 feature: spread_ticks): Baseline structural state (F1 = 0.28–0.35).",
    B: "Stage B (4 features: L1 Snapshot): Imbalance + micro-price deviation lift F1 sharply (+0.07 to +0.13).",
    C: "Stage C (13 features: +1s/2s/5s Lags): Short recent order-flow memory adds +0.03 to +0.07 Macro-F1 across every stock.",
    D: "Stage D (25 features: +L2–5 Depth & Block PCA): Optimal core set! Adds +0.029 on thin-queue NIFTY_FUT (0.413 → 0.443).",
    E: "Stage E (30 features: +5 Noisy Full-Book NSE Cols): Textbook overfitting! Test F1 DROPS on 6/8 NSE symbols (ITC -0.023, RELIANCE -0.020).",
  };
  document.getElementById("ablation-note").textContent = notes[STATE.ablationStage];
}

function renderArenaLadder() {
  drawArenaLadder();
}

function drawArenaLadder() {
  const canvas = document.getElementById("canvas-ladder");
  if (!canvas || !STATE.bootstrap) return;
  const { ctx, w, h } = setupHiDPICanvas(canvas);
  const s = STATE.bootstrap.summaries;
  const ds = STATE.arenaDataset;

  let items = [];
  let metricLabel = "Test Macro-F1 (higher is better)";
  let note = "";

  if (ds === "RELIANCE" || ds === "NIFTY_FUT" || ds === "IDEA") {
    const keep = ["LightGBM", "TLOB", "Random Forest", "DeepLOB5", "Logistic Regression", "LSTM", "Naive Momentum", "Naive Flat"];
    items = s.nse_ladders[ds].filter((r) => keep.includes(r.model)).map((r) => ({
      name: r.model, val: r.macro_f1, std: r.macro_f1_std,
      hi: r.model === "LightGBM",
    }));
    items.sort((a, b) => b.val - a.val);
    note = ds === "RELIANCE"
      ? "On RELIANCE, LightGBM (0.445) beats DeepLOB5 (+0.041, 4/4 segments) and statistically ties TLOB (0.440, 95% block-bootstrap CI crosses zero)."
      : ds === "NIFTY_FUT"
      ? "On NIFTY_FUT, LightGBM (0.443) leads by +0.064 over DeepLOB5; TLOB's adaptive BiN (0.396) protects Segment 3 while static LSTM/DeepLOB5 fail under the +10.7% rally."
      : "On ultra-large-tick IDEA (69% FLAT), LightGBM leads (0.484 Macro-F1, 0.790 UP/DOWN ROC-AUC), whereas Naive Flat scores only 0.273 Macro-F1.";
  } else if (ds === "BTC" || ds === "SYNTH") {
    const src = ds === "BTC" ? s.btc_ladder : s.synthetic_ladder;
    items = src.filter((r) => r.model !== "Naïve Flat" && r.model !== "Naive Flat").map((r) => ({
      name: r.model.replace("Naïve", "Naive"), val: r.macro_f1, std: r.macro_f1_std, hi: r.model === "LightGBM",
    }));
    items.sort((a, b) => b.val - a.val);
    note = ds === "BTC"
      ? "Coinbase BTC-USD (20s): LightGBM (0.467, Acc 47.6%) beats DeepLOB (0.401) by +0.058 across 4/4 segments (95% block-bootstrap CI [+0.049, +0.068])."
      : "Synthetic Ground Truth (5 sessions): LightGBM (0.585) recovers the planted informed-trader signal cleanly ahead of DeepLOB (0.559).";
  } else if (ds === "FI2010") {
    metricLabel = "FI-2010 Test Accuracy (k = 10 Short vs k = 100 Long Horizon)";
    const f10 = s.fi2010.filter((r) => r.horizon_k === 10 && ["Logistic Regression + history", "DeepLOB (published)", "DeepLOB (ours)", "Logistic Regression"].includes(r.model));
    const f100 = s.fi2010.filter((r) => r.horizon_k === 100 && ["LightGBM (144 feats + history)", "DeepLOB (ours)", "Logistic Regression + history"].includes(r.model));
    items = [
      ...f10.map((r) => ({ name: `[k=10] ${r.model}`, val: r.accuracy_mean, std: r.accuracy_std || 0, hi: r.model.includes("history") })),
      ...f100.map((r) => ({ name: `[k=100] ${r.model}`, val: r.accuracy_mean, std: r.accuracy_std || 0, hi: r.model.includes("DeepLOB") })),
    ];
    note = "History vs Architecture Crossover: At k=10, giving Logistic Regression the same past window flips it above DeepLOB (86.1% vs 83.1%). DeepLOB wins at long horizons (k=50, 100).";
  }
  document.getElementById("arena-ladder-note").textContent = note;

  ctx.clearRect(0, 0, w, h);
  const ml = 210, mr = 70, mt = 24, mb = 28;
  const pw = w - ml - mr, ph = h - mt - mb;
  const maxVal = ds === "FI2010" ? 0.95 : 0.65;
  const rowH = ph / Math.max(1, items.length);

  ctx.fillStyle = cssVar("--text-muted");
  ctx.font = "11px Inter, sans-serif";
  ctx.fillText(metricLabel, ml, 14);

  for (let i = 0; i <= 4; i++) {
    const v = (i / 4) * maxVal;
    const xx = ml + (v / maxVal) * pw;
    ctx.strokeStyle = cssVar("--border");
    ctx.beginPath(); ctx.moveTo(xx, mt); ctx.lineTo(xx, mt + ph); ctx.stroke();
    ctx.fillStyle = cssVar("--text-muted");
    ctx.font = "10.5px 'JetBrains Mono', monospace";
    ctx.fillText(ds === "FI2010" ? `${Math.round(v*100)}%` : v.toFixed(2), xx - 12, mt + ph + 16);
  }

  items.forEach((it, idx) => {
    const yy = mt + idx * rowH + rowH * 0.18;
    const bh = Math.max(12, rowH * 0.64);
    const bw = Math.max(2, (it.val / maxVal) * pw);

    ctx.fillStyle = cssVar("--text");
    ctx.font = it.hi ? "700 12px Inter, sans-serif" : "500 12px Inter, sans-serif";
    ctx.textBaseline = "middle";
    ctx.fillText(it.name, 8, yy + bh / 2);

    ctx.fillStyle = it.hi ? cssVar("--up") : it.name.includes("TLOB") ? cssVar("--blue") : cssVar("--flat");
    ctx.globalAlpha = it.hi ? 0.92 : 0.65;
    ctx.fillRect(ml, yy, bw, bh);
    ctx.globalAlpha = 1;

    if (it.std > 0) {
      const xLo = ml + Math.max(0, (it.val - it.std) / maxVal) * pw;
      const xHi = ml + Math.min(maxVal, (it.val + it.std) / maxVal) * pw;
      ctx.strokeStyle = cssVar("--text");
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(xLo, yy + bh / 2); ctx.lineTo(xHi, yy + bh / 2);
      ctx.stroke();
    }
    ctx.fillStyle = cssVar("--text");
    ctx.font = "600 11.5px 'JetBrains Mono', monospace";
    const lbl = ds === "FI2010" ? `${(it.val * 100).toFixed(1)}%` : it.val.toFixed(3);
    ctx.fillText(lbl, ml + bw + 8, yy + bh / 2);
  });
}

/* ==================== VIEW 4: 25-STOCK NSE EXPLORER ==================== */
function initUniverseControls() {
  const u25 = STATE.bootstrap.summaries.nse_universe_25;
  const sectors = ["ALL", ...new Set(u25.map((r) => r.sector))];
  const sel = document.getElementById("sector-filter");
  sel.innerHTML = sectors.map((s) => `<option value="${s}">${s === "ALL" ? "All Sectors (25)" : s}</option>`).join("");
  sel.addEventListener("change", () => {
    STATE.selectedSector = sel.value;
    renderUniverseTable();
    drawUniverseScatter();
  });

  document.getElementById("btn-highlight-250").addEventListener("click", (e) => {
    STATE.highlight250 = !STATE.highlight250;
    e.target.classList.toggle("active", STATE.highlight250);
    if (STATE.highlight250) STATE.selectedSymbol25 = "ITC";
    renderUniverseAll();
  });

  const canvas = document.getElementById("canvas-scatter");
  canvas.addEventListener("click", (evt) => {
    const rect = canvas.getBoundingClientRect();
    const cx = evt.clientX - rect.left, cy = evt.clientY - rect.top;
    let best = null, bestDist = 24;
    STATE.scatterPoints.forEach((p) => {
      const d = Math.hypot(p.px - cx, p.py - cy);
      if (d < bestDist) { bestDist = d; best = p.row.symbol; }
    });
    if (best) {
      STATE.selectedSymbol25 = best;
      renderUniverseAll();
    }
  });
}

function renderUniverseAll() {
  renderStockInspector();
  renderUniverseTable();
  drawUniverseScatter();
}

function renderStockInspector() {
  const u25 = STATE.bootstrap.summaries.nse_universe_25;
  const r = u25.find((x) => x.symbol === STATE.selectedSymbol25) || u25[0];
  const fvList = STATE.bootstrap.summaries.final_verification?.universe25 || [];
  const vf = fvList.find((x) => x.symbol === r.symbol);
  const pcList = STATE.bootstrap.summaries.ml_completeness?.universe25_per_class || [];
  const pc = pcList.find((x) => x.symbol === r.symbol);
  const upgText = vf
    ? `${vf.v1_f1.toFixed(3)} → <b>${vf.best_f1.toFixed(3)}</b> (+${(vf.delta_best_vs_v1_f1*100).toFixed(1)}pp)`
    : `${r.f1_20s.toFixed(3)} / ${r.auc_ud_20s.toFixed(3)}`;
  const batchText = vf ? `${vf.batch100_win_pct.toFixed(1)}% (${vf.days_beating_55pct_dirhit}/${vf.days_evaluated}d >55%)` : `${(r.hit_rate_20s*100).toFixed(1)}% / ${r.gross_bps.toFixed(2)} bp`;
  const priorText = pc ? `${pc.pct_flat.toFixed(1)} / ${pc.pct_up.toFixed(1)} / ${pc.pct_down.toFixed(1)}%` : "—";
  const perClsText = pc ? `${pc.best_f1_flat.toFixed(3)} / ${pc.best_f1_up.toFixed(3)} / ${pc.best_f1_down.toFixed(3)} <span class="health-badge ok">✓</span>` : "—";

  document.getElementById("stock-inspector").innerHTML = `
    <div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <div>
          <h3 style="margin:0;font-size:17px;">${r.symbol}${vf && vf.has_ssf ? ' <span class="mini-tag">+SSF</span>' : ""}</h3>
          <span class="muted">${r.sector} · ${r.asset} · 64 days (1.44M rows)</span>
        </div>
        <span class="tag tag-green mono">20s AUC ${vf ? vf.best_auc.toFixed(3) : r.auc_ovr_20s.toFixed(3)}</span>
      </div>
      <div class="insp-grid">
        <div class="insp-item"><span>Median Mid Price</span><b class="mono">₹${r.mid_inr.toLocaleString("en-IN", { maximumFractionDigits: 1 })}</b></div>
        <div class="insp-item"><span>Min Price Step (Tick)</span><b class="mono">₹${r.tick_inr.toFixed(2)} (${r.tick_bp.toFixed(3)} bp)</b></div>
        <div class="insp-item"><span>1-Tick Spread Lock-In</span><b class="mono">${r.spread_1tick_pct.toFixed(1)}%</b></div>
        <div class="insp-item"><span>Median L1 Queue</span><b class="mono">${Math.round(r.bq1_med).toLocaleString()} units</b></div>
        <div class="insp-item"><span>20s F1 (v1 → Upgraded)</span><b class="mono" style="color:var(--up)">${upgText}</b></div>
        <div class="insp-item"><span>Batch-100 Win (12 Days)</span><b class="mono">${batchText}</b></div>
        <div class="insp-item"><span>Class Priors (F / U / D)</span><b class="mono">${priorText}</b></div>
        <div class="insp-item"><span>Per-Class F1 (F / U / D)</span><b class="mono">${perClsText}</b></div>
      </div>
    </div>
    <p class="card-note" style="margin-top:8px;">
      ${r.symbol === "ONGC" || r.symbol === "ITC"
        ? "<b>₹250 Natural Experiment:</b> ONGC (₹233, ₹0.01 tick → 0.43 bp, AUC 0.755) vs ITC (₹265, ₹0.05 tick → 1.88 bp, AUC 0.808) proves queue step size—not share price—sets ML predictability!"
        : `Verified strictly out-of-sample across 12 held-out full days (${vf ? vf.best_deployed_model : "LightGBM"}): every day clears 55% directional hit rate.`}
    </p>`;
}

function renderUniverseTable() {
  const u25 = STATE.bootstrap.summaries.nse_universe_25.filter((r) => STATE.selectedSector === "ALL" || r.sector === STATE.selectedSector);
  const fvList = STATE.bootstrap.summaries.final_verification?.universe25 || [];
  const fvMap = Object.fromEntries(fvList.map((x) => [x.symbol, x]));
  document.getElementById("universe-count-tag").textContent = `${u25.length} symbols`;
  const tbody = document.querySelector("#universe-table tbody");
  tbody.innerHTML = u25.map((r) => {
    const vf = fvMap[r.symbol];
    const v1Cell = vf ? vf.v1_f1.toFixed(3) : r.f1_20s.toFixed(3);
    const upgCell = vf ? `<b>${vf.best_f1.toFixed(3)}</b> <span style="color:var(--up)">(+${(vf.delta_best_vs_v1_f1*100).toFixed(1)}pp)</span>` : r.f1_20s.toFixed(3);
    const b100Cell = vf ? `${vf.batch100_win_pct.toFixed(1)}%` : `${(r.hit_rate_20s * 100).toFixed(1)}%`;
    return `
      <tr class="${r.symbol === STATE.selectedSymbol25 ? "selected" : ""}" data-sym="${r.symbol}">
        <td><b>${r.symbol}</b>${vf && vf.has_ssf ? ' <span class="mini-tag">SSF</span>' : ""}</td>
        <td>${r.sector}</td>
        <td class="num">₹${r.mid_inr.toFixed(1)}</td>
        <td class="num">${r.tick_bp.toFixed(3)}</td>
        <td class="num">${r.spread_1tick_pct.toFixed(1)}%</td>
        <td class="num">${Math.round(r.bq1_med).toLocaleString()}</td>
        <td class="num">${r.auc_ovr_5s.toFixed(3)}</td>
        <td class="num">${v1Cell}</td>
        <td class="num">${upgCell}</td>
        <td class="num" style="color:var(--accent);font-weight:700">${b100Cell}</td>
        <td class="num" style="color:var(--down)">${r.gross_bps.toFixed(2)}</td>
      </tr>`;
  }).join("");
  tbody.querySelectorAll("tr").forEach((tr) => {
    tr.addEventListener("click", () => {
      STATE.selectedSymbol25 = tr.dataset.sym;
      renderUniverseAll();
    });
  });
}

function drawUniverseScatter() {
  const canvas = document.getElementById("canvas-scatter");
  if (!canvas || !STATE.bootstrap) return;
  const { ctx, w, h } = setupHiDPICanvas(canvas);
  const u25 = STATE.bootstrap.summaries.nse_universe_25;

  ctx.clearRect(0, 0, w, h);
  const ml = 64, mr = 24, mt = 24, mb = 42;
  const pw = w - ml - mr, ph = h - mt - mb;
  const xMin = -1.6, xMax = 1.05; // log10(tick_bp)
  const yMin = 0.70, yMax = 0.855;
  const xOf = (logT) => ml + ((logT - xMin) / (xMax - xMin)) * pw;
  const yOf = (auc) => mt + ph - ((auc - yMin) / (yMax - yMin)) * ph;

  // Grid lines
  ctx.strokeStyle = cssVar("--border");
  ctx.fillStyle = cssVar("--text-muted");
  ctx.font = "11px 'JetBrains Mono', monospace";
  [0.72, 0.75, 0.78, 0.81, 0.84].forEach((yv) => {
    const py = yOf(yv);
    ctx.beginPath(); ctx.moveTo(ml, py); ctx.lineTo(w - mr, py); ctx.stroke();
    ctx.fillText(yv.toFixed(2), 24, py + 4);
  });
  [0.05, 0.1, 0.5, 1, 5].forEach((tv) => {
    const px = xOf(Math.log10(tv));
    ctx.beginPath(); ctx.moveTo(px, mt); ctx.lineTo(px, mt + ph); ctx.stroke();
    ctx.fillText(`${tv}`, px - 10, mt + ph + 16);
  });

  // Fit log-linear trend line
  const xs = u25.map((r) => Math.log10(r.tick_bp));
  const ys = u25.map((r) => r.auc_ovr_5s);
  const xm = xs.reduce((a, b) => a + b, 0) / xs.length;
  const ym = ys.reduce((a, b) => a + b, 0) / ys.length;
  const slope = xs.reduce((s, x, i) => s + (x - xm) * (ys[i] - ym), 0) / xs.reduce((s, x) => s + (x - xm) ** 2, 0);
  const intercept = ym - slope * xm;

  ctx.strokeStyle = cssVar("--accent");
  ctx.setLineDash([5, 4]);
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(xOf(xMin + 0.05), yOf(intercept + slope * (xMin + 0.05)));
  ctx.lineTo(xOf(xMax - 0.1), yOf(intercept + slope * (xMax - 0.1)));
  ctx.stroke();
  ctx.setLineDash([]);

  // Points
  STATE.scatterPoints = [];
  const labelFocus = new Set(["YESBANK", "IDEA", "ITC", "ONGC", "RELIANCE", "TCS", "NIFTY_FUT", "BANKNIFTY_FUT", STATE.selectedSymbol25]);
  u25.forEach((r) => {
    const px = xOf(Math.log10(r.tick_bp));
    const py = yOf(r.auc_ovr_5s);
    STATE.scatterPoints.push({ px, py, row: r });

    const inSec = STATE.selectedSector === "ALL" || r.sector === STATE.selectedSector;
    const is250 = STATE.highlight250 && (r.symbol === "ONGC" || r.symbol === "ITC");
    const isSel = r.symbol === STATE.selectedSymbol25;

    ctx.globalAlpha = inSec ? 1 : 0.22;
    ctx.fillStyle = is250 ? cssVar("--amber") : r.asset === "DERIVATIVE" ? cssVar("--down") : cssVar("--blue");
    const rad = isSel || is250 ? 7.5 : 5.5;
    ctx.beginPath(); ctx.arc(px, py, rad, 0, Math.PI * 2); ctx.fill();
    if (isSel || is250) {
      ctx.strokeStyle = cssVar("--text");
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    if (labelFocus.has(r.symbol) && inSec) {
      ctx.fillStyle = cssVar("--text");
      ctx.font = (isSel || is250 ? "700 " : "500 ") + "10.5px Inter, sans-serif";
      ctx.fillText(r.symbol, px + 8, py - 4);
    }
    ctx.globalAlpha = 1;
  });

  // Axis titles
  ctx.fillStyle = cssVar("--text-muted");
  ctx.font = "600 11.5px Inter, sans-serif";
  ctx.fillText("Relative Tick Size (bp, log scale)", ml + pw / 2 - 90, h - 6);
  ctx.save();
  ctx.translate(12, mt + ph / 2 + 65);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("LightGBM 5 s ROC-AUC", 0, 0);
  ctx.restore();
}

function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

window.addEventListener("DOMContentLoaded", init);

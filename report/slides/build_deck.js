// Builds ML-focused 22-slide presentation: report/LOB_ML_Project_Presentation.pptx
// Designed for UML501 Machine Learning evaluation: emphasizes clear ML concepts,
// visual flowcharts, model architectures, preprocessing, feature engineering,
// easy-to-understand metrics, and core ML takeaways without heavy finance jargon.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const IMG = path.join(__dirname, "img");
const OUT = path.join(__dirname, "..", "LOB_ML_Project_Presentation.pptx");

const THEME = {
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2633", lt1: "FFFFFF", dk2: "123047", lt2: "EEF3F6",
    accent1: "0E7C86", // teal
    accent2: "C8453C", // coral
    accent3: "D99A2B", // amber
    accent4: "5A6B7B", // slate
    accent5: "2F6FA8", // blue
    accent6: "2A7B59", // green
  },
};
const C = THEME.colors;
const W = 13.333, H = 7.5, X0 = 0.55, CW = W - 2 * X0;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Predicting Limit Order Books with Machine Learning";
pres.author = "Aarav Singhal, Sharan Sharma";
pres.subject = "UML501 Machine Learning Project Presentation";

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.lt1 },
  objects: [
    { rect: { x: 0, y: 0, w: W, h: 0.08, fill: { color: C.accent1 } } },
    { text: { text: "UML501 · Machine Learning  |  Aarav Singhal & Sharan Sharma  |  Guided by Dr. Sanju Kumari", options: { x: X0, y: H - 0.36, w: 9.8, h: 0.26, fontSize: 9.5, color: "7F8C98", margin: 0 } } },
  ],
  slideNumber: { x: W - 1.05, y: H - 0.36, w: 0.5, h: 0.26, fontSize: 9.5, color: "7F8C98", align: "right" },
});

pres.defineSlideMaster({
  title: "DARK",
  background: { color: C.dk2 },
  objects: [
    { rect: { x: 0, y: 0, w: W, h: 0.10, fill: { color: C.accent3 } } },
  ],
});

let ICONS = {};
async function iconPng(Comp, color) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

function T(s, text, o) {
  s.addText(text, Object.assign({ isTextBox: true, fontSize: 13, color: C.dk1, fontFace: THEME.bodyFontFace, valign: "top", margin: 0 }, o));
}

function header(s, sectionTag, title, notes) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: X0, y: 0.20, w: 3.1, h: 0.27, rectRadius: 0.05, fill: { color: "DCEEEF" }, line: { type: "none" } });
  T(s, sectionTag.toUpperCase(), { x: X0 + 0.1, y: 0.21, w: 2.9, h: 0.25, fontSize: 9.3, bold: true, color: C.accent1, valign: "middle" });
  T(s, title, { x: X0, y: 0.50, w: CW, h: 0.52, fontSize: 22.5, bold: true, color: C.dk2, fontFace: THEME.headFontFace, valign: "middle" });
  if (notes) s.addNotes(notes);
}

function box(s, x, y, w, h, fill, borderCol) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.07,
    fill: { color: fill || C.lt2 },
    line: borderCol ? { color: borderCol, pt: 1 } : { type: "none" },
  });
}

function iconBadge(s, key, x, y, d, fill) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill || C.accent1 }, line: { type: "none" } });
  const p = d * 0.22;
  s.addImage({ data: ICONS[key], x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p });
}

function card(s, x, y, w, h, head, body, o = {}) {
  box(s, x, y, w, h, o.fill || C.lt2, o.border);
  let tx = x + 0.16, tw = w - 0.32;
  if (o.icon) {
    iconBadge(s, o.icon, x + 0.14, y + 0.13, 0.40, o.iconFill || C.accent1);
    tx = x + 0.64; tw = w - 0.80;
  }
  T(s, head, { x: tx, y: y + 0.12, w: tw, h: 0.40, fontSize: o.headSize || 14, bold: true, color: o.headColor || C.dk2, valign: "middle" });
  const by = y + 0.56;
  if (body) T(s, body, { x: x + 0.16, y: by, w: w - 0.32, h: h - 0.64, fontSize: o.size || 12, color: C.dk1 });
}

function bullets(s, items, o) {
  const runs = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    const b = { bullet: { indent: 14 }, paraSpaceAfter: o.gap ?? 5 };
    if (Array.isArray(it)) {
      runs.push({ text: it[0] + " ", options: Object.assign({}, b, { bold: true, color: C.dk2 }) });
      runs.push({ text: it[1], options: { breakLine: !last, color: C.dk1 } });
    } else {
      runs.push({ text: it, options: Object.assign({}, b, { breakLine: !last, color: C.dk1 }) });
    }
  });
  const opts = Object.assign({}, o);
  delete opts.gap;
  T(s, runs, opts);
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}

function fig(s, name, x, y, w, h) {
  const f = path.join(IMG, name + ".png");
  const [pw, ph] = pngSize(f);
  let iw = w, ih = (w * ph) / pw;
  if (ih > h) { ih = h; iw = (h * pw) / ph; }
  s.addImage({ path: f, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih });
}

function arrowR(s, x, y, w, h = 0.28, col = "94A7B6") {
  s.addShape(pres.shapes.RIGHT_ARROW, { x, y, w, h, fill: { color: col }, line: { type: "none" } });
}

function arrowD(s, x, y, w = 0.32, h = 0.34, col = "94A7B6") {
  s.addShape(pres.shapes.DOWN_ARROW, { x, y, w, h, fill: { color: col }, line: { type: "none" } });
}

function flowNode(s, x, y, w, h, title, sub, fill) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { type: "none" } });
  T(s, title, { x: x + 0.08, y: y + 0.08, w: w - 0.16, h: 0.36, fontSize: 12.8, bold: true, color: C.lt1, align: "center", valign: "middle" });
  if (sub) T(s, sub, { x: x + 0.08, y: y + 0.44, w: w - 0.16, h: h - 0.50, fontSize: 10.6, color: "EAF2F6", align: "center" });
}

const BORDER = { type: "solid", pt: 0.6, color: "CBD5DF" };
function table(s, rows, o) {
  const fs_ = o.fontSize || 11;
  const data = rows.map((r, i) => r.map((c) => {
    const cell = typeof c === "object" && c !== null && c.text !== undefined ? c : { text: String(c) };
    const base = { fontSize: fs_, color: C.dk1, border: BORDER, valign: "middle", margin: [3, 5, 3, 5], fontFace: THEME.bodyFontFace };
    if (i === 0) Object.assign(base, { bold: true, color: C.lt1, fill: { color: C.dk2 } });
    else if (o.hl && o.hl.includes(i)) Object.assign(base, { bold: true, fill: { color: "D9EFEF" } });
    else if (i % 2 === 0) base.fill = { color: "F5F8FA" };
    return { text: cell.text, options: Object.assign(base, cell.options || {}) };
  }));
  s.addTable(data, { x: o.x, y: o.y, w: o.w, colW: o.colW, rowH: o.rowH, autoPage: false });
}

function callout(s, x, y, w, h, lead, body, fill, accent) {
  box(s, x, y, w, h, fill || "E4F2F2");
  s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.08, h, fill: { color: accent || C.accent1 }, line: { type: "none" } });
  T(s, [
    { text: lead + "  ", options: { bold: true, color: accent || C.accent1 } },
    { text: body, options: { color: C.dk1 } },
  ], { x: x + 0.2, y: y + 0.08, w: w - 0.32, h: h - 0.16, fontSize: 12, valign: "middle" });
}

async function main() {
  const iconMap = {
    bulb: fa.FaLightbulb, db: fa.FaDatabase, chart: fa.FaChartLine, brain: fa.FaBrain,
    clock: fa.FaClock, scale: fa.FaBalanceScale, shield: fa.FaShieldAlt, tree: fa.FaTree,
    check: fa.FaCheckCircle, warn: fa.FaExclamationTriangle, bolt: fa.FaBolt, code: fa.FaCode,
    target: fa.FaBullseye, layers: fa.FaLayerGroup, filter: fa.FaFilter, cogs: fa.FaCogs,
    search: fa.FaSearch, list: fa.FaListOl,
  };
  for (const [k, v] of Object.entries(iconMap)) ICONS[k] = await iconPng(v, "FFFFFF");

  // =========================================================================
  // SLIDE 1: TITLE SLIDE
  // =========================================================================
  let s = pres.addSlide({ masterName: "DARK" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: X0, y: 0.80, w: 3.9, h: 0.38, rectRadius: 0.06, fill: { color: "1E4663" }, line: { type: "none" } });
  T(s, "UML501 – MACHINE LEARNING  ·  PROJECT PRESENTATION", { x: X0 + 0.14, y: 0.85, w: 3.65, h: 0.28, fontSize: 10, bold: true, color: C.accent3, valign: "middle" });
  T(s, "Predicting Limit Order Books\nwith Machine Learning", { x: X0, y: 1.40, w: CW, h: 1.75, fontSize: 42, bold: true, color: C.lt1, fontFace: THEME.headFontFace });
  T(s, "End-to-End Preprocessing, Feature Engineering, Classical vs Deep Learning Models, and Inference Speed", { x: X0, y: 3.25, w: 11.6, h: 0.55, fontSize: 17.5, italic: true, color: "CFE3E8" });
  s.addShape(pres.shapes.RECTANGLE, { x: X0, y: 4.05, w: CW, h: 0.02, fill: { color: "385A73" }, line: { type: "none" } });

  T(s, "Submitted By", { x: X0, y: 4.35, w: 5.5, h: 0.3, fontSize: 11.5, bold: true, color: C.accent3 });
  T(s, "Aarav Singhal  (1024030539)\nSharan Sharma  (1024030511)", { x: X0, y: 4.68, w: 5.5, h: 0.75, fontSize: 16, bold: true, color: C.lt1 });

  T(s, "Under the Guidance of", { x: 6.6, y: 4.35, w: 5.8, h: 0.3, fontSize: 11.5, bold: true, color: C.accent3 });
  T(s, "Dr. Sanju Kumari\nAssistant Professor, CSED", { x: 6.6, y: 4.68, w: 5.8, h: 0.75, fontSize: 16, bold: true, color: C.lt1 });

  T(s, "Computer Science & Engineering Department  ·  Thapar Institute of Engineering & Technology, Patiala  ·  November 2026",
    { x: X0, y: 6.55, w: CW, h: 0.35, fontSize: 12, color: "9BB4C7" });
  s.addNotes("Good morning/afternoon Dr. Sanju Kumari. Our project applies Machine Learning to high-frequency Limit Order Book data. Today we will walk through what the dataset represents, how we cleaned and preprocessed 38+ million rows without data leakage, how we engineered features, the 6-model ladder from Logistic Regression and Gradient Boosted Trees to CNN-LSTM and Transformers, and our main finding: why simpler feature-engineered models outperform deep neural networks at short horizons.");

  // =========================================================================
  // SLIDE 2: PRESENTATION FLOW & PROBLEM OVERVIEW
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 1 · Introduction", "Project Goal & Presentation Flow",
    "Explain the ML problem simply: instead of predicting daily prices from 4 numbers per day, we predict 3-class direction (UP, FLAT, DOWN) over the next 5 to 20 seconds from rich second-by-second supply-and-demand tables, comparing classical ML against deep learning.");

  card(s, X0, 1.15, 5.95, 2.55, "The Machine Learning Problem",
    "• Input X: Second-by-second table of waiting buy and sell orders across top 5–10 price levels (20 to 40 numerical columns per second).\n• Target y (3-Class Classification): Will the midpoint price in k seconds (e.g. 20 s) go UP, stay FLAT, or go DOWN?\n• Core ML Question: Do deep neural networks (CNN-LSTM, Transformer) reading raw 2D time windows beat classical ML (Logistic Regression, Random Forest, LightGBM) on engineered features?",
    { icon: "target", size: 11.8 });

  card(s, 6.75, 1.15, 6.03, 2.55, "Why This Problem Is Interesting for ML",
    "• Massive real-world tabular + sequence data (38+ million rows across Indian NSE stocks, Bitcoin, US Nasdaq, and FI-2010).\n• Severe traps for careless ML: class imbalance (up to 70–99% FLAT), strong feature correlation, time-series leakage, and distribution shift.\n• Lets us test accuracy, explainability (SHAP & CNN saliency), and real-time inference latency (microseconds vs milliseconds).",
    { icon: "bulb", iconFill: C.accent3, size: 11.8 });

  T(s, "Presentation Flow (6 Steps)", { x: X0, y: 3.92, w: CW, h: 0.34, fontSize: 14.5, bold: true, color: C.dk2 });
  const flow6 = [
    ["1. Domain & Data", "What an order book is & our datasets", C.accent1],
    ["2. Preprocessing", "Cleaning, labelling & anti-leakage split", C.accent5],
    ["3. Features", "25–27 engineered features & PCA", C.accent1],
    ["4. ML Models", "Tree ensembles vs CNN-LSTM & Transformer", C.accent2],
    ["5. ML Results", "Metrics, confusion matrix, FI-2010, SHAP", C.accent6],
    ["6. Speed & Impact", "C++ 10.5 µs engine & practical limits", C.accent3],
  ];
  const fw = 1.88, fg = 0.18;
  flow6.forEach((f, i) => {
    const x = X0 + i * (fw + fg);
    flowNode(s, x, 4.35, fw, 1.35, f[0], f[1], f[2]);
    if (i < 5) arrowR(s, x + fw + 0.02, 4.90, fg - 0.04, 0.24);
  });

  callout(s, X0, 5.92, CW, 0.90, "Preview of main ML finding:",
    "At short horizons (5–20 s), LightGBM on 25 engineered features beats heavy GPU deep networks (LSTM, DeepLOB) in both Macro-F1 and Accuracy while running 230–540× faster; deep networks win only at longer horizons (50–100 steps).",
    "FDF5E6", C.accent3);

  // =========================================================================
  // SLIDE 3: WHAT IS A LIMIT ORDER BOOK? (DRAWN DIAGRAM)
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 1 · Introduction", "What Is a Limit Order Book? (Explained Visually)",
    "Walk through this diagram slowly, no finance background needed! Think of an online exchange as two waiting lines: sellers asking for ₹100.01, ₹100.02... at the top, and buyers offering ₹99.99, ₹99.98... at the bottom. Halfway between the lowest seller and highest buyer is the Mid-Price. When the buyer bar is much longer than the seller bar, upward pressure builds!");

  // Left: Drawn 5-level Order Book ladder
  box(s, X0, 1.18, 6.1, 5.65, "F7FAFC", "CBD5DF");
  T(s, "Visualizing One Snapshot (1 Row of Data)", { x: X0 + 0.2, y: 1.30, w: 5.7, h: 0.35, fontSize: 14, bold: true, color: C.dk2 });
  const ladder = [
    ["₹100.05", 280, "a", "Ask Level 5"],
    ["₹100.04", 460, "a", "Ask Level 4"],
    ["₹100.03", 240, "a", "Ask Level 3"],
    ["₹100.02", 380, "a", "Ask Level 2"],
    ["₹100.01", 150, "a", "Best Ask (Level 1)"],
    ["₹99.99", 540, "b", "Best Bid (Level 1)"],
    ["₹99.98", 390, "b", "Bid Level 2"],
    ["₹99.97", 610, "b", "Bid Level 3"],
    ["₹99.96", 290, "b", "Bid Level 4"],
    ["₹99.95", 410, "b", "Bid Level 5"],
  ];
  T(s, "SELLERS\n(Ask Queue)", { x: X0 + 0.18, y: 2.15, w: 1.25, h: 0.8, fontSize: 12.5, bold: true, color: C.accent2 });
  T(s, "BUYERS\n(Bid Queue)", { x: X0 + 0.18, y: 4.95, w: 1.25, h: 0.8, fontSize: 12.5, bold: true, color: C.accent1 });

  ladder.forEach((l, i) => {
    const y = 1.78 + i * 0.43 + (l[2] === "b" ? 0.46 : 0);
    const col = l[2] === "a" ? C.accent2 : C.accent1;
    const isBest = i === 4 || i === 5;
    T(s, l[0], { x: X0 + 1.45, y, w: 1.0, h: 0.36, fontSize: 12.2, bold: isBest, align: "right", valign: "middle" });
    const bw = (l[1] / 610) * 2.45;
    s.addShape(pres.shapes.RECTANGLE, { x: X0 + 2.60, y: y + 0.05, w: bw, h: 0.27, fill: { color: col }, line: { type: "none" } });
    T(s, `${l[1]} qty`, { x: X0 + 2.66 + bw, y, w: 0.95, h: 0.36, fontSize: 10.8, color: C.accent4, valign: "middle" });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: X0 + 1.45, y: 3.96, w: 4.35, h: 0.38, rectRadius: 0.05, fill: { color: "FFF4DC" }, line: { color: C.accent3, pt: 0.8 } });
  T(s, "Mid-Price = ₹100.00   |   Gap (Spread) = ₹0.02", { x: X0 + 1.55, y: 3.98, w: 4.15, h: 0.34, fontSize: 11.8, bold: true, color: "9A6500", align: "center", valign: "middle" });

  // Right: Plain-English ML mapping
  card(s, 6.90, 1.18, 5.88, 1.72, "Why Not Ordinary Daily Price Charts?",
    "Standard stock datasets only record 4 numbers per day (Open, High, Low, Close) after trades finish. An Order Book shows the live queue of waiting buyers vs sellers every second, before the price moves.",
    { icon: "search", size: 11.8 });
  card(s, 6.90, 3.04, 5.88, 1.82, "How One Second Becomes a Feature Vector",
    "• 5 levels × 4 numbers (Ask Price, Ask Qty, Bid Price, Bid Qty) = 20 numbers/sec on Indian NSE (40 numbers/sec for 10-level Bitcoin/US books).\n• Notice Level 1 on the left: 540 buyers vs only 150 sellers! Sellers at ₹100.01 are likely to run out first, pushing Mid-Price UP.",
    { icon: "layers", iconFill: C.accent5, size: 11.6 });
  card(s, 6.90, 5.00, 5.88, 1.83, "Target Variable y (Supervised Classification)",
    "Compare future smoothed Mid-Price at t + k seconds against Mid-Price now:\n• Class 1 (UP): rises by more than +θ\n• Class 2 (DOWN): falls by more than −θ\n• Class 0 (FLAT): stays within [−θ, +θ] (θ fit on training set only).",
    { icon: "list", iconFill: C.accent6, size: 11.6 });

  // =========================================================================
  // SLIDE 4: VISUAL PROOF IN REAL DATA (IMBALANCE -> NEXT MOVE)
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 1 · Exploratory Data Analysis", "Exploratory Analysis: Does Queue Imbalance Actually Predict Price Direction?",
    "Before training any complex ML model, we did Exploratory Data Analysis (EDA). Both plots show that when the buyer queue is heavier than the seller queue, the average price move 20 seconds later is positive, and on stocks with larger price steps (right), the correlation is up to 12 times stronger!");

  T(s, "Bitcoin (1.03M rows): Deciles of Queue Imbalance vs Future Move", { x: X0, y: 1.12, w: 5.9, h: 0.3, fontSize: 12.2, bold: true, color: C.dk2 });
  fig(s, "btc_imbalance_signal", X0, 1.44, 5.95, 3.85);

  T(s, "Indian NSE (64 Days): Correlation Across 8 Representative Instruments", { x: 6.72, y: 1.12, w: 6.05, h: 0.3, fontSize: 12.2, bold: true, color: C.dk2 });
  fig(s, "nse_eda_tick_spectrum", 6.72, 1.44, 6.06, 3.85);

  callout(s, X0, 5.48, CW, 1.32, "What Exploratory Data Analysis (EDA) tells us:",
    "1. Left: Sorting 1.03 million Bitcoin snapshots into 10 bins from seller-heavy to buyer-heavy creates a clean monotonic staircase: the signal is real, though noisy.\n2. Right: Across 64 days on India's NSE, stocks whose minimum price step is larger relative to share price (e.g. IDEA, YESBANK, ITC) build much deeper waiting queues and show up to 12× higher correlation with the next 20-second price move.",
    "E4F2F2", C.accent1);

  // =========================================================================
  // SLIDE 5: DATASETS & WHERE THEY COME FROM
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 2 · Data & Preprocessing", "Where Our Data Comes From: 38+ Million Rows Across 4 Real Sources + Synthetic",
    "Explain clearly where every dataset came from and why we used multiple sources: our biggest, primary dataset is 64 full trading days of live Indian NSE order books (25 stocks & index futures = 36M rows), complemented by public Coinbase crypto, Nasdaq academic samples, and the published FI-2010 ML benchmark.");

  table(s, [
    ["Dataset Source", "Origin & How Collected", "Size & Dimensions", "Why Included in Our ML Study"],
    ["1. Indian NSE Archive (Primary Original Data)", "Recorded from live Zerodha Kite NSE feed (Aug–Nov 2026); 23 stocks across 10 sectors + NIFTY & BANKNIFTY", "64 trading days · 25 symbols · 36.0M 1-sec rows × 20 cols", "Largest multi-month dataset; tests all models & feature scaling on Indian stocks"],
    ["2. Coinbase Crypto (BTC-USD & ETH-USD)", "Public CC0 dataset via Kaggle API (April 2021 continuous 24/7 order books)", "12 days × 2 coins · 2.06M 1-sec rows × 40 cols", "High-noise 10-level benchmark & cross-coin feature test"],
    ["3. FI-2010 Academic Benchmark", "Public university benchmark (Ntakaris et al., 5 Nordic stocks, pre-normalised)", "10 days · 144 features · Setup 2 (Days 1–7 train, 8–10 test)", "Direct head-to-head comparison with published DeepLOB paper"],
    ["4. US Nasdaq (LOBSTER) + Optiver", "Academic sample files from lobsterdata.com (INTC, MSFT, AAPL, AMZN, GOOG) + Kaggle", "1.17M 100-ms rows + 5.2M regression rows", "Sub-second horizon decay & time-series data-leakage case study"],
    ["5. Synthetic Order Book", "Generated by our discrete-event Python simulator with a planted pattern", "5 sessions × 60,000 events × 40 cols", "Controlled ground-truth sanity check for pipelines & unit tests"],
  ], { x: X0, y: 1.15, w: CW, colW: [2.35, 3.85, 2.85, 3.18], rowH: 0.64, fontSize: 10.5, hl: [1] });

  callout(s, X0, 5.82, CW, 0.96, "Hardware used for training:",
    "Classical ML (Logistic Regression, Random Forest, LightGBM) and preprocessing ran on an Intel Core i9-13900HX CPU (32 threads, 32 GB RAM). Deep neural networks (LSTM, DeepLOB, TLOB) were trained with PyTorch + CUDA on NVIDIA A100 (Colab) and local RTX 4080 (12 GB VRAM) GPUs.",
    "EEF3F6", C.dk2);

  // =========================================================================
  // SLIDE 6: PREPROCESSING FLOWCHART (5-STEP VISUAL PIPELINE)
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 2 · Data & Preprocessing", "Data Preprocessing Pipeline: From Raw Feeds to Clean ML Tensors",
    "Dr. Kumari asked specifically what preprocessing we did. Point to each of the 5 boxes left-to-right: schema unification, data cleaning/filtering, 1-second time-grid alignment with daily resets, train-only target labelling, and model-specific feature scaling.");

  const pSteps = [
    ["Step 1: Parse", "Unify Parquet / CSV feeds into canonical [AskP, AskQ, BidP, BidQ] columns", C.accent4],
    ["Step 2: Clean", "Drop crossed quotes (Ask ≤ Bid), auction noise, NaNs/Infs & duplicate timestamps", C.accent5],
    ["Step 3: Resample", "Align irregular ticks onto a strict 1-sec grid; reset at daily 09:15–15:30 boundaries", C.accent1],
    ["Step 4: Label y", "Compute k-step forward mid return; fit ±θ cutoff on Train only → UP / FLAT / DOWN", C.accent6],
    ["Step 5: Scale X", "Z-score for linear models; unscaled for trees; train-only log/relative scale for DL", C.accent2],
  ];
  const pw = 2.22, pg = 0.28;
  pSteps.forEach((st, i) => {
    const x = X0 + i * (pw + pg);
    flowNode(s, x, 1.25, pw, 1.55, st[0], st[1], st[2]);
    if (i < 4) arrowR(s, x + pw + 0.02, 1.88, pg - 0.04, 0.28);
  });

  card(s, X0, 3.08, 4.0, 3.70, "1. Cleaning & Anomaly Removal",
    "• Raw NSE archive contained 84 dates; we audited depth completeness and dropped early partial dates + the 1-hour Diwali special session, keeping 64 complete 6.25-hour sessions.\n• Removed pre-open/closing auction crossed quotes where Best Ask ≤ Best Bid.\n• Replaced empty outer book levels and division-by-zero edge cases safely.",
    { icon: "filter", size: 11.3 });

  card(s, X0 + 4.12, 3.08, 4.0, 3.70, "2. Overnight Isolation (Multi-Day Data)",
    "• Stock markets close at 3:30 PM and reopen next morning at 9:15 AM at a new price.\n• Every rolling window, lag (t−1, t−2, t−5), 100-second deep-learning sequence, and forward label resets strictly inside each day.\n• Prevents artificial overnight price jumps from corrupting gradients or labels.",
    { icon: "clock", iconFill: C.accent5, size: 11.3 });

  card(s, X0 + 8.24, 3.08, 3.99, 3.70, "3. Tailored Input Normalisation",
    "• Logistic Regression: StandardScaler (mean 0, std 1) fitted on Train rows only.\n• Decision Trees (RF, LightGBM): No scaling needed (trees split on rank order).\n• Deep Nets (LSTM, DeepLOB): Prices scaled relative to training P₀; volumes log-transformed log(1 + q/V₀); TLOB adds learned per-window Bilinear Norm (BiN).",
    { icon: "cogs", iconFill: C.accent2, size: 11.3 });

  // =========================================================================
  // SLIDE 7: PREVENTING DATA LEAKAGE (DRAWN SPLIT + LEAKAGE PLOT)
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 2 · Data & Preprocessing", "Preventing Time-Series Data Leakage: Purged Splits & Shuffled Controls",
    "Explain why random train_test_split is wrong for time series! If row t is in Train and row t+1 is in Test, they share 99% of the same book and overlapping 20-second labels. Show our drawn chronological split with purged gaps on top, and our real experiment on bottom-left proving how random K-fold creates fake accuracy!");

  // Drawn Chronological Split Bar
  const sy = 1.28, sh = 0.62, sx0 = X0 + 0.05, sW = CW - 0.1;
  s.addShape(pres.shapes.RECTANGLE, { x: sx0, y: sy, w: 0.66 * sW, h: sh, fill: { color: C.accent1 }, line: { type: "none" } });
  s.addShape(pres.shapes.RECTANGLE, { x: sx0 + 0.66 * sW, y: sy, w: 0.04 * sW, h: sh, fill: { color: "263238" }, line: { type: "none" } });
  s.addShape(pres.shapes.RECTANGLE, { x: sx0 + 0.70 * sW, y: sy, w: 0.12 * sW, h: sh, fill: { color: C.accent3 }, line: { type: "none" } });
  s.addShape(pres.shapes.RECTANGLE, { x: sx0 + 0.82 * sW, y: sy, w: 0.03 * sW, h: sh, fill: { color: "263238" }, line: { type: "none" } });
  s.addShape(pres.shapes.RECTANGLE, { x: sx0 + 0.85 * sW, y: sy, w: 0.15 * sW, h: sh, fill: { color: C.accent2 }, line: { type: "none" } });

  T(s, "Train Split (~70% / 11 Days)", { x: sx0, y: sy, w: 0.66 * sW, h: sh, fontSize: 13.5, bold: true, color: C.lt1, align: "center", valign: "middle" });
  T(s, "Purge", { x: sx0 + 0.66 * sW, y: sy, w: 0.04 * sW, h: sh, fontSize: 9.5, bold: true, color: C.lt1, align: "center", valign: "middle" });
  T(s, "Val (~15%)", { x: sx0 + 0.70 * sW, y: sy, w: 0.12 * sW, h: sh, fontSize: 12.5, bold: true, color: C.lt1, align: "center", valign: "middle" });
  T(s, "Purge", { x: sx0 + 0.82 * sW, y: sy, w: 0.03 * sW, h: sh, fontSize: 9.0, bold: true, color: C.lt1, align: "center", valign: "middle" });
  T(s, "Test (~15% / Future)", { x: sx0 + 0.85 * sW, y: sy, w: 0.15 * sW, h: sh, fontSize: 12.5, bold: true, color: C.lt1, align: "center", valign: "middle" });
  T(s, "Strict Chronological Time Order  (Repeat across 4 independent segments per dataset)  →",
    { x: sx0, y: sy + sh + 0.06, w: sW, h: 0.26, fontSize: 11, italic: true, color: C.accent4, align: "center" });

  fig(s, "optiver_leakage", X0, 2.32, 6.25, 4.45);

  card(s, 7.02, 2.32, 5.76, 2.15, "What the Plot on the Left Demonstrates",
    "• Left panel: Using random shuffled K-fold + look-ahead neighbour features artificially cuts error from 0.246 down to 0.219, a fake gain caused by peeking at neighbouring timestamps!\n• Once evaluated properly on future folds, the fake advantage vanishes.",
    { icon: "warn", iconFill: C.accent2, fill: "FDF0EE", size: 11.4 });

  card(s, 7.02, 4.62, 5.76, 2.15, "Our Verification Safeguards (43 Pytest Tests)",
    "• Purging: Drop the last k rows before Val/Test boundaries because their k-second labels reach into the next split.\n• Future-mutation test: Mutating rows t+1..N in code is proven to change 0.000000 of features at row t.\n• Shuffled-label control: Training on shuffled y scores 0.00 correlation across 69 test cases.",
    { icon: "shield", iconFill: C.accent6, size: 11.3 });

  // =========================================================================
  // SLIDE 8: FEATURE ENGINEERING (PLAIN-WORDS FAMILIES)
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 3 · Feature Engineering", "Feature Engineering: Turning Raw Order Books Into 25–27 Informative Signals",
    "Instead of feeding raw ₹23,000 price numbers into classical ML models, we engineered 6 intuitive families of scale-free features using domain knowledge and PCA dimensionality reduction.");

  flowNode(s, X0, 1.18, 3.3, 0.78, "Raw Snapshot (20–40 cols)", "Prices & sizes at 5–10 levels", C.accent4);
  arrowR(s, X0 + 3.4, 1.43, 0.45, 0.28);
  flowNode(s, X0 + 3.95, 1.18, 4.3, 0.78, "Causal Feature Engineering + Train PCA", "Ratios, differences, rolling buckets & lags", C.accent1);
  arrowR(s, X0 + 8.35, 1.43, 0.45, 0.28);
  flowNode(s, X0 + 8.90, 1.18, 3.33, 0.78, "Clean Feature Vector (25–27)", "Scale-invariant tabular input X_t", C.accent6);

  const fCards = [
    ["scale", "1. Queue Imbalance (4)", "Compares waiting buyer size vs seller size at top 1, 3, 5, and 10 levels:\n(BidQty − AskQty) / (BidQty + AskQty).\nBounded cleanly in [−1, +1].", C.accent1],
    ["target", "2. Weighted Mid-Price (1)", "Shifts the midpoint toward whichever side has a thinner queue, capturing sub-step upward or downward pressure.", C.accent5],
    ["bolt", "3. Order Flow Imbalance (5)", "Measures net new buying minus selling pressure arriving between t−1 and t (orders added minus cancelled/executed).", C.accent1],
    ["layers", "4. Multi-Level PCA (3)", "Applies Principal Component Analysis (fitted on Train only) to compress multi-level order flows into 3 orthogonal PCs (80.5% variance).", C.accent6],
    ["chart", "5. Spread, Slopes & Flow (3)", "Gap between Best Ask & Best Bid (in integer ticks), price-depth slope on each side, and recent volume-bucket toxicity.", C.accent3],
    ["clock", "6. Temporal Lags (9)", "Values of Level-1 Imbalance, Order Flow, and Weighted Mid 1 s, 2 s, and 5 s ago, giving tree models short-term memory.", C.accent2],
  ];
  fCards.forEach((fc, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    card(s, X0 + col * 4.14, 2.16 + row * 2.38, 3.95, 2.22, fc[1], fc[2], { icon: fc[0], iconFill: fc[3], size: 11.2 });
  });

  // =========================================================================
  // SLIDE 9: FORMULAS & FEATURE CORRELATION HEATMAP
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 3 · Feature Engineering", "Key Feature Formulas & Multicollinearity Analysis (Correlation Heatmap)",
    "Walk through the 4 simple equations on the right, then point to the 27x27 correlation heatmap on the left: within-group imbalance features are correlated (red squares), while across-group features have median correlation of just 0.06, meaning each family adds new information!");

  fig(s, "feature_corr", X0, 1.15, 5.75, 5.65);

  const eqs = [
    ["Queue Imbalance", "I_L = (Σ BidQty − Σ AskQty) / (Σ BidQty + Σ AskQty)", "Computed for L = 1, 3, 5 (and 10). Scale-free across ₹13 to ₹60,000 assets."],
    ["Weighted Mid Dev.", "ΔP_micro = I_1 × (Spread / 2)", "Combines queue ratio with current bid-ask gap."],
    ["Order Flow (OFI)", "OFI_t = ΔDemand_bid(t) − ΔSupply_ask(t)", "Tracks dynamic changes rather than static resting queues."],
    ["PCA Compression", "Z_1..3 = W_PCA^T · [OFI_lvl1, ..., OFI_lvlK]", "W_PCA estimated on Train split only; removes multi-level redundancy."],
  ];
  eqs.forEach((e, i) => {
    const y = 1.20 + i * 1.05;
    box(s, 6.55, y, 6.23, 0.94, C.lt2);
    T(s, e[0], { x: 6.70, y: y + 0.08, w: 2.1, h: 0.34, fontSize: 12.5, bold: true, color: C.dk2 });
    T(s, e[1], { x: 8.75, y: y + 0.08, w: 3.9, h: 0.34, fontSize: 12, bold: true, fontFace: THEME.headFontFace, color: C.accent1 });
    T(s, e[2], { x: 6.70, y: y + 0.46, w: 5.95, h: 0.40, fontSize: 10.8, color: C.accent4 });
  });

  card(s, 6.55, 5.45, 6.23, 1.35, "What the Correlation Heatmap Shows",
    "• Red diagonal blocks: Imbalance at levels 1/3/5/10 correlate strongly (0.79–0.93), and OFI PC1 tracks L1 OFI (0.91).\n• Off-diagonal blocks: Median |r| across all feature pairs is only 0.06! L2 regularisation (Logistic) and tree splits handle correlated blocks cleanly.",
    { fill: "EAF4F4", size: 11.1 });

  // =========================================================================
  // SLIDE 10: END-TO-END SYSTEM ARCHITECTURE FLOWCHART
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 4 · ML Models & Architectures", "End-to-End Machine Learning System Flowchart (Two Parallel Branches)",
    "Dr. Kumari asked for a system flowchart. Walk from left to right: unified raw books go into cleaning, labelling, and chronological splitting, then fork into TWO fair branches evaluated on the exact same test timestamps: Branch A (engineered features -> classical ML) vs Branch B (raw 100-step 2D windows -> GPU Deep Learning).");

  // Left shared stages
  flowNode(s, X0, 2.55, 2.05, 1.45, "1. Raw Data", "NSE (25 stocks)\nBTC & ETH\nFI-2010 & Nasdaq", C.accent4);
  arrowR(s, X0 + 2.10, 3.13, 0.28, 0.28);
  flowNode(s, X0 + 2.43, 2.55, 2.15, 1.45, "2. Clean & Split", "1-sec alignment\nTrain-only ±θ labels\nPurged 70/15/15 split", C.dk2);

  // Fork arrows to Branch A (top) and Branch B (bottom)
  arrowR(s, X0 + 4.65, 2.72, 0.38, 0.24, C.accent1);
  arrowR(s, X0 + 4.65, 3.60, 0.38, 0.24, C.accent2);

  // Branch A: Tabular ML
  box(s, 5.65, 1.22, 4.55, 2.05, "E6F4F4", C.accent1);
  T(s, "BRANCH A · Classical / Tabular ML (CPU)", { x: 5.80, y: 1.30, w: 4.25, h: 0.30, fontSize: 12, bold: true, color: C.accent1 });
  flowNode(s, 5.80, 1.70, 1.95, 1.38, "Feature Eng.", "25–27 causal features\n(Imbalance, OFI,\nPCA, Lags)", C.accent1);
  arrowR(s, 7.80, 2.25, 0.26, 0.24, C.accent1);
  flowNode(s, 8.10, 1.70, 1.95, 1.38, "Tabular Models", "Logistic Regression\nRandom Forest\nLightGBM (+ C++)", C.accent6);

  // Branch B: Deep Learning
  box(s, 5.65, 3.55, 4.55, 2.05, "FDF0EE", C.accent2);
  T(s, "BRANCH B · Deep Sequence Learning (GPU)", { x: 5.80, y: 3.63, w: 4.25, h: 0.30, fontSize: 12, bold: true, color: C.accent2 });
  flowNode(s, 5.80, 4.03, 1.95, 1.38, "2D Windows", "Raw sliding tensor\n(100 steps × 20/40)\nTrain-only norm", C.accent2);
  arrowR(s, 7.80, 4.58, 0.26, 0.24, C.accent2);
  flowNode(s, 8.10, 4.03, 1.95, 1.38, "Deep Networks", "2-Layer LSTM\nDeepLOB / DeepLOB5\nTLOB Transformer", "9E2F28");

  // Converge to Unified Evaluation
  arrowR(s, 10.26, 2.25, 0.32, 0.24, C.dk2);
  arrowR(s, 10.26, 4.58, 0.32, 0.24, C.dk2);
  flowNode(s, 10.64, 1.55, 2.14, 3.75, "Unified Evaluation\n(Identical Rows)", "• Accuracy & Macro-F1\n• ROC-AUC & Log-Loss\n• Confusion Matrices\n• Block-Bootstrap CIs\n• SHAP & Saliency\n• Latency (µs vs ms)", C.accent5);

  callout(s, X0, 5.82, CW, 0.94, "Fairness guarantee:",
    "Every model in Branch A and Branch B is scored on the exact same test timestamps (each 100-second deep-learning window ends at test row t), using identical balanced class weights so comparisons are strictly apples-to-apples.",
    "EEF3F6", C.dk2);

  // =========================================================================
  // SLIDE 11: WHY WE CHOSE THESE MODELS ,  THE 6-TIER MODEL LADDER
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 4 · ML Models & Architectures", "Why We Chose These Models: A 6-Tier Complexity Ladder",
    "Explain why we didn't just pick one model: in rigorous Machine Learning, we build a complexity ladder from zero-learning baselines up to 200,000-parameter neural networks so each tier must justify its added complexity and compute cost.");

  const tiers = [
    ["Tier 0", "Naive Baselines", "Naive Flat (majority class) & Momentum (repeats last label).\n\nWhy: Sets the floor any real ML model must beat.", "0 params", C.accent4],
    ["Tier 1", "Logistic Regression", "Multinomial linear classifier with L2 penalty.\n\nWhy: Measures how much of the signal is linearly separable.", "84 params", C.accent5],
    ["Tier 2", "Random Forest", "100 parallel bagging trees (max depth 10).\n\nWhy: Tests non-linear splits & variance reduction.", "100 trees", C.accent1],
    ["Tier 3", "LightGBM (GBDT)", "Sequential gradient-boosted trees (63 leaves, early stop).\n\nWhy: State-of-the-art for structured tabular data.", "~300 trees", C.accent6],
    ["Tier 4", "LSTM Network", "2-layer gated recurrent network on 100-step raw sequences.\n\nWhy: Classic sequence baseline for temporal memory.", "58k–63k params", C.accent3],
    ["Tier 5", "DeepLOB & TLOB", "2D CNN + Inception + LSTM, and Self-Attention Transformer.\n\nWhy: Published SOTA deep architectures for order books.", "88k–214k params", C.accent2],
  ];
  const tw = 1.90, tg = 0.16;
  tiers.forEach((t, i) => {
    const x = X0 + i * (tw + tg), hgt = 2.65 + i * 0.42, y = 5.75 - hgt;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: tw, h: hgt, rectRadius: 0.08, fill: { color: t[4] }, line: { type: "none" } });
    T(s, t[0], { x: x + 0.1, y: y + 0.10, w: tw - 0.2, h: 0.26, fontSize: 10.5, bold: true, color: "E6EEF4" });
    T(s, t[1], { x: x + 0.1, y: y + 0.36, w: tw - 0.2, h: 0.48, fontSize: 13.5, bold: true, color: C.lt1 });
    T(s, t[2], { x: x + 0.1, y: y + 0.88, w: tw - 0.2, h: hgt - 1.32, fontSize: 10.6, color: C.lt1 });
    T(s, t[3], { x: x + 0.1, y: y + hgt - 0.36, w: tw - 0.2, h: 0.28, fontSize: 10.2, bold: true, italic: true, color: "FFF3D6" });
  });

  callout(s, X0, 5.92, CW, 0.86, "Handling Class Imbalance Across Every Tier:",
    "Every trained model (Tiers 1–5) uses inverse-frequency balanced class weights w_c = N / (3 · N_c) in its loss function, ensuring minority UP and DOWN movements receive equal priority to majority FLAT rows.",
    "E4F2F2", C.accent1);

  // =========================================================================
  // SLIDE 12: HOW TABULAR MODELS WORK (LINEAR VS BAGGING VS BOOSTING DIAGRAM)
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 4 · ML Models & Architectures", "How Our Classical ML Models Work: Linear vs Bagging vs Gradient Boosting",
    "Explain the difference between our three classical ML models using the visual diagrams inside each column: Logistic Regression draws straight hyperplanes; Random Forest averages 100 independent parallel trees (Bagging); LightGBM builds trees sequentially where each tree predicts the residual errors of the previous ensemble (Boosting).");

  // Col 1: Logistic Regression
  box(s, X0, 1.18, 3.92, 5.62, "F5F8FA", "CBD5DF");
  T(s, "1. Logistic Regression (Linear)", { x: X0 + 0.18, y: 1.30, w: 3.56, h: 0.36, fontSize: 14.5, bold: true, color: C.accent5 });
  flowNode(s, X0 + 0.35, 1.80, 3.22, 0.65, "Linear Score: z_c = w_c^T x + b_c", "Weighted sum of 25–27 scaled features", C.accent5);
  arrowD(s, X0 + 1.80, 2.50, 0.32, 0.28);
  flowNode(s, X0 + 0.35, 2.83, 3.22, 0.65, "Softmax: P(y=c) = e^{z_c} / Σ e^{z_j}", "Outputs calibrated 3-class probabilities", C.dk2);
  bullets(s, [
    ["Strength:", "Fast, convex, interpretable weights; L2 regularisation (C=1) prevents collinearity blow-up."],
    ["Limitation:", "Can only draw flat linear decision boundaries; misses non-linear threshold interactions (e.g., Imbalance only matters when Spread = 1 tick)."],
  ], { x: X0 + 0.20, y: 3.70, w: 3.52, h: 2.95, fontSize: 11.2, gap: 8 });

  // Col 2: Random Forest (Bagging)
  const cx2 = X0 + 4.15;
  box(s, cx2, 1.18, 3.92, 5.62, "F5F8FA", "CBD5DF");
  T(s, "2. Random Forest (Parallel Bagging)", { x: cx2 + 0.18, y: 1.30, w: 3.56, h: 0.36, fontSize: 14.5, bold: true, color: C.accent1 });
  flowNode(s, cx2 + 0.18, 1.80, 1.08, 0.65, "Tree 1", "Bootstrap", C.accent1);
  flowNode(s, cx2 + 1.42, 1.80, 1.08, 0.65, "Tree 2", "Bootstrap", C.accent1);
  flowNode(s, cx2 + 2.66, 1.80, 1.08, 0.65, "Tree 100", "Bootstrap", C.accent1);
  arrowD(s, cx2 + 1.80, 2.50, 0.32, 0.28);
  flowNode(s, cx2 + 0.35, 2.83, 3.22, 0.65, "Average Class Probabilities", "Variance drops as uncorrelated errors cancel", C.dk2);
  bullets(s, [
    ["How it works:", "Grows 100 decision trees (depth 10) independently in parallel on random row & feature subsets (√d features per split)."],
    ["Strength:", "Captures non-linear if/else thresholds and resists overfitting, though all trees carry equal weight."],
  ], { x: cx2 + 0.20, y: 3.70, w: 3.52, h: 2.95, fontSize: 11.2, gap: 8 });

  // Col 3: LightGBM (Boosting)
  const cx3 = X0 + 8.30;
  box(s, cx3, 1.18, 3.93, 5.62, "EAF6F4", C.accent6);
  T(s, "3. LightGBM (Gradient Boosting)", { x: cx3 + 0.18, y: 1.30, w: 3.56, h: 0.36, fontSize: 14.5, bold: true, color: C.accent6 });
  flowNode(s, cx3 + 0.16, 1.80, 0.98, 0.65, "Tree 1", "Initial fit", C.accent6);
  arrowR(s, cx3 + 1.18, 2.00, 0.22, 0.22, C.accent6);
  flowNode(s, cx3 + 1.44, 1.80, 0.98, 0.65, "Tree 2", "Fits errors", C.accent6);
  arrowR(s, cx3 + 2.46, 2.00, 0.22, 0.22, C.accent6);
  flowNode(s, cx3 + 2.72, 1.80, 1.05, 0.65, "Tree M", "Early stop", C.accent6);
  arrowD(s, cx3 + 1.80, 2.50, 0.32, 0.28);
  flowNode(s, cx3 + 0.35, 2.83, 3.22, 0.65, "F(x) = Σ η · f_m(x)   (η = 0.05)", "Histogram leaf-wise growth (63 leaves)", C.dk2);
  bullets(s, [
    ["How it works:", "Adds trees sequentially; each new tree fits the negative gradient (errors) of the multi-class log-loss."],
    ["Why it excels here:", "Histogram binning handles millions of rows in ~9 seconds, captures sharp threshold interactions, and stops automatically on validation loss."],
  ], { x: cx3 + 0.20, y: 3.70, w: 3.52, h: 2.95, fontSize: 11.2, gap: 8 });

  // =========================================================================
  // SLIDE 13: HOW DEEP MODELS WORK (DRAWN DEEPLOB ARCHITECTURE DIAGRAM)
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 4 · ML Models & Architectures", "How Our Deep Neural Networks Work: DeepLOB, LSTM & Transformer",
    "Dr. Kumari asked for architecture diagrams of the neural networks. Walk across the 7 blocks of DeepLOB on top: it treats the 100-second by 20/40-column order book like a 2D image! Three structured Conv layers fuse Price+Size, Bid+Ask, and all levels; parallel Inception kernels scan multiple time horizons; and an LSTM classifies the final state.");

  T(s, "DeepLOB / DeepLOB5 Architecture Diagram (~209k–214k Parameters)", { x: X0, y: 1.15, w: CW, h: 0.32, fontSize: 13.5, bold: true, color: C.dk2 });
  const dlBlocks = [
    ["2D Input", "100 steps ×\n20 or 40 cols\n(raw book)", C.accent4],
    ["Conv Block 1", "Kernel 1×2\nFuses each\nPrice + Qty", C.accent2],
    ["Conv Block 2", "Kernel 1×2\nFuses Bid side\nwith Ask side", C.accent2],
    ["Conv Block 3", "Kernel 1×5/10\nFuses across\nall book levels", "A6342C"],
    ["Inception ×3", "Parallel 1×1,\n3×1, 5×1 & Pool\n(multi-scale)", C.accent5],
    ["LSTM Layer", "64 hidden units\nSequential\ntime memory", C.accent1],
    ["Softmax Out", "Dense → 3 probs\nUP / FLAT /\nDOWN", C.dk2],
  ];
  const dbw = 1.54, dbg = 0.24;
  dlBlocks.forEach((b, i) => {
    const x = X0 + i * (dbw + dbg);
    flowNode(s, x, 1.54, dbw, 1.50, b[0], b[1], b[2]);
    if (i < 6) arrowR(s, x + dbw + 0.02, 2.15, dbg - 0.04, 0.24);
  });

  card(s, X0, 3.28, 3.95, 2.25, "Why Structured 1×2 Convolutions?",
    "Unlike ordinary images, adjacent columns in our input have exact meaning: [AskP_1, AskQ_1, BidP_1, BidQ_1]. Stride-2 (1×2) convolutions force Layer 1 to combine Price with Quantity, and Layer 2 to combine Ask with Bid (learning spread & imbalance automatically).",
    { icon: "layers", iconFill: C.accent2, size: 11.1 });

  card(s, X0 + 4.14, 3.28, 3.95, 2.25, "Pure LSTM & Transformer (TLOB)",
    "• LSTM (58k params): 2 stacked recurrent layers with forget/input gates reading 100 steps sequentially.\n• TLOB Transformer (88k params): Uses Bilinear Normalisation (BiN) + 4-head Self-Attention so step 100 can attend directly to any earlier step.",
    { icon: "brain", iconFill: C.accent5, size: 11.1 });

  card(s, X0 + 8.28, 3.28, 3.95, 2.25, "PyTorch GPU Training Setup",
    "• Loss: Class-weighted Cross-Entropy.\n• Optimiser: AdamW (lr = 5×10⁻⁴, weight decay 10⁻⁴) + ReduceLROnPlateau + grad clipping.\n• Early stopping (patience 3–4) + validation-only retry if a seed collapses.",
    { icon: "cogs", iconFill: C.accent6, size: 11.1 });

  callout(s, X0, 5.74, CW, 1.02, "Native 5-Level Adaptation (DeepLOB5):",
    "Standard DeepLOB assumes 10 price levels (40 columns). Because Indian NSE feeds provide 5 levels (20 columns), padding 20 fake zeros would distort convolutions, so we built DeepLOB5 with a native 1×5 spatial kernel in Conv Block 3.",
    "EEF3F6", C.dk2);

  // =========================================================================
  // SLIDE 14: EASY-TO-UNDERSTAND EVALUATION METRICS & ACCURACY TRAP
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 5 · Evaluation & ML Results", "Evaluation Metrics Explained Simply (And Why Plain Accuracy Misleads!)",
    "Explain clearly why plain Accuracy can trick you on imbalanced classes, using the concrete IDEA stock example on the left, and introduce our 4 clean ML metrics on the right: Macro-F1, ROC-AUC, Log-Loss/Calibration, and Latency.");

  card(s, X0, 1.20, 5.15, 5.55, "The Accuracy Trap on Imbalanced Data",
    "Real example from our Indian NSE dataset (IDEA stock, 20 s horizon):\n\n• 69.4% of true test labels are FLAT\n• 15.3% are UP, and 15.3% are DOWN\n\nA dummy 'Naive Flat' classifier that predicts FLAT 100% of the time achieves:\n   • Plain Accuracy = 69.4%  (looks great!)\n   • Recall on UP & DOWN = 0.0%  (learns nothing!)\n   • Macro-F1 Score = 0.273  (exposes the failure!)\n\nBy contrast, LightGBM achieves Macro-F1 = 0.484 and UP-vs-DOWN ROC-AUC = 0.790.",
    { icon: "warn", iconFill: C.accent2, fill: "FDF0EE", size: 11.6 });

  table(s, [
    ["ML Metric", "Plain-English Meaning", "Chance / Baseline", "Better"],
    ["Macro-F1 (Main)", "Computes F1 (Precision & Recall balance) separately for UP, FLAT, DOWN, then averages equally (1/3 each)", "0.333 (uniform)\n0.16–0.27 (all-FLAT)", "Higher ↑"],
    ["ROC-AUC (OvR & U/D)", "Probability the model ranks a true positive higher than a negative across all thresholds", "0.500 (coin flip)", "Higher ↑ (1.0 max)"],
    ["Plain Accuracy", "Overall percentage of rows predicted right (valid when classes are balanced, e.g. BTC & FI-2010)", "33.3% (balanced)", "Higher ↑"],
    ["Log-Loss & Calibration", "Penalises confident wrong probabilities; checks if '70% confident' is right 70% of the time", "1.099 (uniform)", "Lower ↓"],
    ["Block-Bootstrap 95% CI", "Resamples 5-minute blocks 1,000× to test if Model A's lead over Model B is statistically real", "CI includes 0 = tie", "CI > 0"],
  ], { x: 5.95, y: 1.20, w: 6.83, colW: [1.65, 2.83, 1.35, 1.0], rowH: [0.46, 1.06, 0.98, 0.98, 0.98, 0.98], fontSize: 10.3, hl: [1, 2] });

  // =========================================================================
  // SLIDE 15: HEADLINE ML RESULT #1 ,  TREES BEAT DEEP NETWORKS
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 5 · Evaluation & ML Results", "Cool ML Result #1: Simpler Tree Ensembles Beat Deep Neural Networks!",
    "This is our #1 Machine Learning finding! Across Synthetic data, Bitcoin, and all three 64-day Indian NSE benchmarks at k=20s, LightGBM on 25 engineered features beats LSTM and DeepLOB/DeepLOB5 in every single time segment, while training in 9 seconds on CPU instead of 18 minutes on GPU!");

  fig(s, "nse_model_ladder", X0, 1.12, CW, 3.25);

  table(s, [
    ["Dataset (k = 20 s, 4 Segments)", "Naive Momentum", "Logistic Reg.", "Random Forest", "LightGBM (Ours)", "LSTM (GPU)", "DeepLOB / DL5", "TLOB (Transformer)"],
    ["NSE · RELIANCE (Macro-F1)", "0.418", "0.397", "0.438", "0.445 (#1)", "0.351", "0.395", "0.440 (ties #1)"],
    ["NSE · NIFTY_FUT (Macro-F1)", "0.380", "0.429", "0.429", "0.443 (#1)", "0.304", "0.336", "0.396"],
    ["NSE · IDEA (Macro-F1 / U-D AUC)", "0.451 (0.52)", "0.367 (0.79)", "0.468 (0.79)", "0.484 (0.79 #1)", "0.364", "0.442 (0.74)", "0.415 (0.70)"],
    ["Bitcoin · BTC-USD (Macro-F1 / Acc)", "0.413 (41.4%)", "0.448 (45.9%)", "0.463 (47.2%)", "0.467 (47.6% #1)", "0.397", "0.401 (42.9%)", "0.449 (45.5%)"],
    ["Synthetic Ground-Truth (Macro-F1)", "0.362", "0.551", "0.578", "0.584 (#1)", "0.521", "0.559", "0.492"],
  ], { x: X0, y: 4.46, w: CW, colW: [2.55, 1.25, 1.25, 1.25, 1.65, 1.15, 1.48, 1.65], rowH: 0.37, fontSize: 10.0, hl: [1, 2, 3, 4] });

  T(s, "5-minute moving-block bootstrap 95% CIs confirm LightGBM beats DeepLOB/DeepLOB5 in 4/4 segments across BTC (+0.058), RELIANCE (+0.041), NIFTY (+0.064) & IDEA (+0.040).",
    { x: X0, y: 6.74, w: CW, h: 0.28, fontSize: 10.0, bold: true, color: C.accent1 });

  // =========================================================================
  // SLIDE 16: CONFUSION MATRICES & PROBABILITY CALIBRATION
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 5 · Evaluation & ML Results", "Diagnostic ML Plots: Confusion Matrices, ROC Curves & Calibration",
    "Every ML course teaches confusion matrices, ROC curves, and probability calibration! Left: LightGBM predicts UP, FLAT, and DOWN symmetrically (~50% diagonal recall), whereas DeepLOB develops a heavy bias toward predicting DOWN (41% of FLAT rows misclassified as DOWN). Right: LightGBM's calibration curve hugs the ideal 45-degree diagonal!");

  T(s, "Row-Normalised 3-Class Confusion Matrices (Bitcoin, 20 s)", { x: X0, y: 1.12, w: 6.0, h: 0.30, fontSize: 12.2, bold: true, color: C.dk2 });
  fig(s, "btc_confusion", X0, 1.45, 6.0, 3.75);

  T(s, "ROC Curves (AUC ~0.66) & Probability Calibration Curve", { x: 6.72, y: 1.12, w: 6.05, h: 0.30, fontSize: 12.2, bold: true, color: C.dk2 });
  fig(s, "btc_roc_calibration", 6.72, 1.45, 6.05, 3.75);

  card(s, X0, 5.35, 5.95, 1.45, "What the Confusion Matrices Reveal",
    "• FLAT is the hardest class for every model because tiny price moves sit right near the ±θ boundary.\n• LightGBM balances mistakes evenly across UP (50%) and DOWN (52%), whereas DeepLOB suffers class skew (calling 41% of true FLAT rows DOWN).",
    { icon: "check", size: 11.1 });

  card(s, 6.75, 5.35, 6.03, 1.45, "Well-Calibrated Probabilities",
    "• The reliability curve (far right) lies right on the dashed y = x diagonal: when LightGBM outputs 70% class probability, it is empirically right ~70% of the time!\n• On NSE large-tick stocks (5 s), OvR ROC-AUC reaches 0.81–0.84.",
    { icon: "chart", iconFill: C.accent5, size: 11.1 });

  // =========================================================================
  // SLIDE 17: HEADLINE ML RESULT #2 ,  FI-2010 HISTORY VS ARCHITECTURE
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 5 · Evaluation & ML Results", "Cool ML Result #2: Feature History Beats Architecture at Short Horizons",
    "Why did published papers claim DeepLOB beat simple models on FI-2010? Because in those papers, simple models were only given 1 single snapshot while DeepLOB saw 100 past steps! When we give simple Logistic Regression summary features of the same 100-step history, Logistic Regression beats DeepLOB 86.1% vs 83.1% at k=10! DeepLOB wins only at long horizons k=50–100.");

  fig(s, "fi2010", X0, 1.22, 6.45, 4.15);

  card(s, 7.20, 1.22, 5.58, 2.05, "1. Faithful DeepLOB Replication",
    "On the standard FI-2010 academic benchmark (Setup 2: train days 1–7, test days 8–10), our PyTorch DeepLOB reaches 83.1% accuracy at k=10 and 77.5% at k=100, reproducing Zhang et al. (2019) within 0.7–2.3 percentage points.",
    { icon: "check", iconFill: C.accent6, size: 11.3 });

  card(s, 7.20, 3.42, 5.58, 2.15, "2. The Crossover: Short vs Long Horizons",
    "• Short horizons (k = 10, 20): Giving Logistic Regression the same past-history window reverses the result: Logistic wins 86.1% vs 83.1% (k=10) and 78.3% vs 74.1% (k=20)!\n• Long horizons (k = 50, 100): DeepLOB clearly wins (78.2% vs 66.5% at k=50), where multi-scale Inception + LSTM patterns add real value.",
    { icon: "bulb", iconFill: C.accent3, size: 11.2 });

  callout(s, X0, 5.75, CW, 1.00, "Key Machine Learning Lesson:",
    "Before crediting a 200,000-parameter neural network for higher accuracy, give simpler models access to the same temporal context: at short horizons, input representation matters more than network depth.",
    "FDF5E6", C.accent3);

  // =========================================================================
  // SLIDE 18: HEADLINE ML RESULT #3 ,  SHAP, SALIENCY & OVERFITTING EXTRA FEATURES
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 5 · Evaluation & ML Results", "Cool ML Result #3: Explainable AI (SHAP & Saliency) & When Extra Features Hurt",
    "Two textbook ML insights on feature selection and explainability! Left: 5-stage feature ablation on NSE shows adding 5 extra noisy NSE columns (Stage E) actually lowers test F1 on 6 of 8 stocks because trees overfit training drift! Bottom-left: CNN gradient saliency proves DeepLOB learns to look at the exact same recent Level-1 inputs we engineered!");

  fig(s, "nse_ablation_shap", X0, 1.15, 6.85, 3.05);
  fig(s, "deeplob_attribution", X0, 4.28, 6.85, 2.55);

  card(s, 7.58, 1.18, 5.20, 2.65, "Textbook Overfitting: More Columns Can Hurt!",
    "• Stages A → D (1 to 25 features): Each clean feature group boosts test Macro-F1 (lags add +0.03 to +0.07; deeper levels L2–5 add +0.029 when L1 queues are thin).\n• Stage D → E (+5 noisy full-book columns): Drops test F1 on 6/8 NSE stocks (−0.023 ITC, −0.020 RELIANCE), even though TreeSHAP gives them 25.5% split importance!\n• Why: Distant resting orders drift across days, so trees memorise training-set noise.",
    { icon: "warn", iconFill: C.accent2, fill: "FDF0EE", size: 11.1 });

  card(s, 7.58, 3.98, 5.20, 2.82, "TreeSHAP & CNN Saliency Agree Completely",
    "• Bottom-left saliency map: Computing |Gradient × Input| across DeepLOB's 100×40 window shows 63–70% of all attention sits on the last 10 seconds and Level 1.\n• DeepLOB spends 200k parameters rediscovering from scratch the exact top-of-book imbalance and recent flow lags that our 25 features give LightGBM directly!",
    { icon: "search", iconFill: C.accent1, size: 11.1 });

  // =========================================================================
  // SLIDE 19: HEADLINE ML RESULT #4 ,  DISTRIBUTION SHIFT VS ADAPTIVE NORMALISATION
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 5 · Evaluation & ML Results", "Cool ML Result #4: How Distribution Shift Breaks Standard Neural Networks",
    "Explain this super intuitive ML phenomenon we discovered on NSE Segment 3! Between training days and test days, NIFTY's price rose +10.7% (from ₹22,540 to ₹24,949). Standard neural nets with fixed training normalisation crashed to random guessing on test, whereas adaptive per-window Bilinear Normalisation (TLOB) and scale-free ratio features (LightGBM) had zero drop!");

  // Visual Covariate Shift Flowchart
  flowNode(s, X0, 1.30, 3.55, 1.25, "Training Days (Oct 19–Nov 2)", "NIFTY Price ≈ ₹22,540\nStatic scaler learns train range", C.accent5);
  arrowR(s, X0 + 3.63, 1.78, 0.55, 0.30, C.accent2);
  flowNode(s, X0 + 4.26, 1.30, 3.55, 1.25, "Test Days (Nov 5–Nov 9)", "NIFTY Rallies +10.7% → ₹24,949\n(Covariate / Distribution Shift!)", C.accent2);
  arrowR(s, X0 + 7.89, 1.78, 0.55, 0.30, C.dk2);
  flowNode(s, X0 + 8.52, 1.30, 3.71, 1.25, "Who Survives the Shift?", "Depends entirely on input\nrepresentation & normalisation!", C.dk2);

  card(s, X0, 2.85, 3.92, 2.75, "Static-Scaled LSTM & DeepLOB5",
    "Test Macro-F1:  0.184  &  0.197\n(ROC-AUC ≈ 0.50 = Coin Flip!)\n\n• Even after validation retry recovered Val F1 (0.322), raw prices divided by a fixed constant shifted out-of-distribution on Test.\n• On stable segments 0–2, DeepLOB5 scored 0.382.",
    { icon: "warn", iconFill: C.accent2, fill: "FDF0EE", size: 11.2 });

  card(s, X0 + 4.15, 2.85, 3.92, 2.75, "TLOB Transformer (+ BiN Layer)",
    "Test Macro-F1:  0.439\n(Zero degradation on Segment 3!)\n\n• Bilinear Normalisation (BiN) re-centres and scales each 100-second window dynamically across both time and features inside the network.",
    { icon: "layers", iconFill: C.accent5, fill: "EBF3FA", size: 11.2 });

  card(s, X0 + 8.30, 2.85, 3.93, 2.75, "LightGBM on Ratio Features",
    "Test Macro-F1:  0.432\n(Consistent across all 4 segments!)\n\n• Engineered ratios like (Bid − Ask)/(Bid + Ask) and tick-normalised spreads are naturally invariant whether price is ₹22,000 or ₹25,000.",
    { icon: "check", iconFill: C.accent6, fill: "E6F4F0", size: 11.2 });

  callout(s, X0, 5.82, CW, 0.94, "General ML Takeaway:",
    "Real-world time series are non-stationary. Feeding raw magnitudes with static training-set scaling into neural networks fails under level shifts; either engineer scale-invariant ratios or use adaptive per-window normalisation.",
    "EEF3F6", C.dk2);

  // =========================================================================
  // SLIDE 20: SCALING TO 25 INDIAN STOCKS (36M ROWS) & HORIZON DECAY
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 6 · Large-Scale NSE Study & Speed", "Scaling Across 25 Indian Stocks (36M Rows) & How Fast Signals Fade",
    "Show the scale of our Indian NSE dataset: 25 stocks & index derivatives across 10 sectors = 36 million rows! Left: stocks with coarser price steps form deeper waiting queues and reach 0.81–0.84 ROC-AUC (r = +0.80). Right: across all 8 core stocks, predictive skill peaks at 5 seconds and fades back to baseline by 100 seconds.");

  T(s, "25 Indian NSE Instruments (36M Rows): Predictability Law", { x: X0, y: 1.12, w: 6.1, h: 0.30, fontSize: 12.2, bold: true, color: C.dk2 });
  fig(s, "nse_universe_25", X0, 1.44, 6.15, 3.70);

  T(s, "How Quickly Order-Book Memory Fades (5 s to 100 s)", { x: 6.85, y: 1.12, w: 5.9, h: 0.30, fontSize: 12.2, bold: true, color: C.dk2 });
  fig(s, "nse_signal_decay", 6.85, 1.44, 5.93, 3.70);

  callout(s, X0, 5.32, CW, 1.48, "Two large-scale patterns across 36 million Indian NSE rows:",
    "1. Data structure sets the ML ceiling (left): 5-second LightGBM ROC-AUC scales tightly with relative price step (log-linear r = +0.80, Spearman ρ = +0.77, p < 10⁻⁵), rising from 0.72 on index futures to 0.84 on large-step stocks, confirmed by NSE's ₹250 step-size jump (ONGC 0.755 vs ITC 0.808 at nearly the same share price).\n2. Short memory window (right): Order-book information peaks at 5 seconds and decays to baseline by 50–100 seconds.",
    "E4F2F2", C.accent1);

  // =========================================================================
  // SLIDE 21: SPEED VS ACCURACY & C++ COMPILER + REAL-WORLD DEPLOYMENT LESSON
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT" });
  header(s, "Section 6 · Large-Scale NSE Study & Speed", "Inference Speed vs Accuracy (10.5 µs C++ Tree Engine) & Practical Impact",
    "Finish with two practical ML deployment lessons! Left: usually in ML, higher accuracy costs slower speed, here there is NO trade-off! Compiling our trained LightGBM trees into standalone C++ gives bit-identical predictions in 10.5 microseconds (500x faster than GPU DeepLOB) AND higher accuracy! Right: an easy-to-understand intuition on why 73% accuracy still needs low transaction cost.");

  fig(s, "speed_vs_accuracy", X0, 1.18, 6.15, 3.95);

  card(s, 6.90, 1.18, 5.88, 2.30, "No Speed–Accuracy Trade-Off (C++ Tree Compiler)",
    "We wrote a Python-to-C++ code generator that converts trained LightGBM trees into pure C++ if/else code:\n• Bit-identical output on 20,000 test rows (max diff < 10⁻¹²).\n• Single-row latency: 10.5 µs in C++ vs 150 µs in Python vs 2,400–7,100 µs for DeepLOB on GPU (230–540× faster!).",
    { icon: "code", iconFill: C.accent6, size: 11.2 });

  card(s, 6.90, 3.64, 5.88, 3.12, "Intuitive Real-World Lesson: Accuracy vs Friction",
    "• Filtering for top-2% high-confidence predictions raises directional accuracy to 72.9% (NIFTY) and 69.7% (IDEA)!\n• Simple math intuition: If a stock moves ₹1 on average in 20 s, a 73% accurate classifier gains ~₹0.46 per call. But if buying & selling immediately costs ₹2 in price gap + fees, standalone back-and-forth trading loses −₹1.54 (seen across all 25 NSE symbols).\n• Best practical ML use: Use the classifier to time a purchase or sale a user already plans to make!",
    { icon: "bulb", iconFill: C.accent3, fill: "FDF5E6", size: 11.0 });

  T(s, "Top-left corner = ideal ML model (fastest AND most accurate).", { x: X0, y: 5.20, w: 6.15, h: 0.35, fontSize: 11.2, italic: true, color: C.accent4, align: "center" });

  // =========================================================================
  // SLIDE 22: CONCLUSION & SUMMARY (DARK)
  // =========================================================================
  s = pres.addSlide({ masterName: "DARK" });
  T(s, "Conclusion: Four Core Machine Learning Takeaways", { x: X0, y: 0.48, w: CW, h: 0.58, fontSize: 30, bold: true, color: C.lt1, fontFace: THEME.headFontFace });

  const boxW = 5.92, boxH = 2.05;
  const cards4 = [
    [X0, 1.28, "01 · Leakage-Free Time-Series Hygiene", C.accent3,
      "Random K-fold and full-sample scaling create fake time-series accuracy. Chronological splits with purged gaps, daily resets, and shuffled-label controls (43 unit tests) keep evaluation honest."],
    [X0 + 6.28, 1.28, "02 · Feature Engineering + GBDTs Beat Deep Nets", "63D2D6",
      "At 5–20 s horizons across Bitcoin, Synthetic, and 64-day Indian NSE data, LightGBM on 25 engineered features beats LSTM & DeepLOB in accuracy/Macro-F1, while running 230–540× faster in C++ (10.5 µs)."],
    [X0, 3.52, "03 · Feature Selection & Distribution Shift Matter", "8CC4FF",
      "Adding 5 noisy columns lowered test F1 on 6/8 NSE stocks (overfitting), and a +10.7% price shift broke static CNN/LSTM scaling, while scale-free ratios (LightGBM) and adaptive BiN (TLOB) survived."],
    [X0 + 6.28, 3.52, "04 · Pick Metrics That Match Reality", "F49D96",
      "Plain accuracy fails under class imbalance (Macro-F1 & ROC-AUC up to 0.84 across 25 Indian stocks tell the truth), and real-world deployment must account for action costs alongside accuracy."],
  ];
  cards4.forEach(([x, y, hd, hcol, bd]) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: boxW, h: boxH, rectRadius: 0.08, fill: { color: "1A3B54" }, line: { color: "2E5B7B", pt: 1 } });
    T(s, hd, { x: x + 0.20, y: y + 0.14, w: boxW - 0.40, h: 0.36, fontSize: 15, bold: true, color: hcol, fontFace: THEME.headFontFace });
    T(s, bd, { x: x + 0.20, y: y + 0.56, w: boxW - 0.40, h: boxH - 0.68, fontSize: 11.8, color: "E6EFF5" });
  });

  T(s, "Thank You  ·  Questions?", { x: X0, y: 5.82, w: CW, h: 0.50, fontSize: 25, bold: true, color: C.lt1, fontFace: THEME.headFontFace, align: "center" });
  T(s, "Aarav Singhal (1024030539)  ·  Sharan Sharma (1024030511)  ·  Guided by Dr. Sanju Kumari  ·  UML501 Machine Learning",
    { x: X0, y: 6.42, w: CW, h: 0.35, fontSize: 12.2, color: "9BB4C7", align: "center" });
  s.addNotes("Summarise our four core ML takeaways: (1) strict anti-leakage time-series preprocessing; (2) feature-engineered LightGBM beating deep learning in both accuracy and 10.5-microsecond speed at short horizons; (3) textbook lessons on overfitting noisy features and surviving distribution shift; and (4) choosing proper imbalanced-class metrics (Macro-F1, ROC-AUC up to 0.84 across 25 Indian stocks). Thank you, Dr. Kumari!");

  await pres.writeFile({ fileName: OUT });
  console.log("Wrote ML-focused 22-slide presentation:", OUT);
}

main().catch((e) => { console.error(e); process.exit(1); });

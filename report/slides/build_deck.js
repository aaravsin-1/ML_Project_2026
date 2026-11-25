// Builds report/LOB_ML_Project_Presentation.pptx from the report's content and figures.
// Run from report/slides:  npm install && node build_deck.js
// Figures are PNG exports of report/figures/*.pdf (pdftoppm -png -r 300 -singlefile).
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const SKILL_THEME = process.env.APPLY_THEME_JS;
const IMG = path.join(__dirname, "img");
const OUT = path.join(__dirname, "..", "LOB_ML_Project_Presentation.pptx");

const THEME = {
  name: "LOB ML",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2633", lt1: "FFFFFF", dk2: "123047", lt2: "EEF3F6",
    accent1: "0E7C86", // teal  = bid side / tabular models
    accent2: "C8453C", // coral = ask side / deep models / costs
    accent3: "D99A2B", // amber = highlight
    accent4: "5A6B7B", // slate = baselines
    accent5: "2F6FA8", // blue
    accent6: "6E9F72", // green
    hlink: "2F6FA8", folHlink: "6B4E9B",
  },
};
const HX = THEME.colors;
const W = 13.333, H = 7.5, X0 = 0.6, CW = W - 2 * X0;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Machine Learning on Limit Order Books";
pres.author = "Aarav Singhal";
pres.subject = "Machine Learning project presentation";
const C = pres.SchemeColor;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: X0, y: 2.0, w: CW, h: 1.6, fontFace: THEME.headFontFace, fontSize: 44, bold: true, color: C.background1, valign: "bottom", margin: 0, align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: X0, y: 3.85, w: CW, h: 1.2, fontSize: 20, color: "CFE3E6", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "num", type: "body", x: X0, y: 1.9, w: 3, h: 1.1, fontFace: THEME.headFontFace, fontSize: 60, bold: true, color: C.accent3, margin: 0, valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: X0, y: 3.05, w: CW, h: 1.0, fontFace: THEME.headFontFace, fontSize: 40, bold: true, color: C.background1, margin: 0, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: X0, y: 4.15, w: 10.5, h: 1.4, fontSize: 18, color: "CFE3E6", margin: 0, valign: "top" }, text: "" } },
  ],
  slideNumber: { x: W - 1.1, y: H - 0.5, w: 0.6, h: 0.3, fontSize: 10, color: "9FB4C2", align: "right" },
});
pres.defineSlideMaster({
  title: "DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: X0, y: 0.32, w: CW, h: 0.8, fontFace: THEME.headFontFace, fontSize: 30, bold: true, color: C.background1, valign: "middle", margin: 0, align: "left" }, text: "" } },
  ],
  slideNumber: { x: W - 1.1, y: H - 0.45, w: 0.5, h: 0.3, fontSize: 10, color: "9FB4C2", align: "right", margin: 0 },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: X0, y: 0.32, w: CW, h: 0.8, fontFace: THEME.headFontFace, fontSize: 30, bold: true, color: C.text2, valign: "middle", margin: 0, align: "left" }, text: "" } },
    { text: { text: "Machine Learning on Limit Order Books", options: { x: X0, y: H - 0.45, w: 6, h: 0.3, fontSize: 10, color: "8A97A3", margin: 0 } } },
  ],
  slideNumber: { x: W - 1.1, y: H - 0.45, w: 0.5, h: 0.3, fontSize: 10, color: "8A97A3", align: "right", margin: 0 },
});

// ---------- helpers ----------
let ICONS = {};
async function iconPng(Comp, color) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

let curSection = "";
function section(num, title, body, notes) {
  curSection = title;
  pres.addSection({ title });
  const s = pres.addSlide({ masterName: "SECTION", sectionTitle: curSection });
  s.addText(num, { placeholder: "num" });
  s.addText(title, { placeholder: "title" });
  s.addText(body, { placeholder: "body" });
  if (notes) s.addNotes(notes);
  return s;
}
function content(title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: curSection });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function T(s, text, o) {
  s.addText(text, Object.assign({ isTextBox: true, fontSize: 14, color: C.text1, valign: "top", margin: 0 }, o));
}
// bullet list: items are strings or [lead, rest]
function bullets(s, items, o) {
  const runs = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    const b = { bullet: { indent: 16 }, paraSpaceAfter: o.gap ?? 6 };
    if (Array.isArray(it)) {
      runs.push({ text: it[0] + " ", options: Object.assign({}, b, { bold: true, color: C.text2 }) });
      runs.push({ text: it[1], options: { breakLine: !last } });
    } else runs.push({ text: it, options: Object.assign({}, b, { breakLine: !last }) });
  });
  const opts = Object.assign({}, o); delete opts.gap;
  T(s, runs, opts);
}
function box(s, x, y, w, h, fill, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || C.background2 }, line: { type: "none" }, objectName: name });
}
function icon(s, key, x, y, d, fill) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill || C.accent1 }, line: { type: "none" } });
  const p = d * 0.22;
  s.addImage({ data: ICONS[key], x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p });
}
// card: tinted rounded box with optional icon, heading and body
function card(s, x, y, w, h, head, body, o = {}) {
  box(s, x, y, w, h, o.fill, "card: " + head);
  let tx = x + 0.2, tw = w - 0.4;
  if (o.icon) { icon(s, o.icon, x + 0.2, y + 0.2, 0.55, o.iconFill); tx = x + 0.95; tw = w - 1.15; }
  T(s, head, { x: tx, y: y + 0.18, w: tw, h: 0.6, fontSize: o.headSize || 16, bold: true, color: o.headColor || C.text2, valign: o.icon ? "middle" : "top" });
  const by = o.icon ? y + 0.88 : y + 0.62;
  if (body) T(s, body, { x: x + 0.2, y: by, w: w - 0.4, h: h - (by - y) - 0.12, fontSize: o.size || 14, color: C.text1 });
}
// "In plain words" callout
function plain(s, x, y, w, h, text, label) {
  box(s, x, y, w, h, "E3F1F2", "plain-words box");
  icon(s, "bulb", x + 0.18, y + 0.16, 0.42, C.accent3);
  T(s, label || "In plain words", { x: x + 0.72, y: y + 0.16, w: w - 0.9, h: 0.42, fontSize: 14, bold: true, color: C.accent1, valign: "middle" });
  T(s, text, { x: x + 0.2, y: y + 0.66, w: w - 0.4, h: h - 0.76, fontSize: 14 });
}
function pngSize(file) {
  const b = fs.readFileSync(file);
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}
// figure fitted (aspect kept) inside box, centred horizontally, top-aligned
function fig(s, name, x, y, w, h, alt) {
  const f = path.join(IMG, name + ".png");
  const [pw, ph] = pngSize(f);
  let iw = w, ih = (w * ph) / pw;
  if (ih > h) { ih = h; iw = (h * pw) / ph; }
  s.addImage({ path: f, x: x + (w - iw) / 2, y, w: iw, h: ih, altText: alt || name });
  return { x: x + (w - iw) / 2, y, w: iw, h: ih };
}
function stat(s, x, y, w, big, label, color, size) {
  T(s, big, { x, y, w, h: 0.85, fontSize: size || 40, bold: true, color: color || C.accent1, fontFace: THEME.headFontFace, valign: "bottom" });
  T(s, label, { x, y: y + 0.9, w, h: 0.75, fontSize: 13, color: C.accent4 });
}
const BORDER = { type: "solid", pt: 0.75, color: "C9D3DB" };
function table(s, rows, o) {
  const fs_ = o.fontSize || 12;
  const data = rows.map((r, i) => r.map((c) => {
    const cell = typeof c === "object" && c !== null && c.text !== undefined ? c : { text: String(c) };
    const base = { fontSize: fs_, color: C.text1, border: BORDER, valign: "middle", margin: [3, 5, 3, 5] };
    if (i === 0) Object.assign(base, { bold: true, color: C.background1, fill: { color: C.text2 } });
    else if (o.hl && o.hl.includes(i)) Object.assign(base, { bold: true, fill: { color: "D7ECEE" } });
    else if (i % 2 === 0) base.fill = { color: C.background2 };
    return { text: cell.text, options: Object.assign(base, cell.options || {}) };
  }));
  s.addTable(data, { x: o.x, y: o.y, w: o.w, colW: o.colW, rowH: o.rowH, align: o.align || "left", autoPage: false });
}
function arrowR(s, x, y, w) {
  s.addShape(pres.shapes.RIGHT_ARROW, { x, y, w, h: 0.32, fill: { color: "9FB0BE" }, line: { type: "none" } });
}
const chartText = { catAxisLabelColor: "4A5866", valAxisLabelColor: "4A5866", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontSize: 11, legendFontSize: 12, dataLabelColor: "1B2633", titleColor: "123047", titleFontSize: 14 };

async function main() {
  const icons = {
    bulb: fa.FaLightbulb, db: fa.FaDatabase, btc: fa.FaBitcoin, chart: fa.FaChartLine, cogs: fa.FaCogs, brain: fa.FaBrain,
    clock: fa.FaClock, dollar: fa.FaDollarSign, scale: fa.FaBalanceScale, flask: fa.FaFlask, layers: fa.FaLayerGroup,
    shield: fa.FaShieldAlt, tree: fa.FaTree, check: fa.FaCheckCircle, warn: fa.FaExclamationTriangle, cloud: fa.FaCloudDownloadAlt,
    uni: fa.FaUniversity, random: fa.FaRandom, search: fa.FaSearch, bolt: fa.FaBolt, code: fa.FaCode, building: fa.FaBuilding,
    server: fa.FaServer, book: fa.FaBook, ruler: fa.FaRulerHorizontal, filter: fa.FaFilter, cut: fa.FaCut, tag: fa.FaTag,
    eye: fa.FaEye, hourglass: fa.FaHourglassHalf, target: fa.FaBullseye, gear: fa.FaMicrochip, list: fa.FaListOl, road: fa.FaRoad,
  };
  for (const [k, v] of Object.entries(icons)) ICONS[k] = await iconPng(v, "FFFFFF");

  // =====================================================================
  // 1. Title
  pres.addSection({ title: "Opening" }); curSection = "Opening";
  let s = pres.addSlide({ masterName: "TITLE", sectionTitle: curSection });
  s.addText("Machine Learning on Limit Order Books", { placeholder: "title" });
  s.addText("Predicting the next few seconds of a market from its order book, and finding out whether accuracy and speed turn into profit", { placeholder: "body" });
  T(s, [
    { text: "Aarav Singhal  ·  1024030539", options: { bold: true, breakLine: true } },
    { text: "Machine Learning project  ·  Thapar Institute of Engineering and Technology  ·  November 2026" },
  ], { x: X0, y: 5.75, w: CW, h: 0.8, fontSize: 16, color: C.background1 });
  s.addNotes("This project uses limit order book data, a far richer input than ordinary price charts, to ask two questions: how much can machine learning predict about the next few seconds of price movement, and does a correct prediction actually make money once costs and speed are taken into account?");

  // 2. Project at a glance
  s = content("The project at a glance",
    "Two questions drive everything. First the ML question: can models predict short-term price moves from the order book, which models are best, and is the gain statistically real? Second the economic question: does a correct prediction make money, given model accuracy, model speed and trading cost? The four numbers on the right are the headline answers; each is explained later in the deck.");
  card(s, X0, 1.35, 6.1, 2.4, "Question 1: How predictable is the book?",
    "Using a live record of every buy and sell order waiting in a market, can machine learning predict whether the price will go UP, stay FLAT or go DOWN over the next few seconds? Which models do it best, and is the improvement statistically real?", { icon: "search" });
  card(s, X0, 3.95, 6.1, 2.6, "Question 2: Does a correct prediction make money?",
    "A trading system has to act on each prediction before it expires and has to pay to trade. How do the model's accuracy, its speed (how long one prediction takes) and the trading cost combine to decide whether there is any profit?", { icon: "dollar", iconFill: C.accent2 });
  const sx = 7.15, sw = 2.75;
  stat(s, sx, 1.3, sw, "0.467", "Best macro-F1 on Bitcoin (LightGBM). Random guessing scores 0.333", C.accent1);
  stat(s, sx + 2.95, 1.3, sw, "0.658", "ROC-AUC of the same model (0.5 = no skill, 1 = perfect)", C.accent1);
  stat(s, sx, 3.65, sw, "10.5 µs", "Time for one prediction after compiling LightGBM to C++", C.accent5);
  stat(s, sx + 2.95, 3.65, sw, "35×", "How much more accuracy was worth than speed in the latency test", C.accent3);
  T(s, "The book is predictable, but only modestly. Simple, fast tree models beat deep networks at these short horizons, and trading costs, not accuracy, decide whether any of it is profitable.",
    { x: sx, y: 5.6, w: 5.6, h: 1.0, fontSize: 14, italic: true, color: C.text2 });

  // 3. Roadmap
  s = content("Roadmap",
    "The deck follows the report: background first, then data, features, models, results, and finally the link from prediction to profit.");
  const road = [
    ["1", "Order books, explained", "What a limit order book is and why it is a new kind of input for ML"],
    ["2", "The data", "Six datasets, how they were collected, explored and cleaned"],
    ["3", "Features", "Turning 40 raw numbers per second into 27 meaningful signals"],
    ["4", "Models and validation", "From naive baselines to deep networks, and how we kept the test honest"],
    ["5", "Results", "Accuracy, AUC, significance, signal decay, benchmarks, explanations"],
    ["6", "Prediction to profit", "How accuracy, speed and cost together decide the P&L"],
    ["7", "Conclusions", "What it means, limitations and future work"],
  ];
  road.forEach((r, i) => {
    const col = i < 4 ? 0 : 1, row = i < 4 ? i : i - 4;
    const x = X0 + col * 6.2, y = 1.4 + row * 1.32;
    box(s, x, y, 5.9, 1.12);
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.26, w: 0.6, h: 0.6, fill: { color: i < 4 ? C.accent1 : C.accent2 }, line: { type: "none" } });
    T(s, r[0], { x: x + 0.2, y: y + 0.26, w: 0.6, h: 0.6, fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle" });
    T(s, r[1], { x: x + 1.0, y: y + 0.14, w: 4.75, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
    T(s, r[2], { x: x + 1.0, y: y + 0.55, w: 4.75, h: 0.5, fontSize: 13, color: C.accent4 });
  });

  // =====================================================================
  section("01", "Order books, explained", "What a limit order book is, the vocabulary used in the rest of the deck, and why this data is a novel input for machine learning.");

  // 4. What is a LOB (drawn)
  s = content("What is a limit order book?",
    "Walk through the picture. Sellers' resting orders (asks) are above, buyers' (bids) below. The best ask is the cheapest anyone will sell for right now; the best bid is the most anyone will pay. The gap is the spread, and the halfway point is the mid price, which is what the models predict. A market order trades immediately by taking resting orders from the other side. Each snapshot of 10 levels on each side, with a price and a size for each, is 40 numbers.");
  const ladder = [["100.05", 300, "a"], ["100.04", 520, "a"], ["100.03", 260, "a"], ["100.02", 410, "a"], ["100.01", 180, "a"],
    ["99.99", 450, "b"], ["99.98", 330, "b"], ["99.97", 610, "b"], ["99.96", 240, "b"], ["99.95", 380, "b"]];
  const ly0 = 1.45, rh = 0.43, px = 2.55, bx = 3.75;
  T(s, "Price ($)", { x: px - 0.1, y: ly0 - 0.12, w: 1.1, h: 0.3, fontSize: 12, bold: true, color: C.accent4, align: "center" });
  T(s, "Size waiting (shares)", { x: bx, y: ly0 - 0.12, w: 2.6, h: 0.3, fontSize: 12, bold: true, color: C.accent4 });
  ladder.forEach((l, i) => {
    const y = ly0 + 0.25 + i * rh + (l[2] === "b" ? 0.42 : 0);
    const col = l[2] === "a" ? C.accent2 : C.accent1;
    T(s, l[0], { x: px - 0.1, y, w: 1.1, h: rh - 0.06, fontSize: 14, align: "center", valign: "middle", bold: i === 4 || i === 5 });
    s.addShape(pres.shapes.RECTANGLE, { x: bx, y: y + 0.04, w: (l[1] / 610) * 2.4, h: rh - 0.14, fill: { color: col }, line: { type: "none" } });
    T(s, String(l[1]), { x: bx + (l[1] / 610) * 2.4 + 0.08, y, w: 0.7, h: rh - 0.06, fontSize: 12, color: C.accent4, valign: "middle" });
  });
  T(s, "ASKS\n(sell orders)", { x: X0, y: ly0 + 0.55, w: 1.7, h: 1.0, fontSize: 16, bold: true, color: C.accent2 });
  T(s, "BIDS\n(buy orders)", { x: X0, y: ly0 + 3.3, w: 1.7, h: 1.0, fontSize: 16, bold: true, color: C.accent1 });
  const gapY = ly0 + 0.25 + 5 * rh;
  T(s, "spread = 2¢   ·   mid = $100.00", { x: px - 0.1, y: gapY - 0.02, w: 3.9, h: 0.42, fontSize: 13, italic: true, bold: true, color: C.accent3, valign: "middle" });
  T(s, "best ask →", { x: X0 + 0.65, y: ly0 + 0.25 + 4 * rh, w: 1.3, h: rh - 0.06, fontSize: 12, color: C.accent2, align: "right", valign: "middle" });
  T(s, "best bid →", { x: X0 + 0.65, y: gapY + 0.42, w: 1.3, h: rh - 0.06, fontSize: 12, color: C.accent1, align: "right", valign: "middle" });
  const gx = 7.25, gw = 5.48;
  card(s, gx, 1.35, gw, 1.25, "Limit order", "“I will buy 200 shares at $99.99 or less.” It waits (rests) in the book until someone trades with it or it is cancelled.", { size: 13 });
  card(s, gx, 2.72, gw, 1.25, "Market order", "“Buy now at whatever price.” It crosses the spread and takes the cheapest asks, walking up the levels if it is large.", { size: 13 });
  card(s, gx, 4.09, gw, 1.25, "Bid, ask, spread, mid", "Best bid = highest buy price, best ask = lowest sell price. Spread = ask − bid. Mid price = their average: the price our models forecast.", { size: 13 });
  card(s, gx, 5.46, gw, 1.25, "One snapshot = 40 numbers", "10 levels × (ask price, ask size, bid price, bid size). The book updates thousands of times per second on busy markets.", { size: 13 });

  // 5. Real book
  s = content("A real order book: one second of Bitcoin on Coinbase",
    "This is the raw material. On the left, the first 10 price levels on each side at one instant, with the amount of Bitcoin resting at each. On the right, one hour of the mid price with the 10th level on each side drawn around it, showing how tight the book is relative to price movement. The dataset has about 1.03 million such one-second snapshots.");
  fig(s, "btc_book_snapshot", X0, 1.3, CW, 4.3, "Coinbase BTC-USD book snapshot and one hour of mid price");
  plain(s, X0, 5.65, CW, 1.15, "Left: how much Bitcoin buyers (bids) and sellers (asks) have waiting at each price at one moment. Right: one hour of the mid price. We have about 1.03 million of these one-second pictures.");

  // 6. Vocabulary
  s = content("Three ideas used throughout: tick, basis point, horizon",
    "These three terms come up on almost every results slide. The tick size turns out to matter a great deal: on large-tick stocks queues are long and the mid price moves rarely, so queue imbalance is very informative. On small-tick stocks and Bitcoin, the book changes constantly and is harder to read. Basis points are how we measure tiny moves and costs; one basis point is one hundredth of one percent.");
  card(s, X0, 1.35, 3.85, 3.5, "Tick size", "The smallest allowed price step (for example 1 cent).\n\nLarge-tick stocks (INTC, MSFT): 1¢ is 3.3–3.7 bp of the price, so about 13,000 shares queue at each best price.\n\nSmall-tick (AAPL, AMZN, GOOG): 0.17–0.45 bp, about 100 shares. Bitcoin: 0.0017 bp.", { icon: "ruler", size: 13 });
  card(s, X0 + 4.14, 1.35, 3.85, 3.5, "Basis point (bp)", "1 bp = 0.01% of the price.\n\nOn a $60,000 Bitcoin, 1 bp is $6.\n\nThe typical Bitcoin move over 20 seconds is only 3.3 bp, while a round-trip trading fee is 8 bp. Remember this pair of numbers: it decides the profit story.", { icon: "target", iconFill: C.accent3, size: 13 });
  card(s, X0 + 8.28, 1.35, 3.85, 3.5, "Horizon (k)", "How far ahead we predict: k rows of data.\n\nBitcoin: 10, 20, 50, 100 seconds.\nNasdaq: 100 ms to 30 s.\nFI-2010: 10 to 100 order events.\n\nShorter horizons are easier to predict but give less time to act.", { icon: "hourglass", iconFill: C.accent5, size: 13 });
  plain(s, X0, 5.1, CW, 1.6, "A large tick is like a big queue at a shop counter: the price only moves when a whole queue is used up, so watching the queues tells you a lot. A tiny tick is like dozens of tills with one person each: queues appear and vanish constantly, so they tell you less.", "Analogy");

  // 7. Novelty
  s = content("Why LOB data is a novel input for machine learning",
    "Most ML projects on prices use bars: one open, high, low, close and volume record per minute or per day. Bars only record trades that already happened. The order book also records the orders that are waiting to trade, which is where buying and selling pressure shows up before the price moves. The cost of that richness is a much harder learning problem: enormous, autocorrelated and non-stationary data, easy leakage, and moves that are tiny compared with costs.");
  table(s, [
    ["", "OHLCV price bars (typical ML)", "Limit order book (this project)"],
    ["What is recorded", "Prices of completed trades, summarised per bar", "Every resting order at 10 levels per side, plus each add, trade and cancel"],
    ["Time resolution", "Minutes to days", "100 ms (Nasdaq) to 1 s (Bitcoin); events microseconds apart"],
    ["Rows per day", "390 (1-minute stock bars)", "86,400 (Bitcoin, 1 s) to 234,000 (Nasdaq, 100 ms)"],
    ["Values per row", "5", "40 raw values, plus order flows"],
    ["Shows supply and demand before the price moves?", { text: "No", options: { color: C.accent2, bold: true } }, { text: "Yes: queue sizes, imbalance, cancellations", options: { color: C.accent1, bold: true } }],
    ["Main ML difficulty", "Very low signal-to-noise", "Huge, autocorrelated, non-stationary data; leakage; tiny moves relative to costs"],
    ["Who can trade on the prediction", "Anyone, slowly", "Only a system fast enough to act within seconds"],
  ], { x: X0, y: 1.35, w: CW, colW: [3.0, 4.0, 5.13], fontSize: 13, rowH: 0.56 });
  T(s, "The novelty is the information the model gets to see: the queue of waiting orders, not just the trail of past prices.", { x: X0, y: 6.15, w: CW, h: 0.5, fontSize: 15, italic: true, bold: true, color: C.accent1 });

  // 8. Intuition: imbalance
  s = content("Intuition: a lopsided queue hints at the next move",
    "Queue imbalance is the single most important idea in the project. If many more shares are waiting to buy at the best bid than to sell at the best ask, the ask queue is likely to be used up first, so the next price change is more likely to be up. The imbalance number runs from minus one (all ask) to plus one (all bid). Literature: Stoikov's micro-price and Cont et al.'s order flow imbalance build on exactly this idea.");
  // two queues drawn
  const qx = X0 + 0.2;
  T(s, "Best ask: 400 shares", { x: qx, y: 1.5, w: 3.2, h: 0.4, fontSize: 15, bold: true, color: C.accent2 });
  s.addShape(pres.shapes.RECTANGLE, { x: qx, y: 1.95, w: 2.4, h: 0.55, fill: { color: C.accent2 }, line: { type: "none" } });
  T(s, "Best bid: 600 shares", { x: qx, y: 2.8, w: 3.2, h: 0.4, fontSize: 15, bold: true, color: C.accent1 });
  s.addShape(pres.shapes.RECTANGLE, { x: qx, y: 3.25, w: 3.6, h: 0.55, fill: { color: C.accent1 }, line: { type: "none" } });
  box(s, X0, 4.15, 5.6, 2.55, C.background2);
  T(s, [
    { text: "Queue imbalance", options: { bold: true, color: C.text2, fontSize: 16, breakLine: true } },
    { text: "I = (bid size − ask size) / (bid size + ask size)", options: { fontFace: "Cambria", fontSize: 16, breakLine: true } },
    { text: "  = (600 − 400) / 1000 = +0.2", options: { fontFace: "Cambria", fontSize: 16, breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "+1 means only buyers are waiting, −1 only sellers. A positive value means the ask is likely to run out first, so the price is more likely to tick up.", options: { fontSize: 14 } },
  ], { x: X0 + 0.25, y: 4.3, w: 5.1, h: 2.3 });
  card(s, 6.6, 1.35, 6.13, 1.6, "Why it works", "Price moves when one best queue is emptied. The smaller queue is more likely to empty first, so the side with less waiting tends to be the side the price moves through.", { icon: "eye" });
  card(s, 6.6, 3.1, 6.13, 1.6, "What the models add", "Imbalance at deeper levels, how it is changing (order flow), how one-sided recent trades were, and the last few seconds of history, all combined non-linearly.", { icon: "layers", iconFill: C.accent5 });
  card(s, 6.6, 4.85, 6.13, 1.85, "The catch", "The effect is real but small: a fraction of a basis point on average. The question is not only whether a model can learn it, but whether anyone could trade on it after costs.", { icon: "warn", iconFill: C.accent2 });

  // 9. Evidence BTC
  s = content("Evidence: the book does predict the next move (Bitcoin)",
    "Every one of the 1.03 million Bitcoin snapshots was sorted into ten groups by its queue imbalance, and we plotted the average price change over the next 20 seconds for each group. The relation rises steadily: when the book leans towards buyers, the price tends to rise. The right panel shows why this is hard: most 20-second moves are almost zero, with occasional large ones. The correlation between level-1 imbalance and the next 20-second move is 0.14.");
  fig(s, "btc_imbalance_signal", X0, 1.3, CW, 4.1);
  const ey = 5.55;
  card(s, X0, ey, 3.9, 1.2, "Monotone relation", "More bid-heavy book → higher average next move. Correlation 0.14.", { size: 13 });
  card(s, X0 + 4.12, ey, 3.9, 1.2, "Tiny effect", "A fraction of a basis point per decile; the mean |20 s move| is 3.3 bp.", { size: 13 });
  card(s, X0 + 8.24, ey, 3.89, 1.2, "Peaked, heavy tails", "59% of 1 s moves are exactly zero, so labels need a threshold.", { size: 13 });

  // 10. Evidence Nasdaq
  s = content("The signal is much stronger on large-tick stocks",
    "The same analysis on Nasdaq data from LOBSTER, using a 100-millisecond clock and a 5-second horizon. On Intel, a large-tick stock, the next move rises steeply with queue imbalance (correlation 0.24). On Amazon, a small-tick stock, the relation is weak (correlation 0.04) because its queues are small and change constantly. This difference shows up again in every model result.");
  fig(s, "lobster_imbalance_signal", X0, 1.3, 8.0, 4.6);
  stat(s, 9.0, 1.5, 3.7, "0.24", "Correlation of imbalance with the next 5 s move on INTC (large tick)", C.accent1);
  stat(s, 9.0, 3.25, 3.7, "0.04", "Same correlation on AMZN (small tick)", C.accent2);
  plain(s, 9.0, 4.95, 3.73, 1.8, "Long queues (large ticks) make the book easy to read. Short, busy queues (small ticks) make it noisy.");

  // 11. Research questions
  s = content("Three questions the experiments answer",
    "Published deep-learning papers report over 80% accuracy on order-book data. Those numbers raise three questions. Is it real or is it leakage? Which models and why? And does accuracy become profit? Each later section maps to one of these.");
  const qs = [
    ["shield", "Is the predictability real?", "High-frequency data make it easy to leak the future into training (overlapping labels, full-sample normalising, shuffled cross-validation). Accuracy is also misleading when most moves are flat. Which models beat simple baselines on held-out future data, by a statistically significant margin?", C.accent1],
    ["brain", "Which models, and why?", "Do deep networks that read the raw book (DeepLOB, a transformer) beat gradient-boosted trees on hand-made features? Which parts of the book carry the signal, and for how long does it last?", C.accent5],
    ["dollar", "Does accuracy become profit?", "A prediction only pays if the system can act before it expires and after paying the spread and fees. How do accuracy, inference speed and trading cost combine?", C.accent2],
  ];
  qs.forEach((q, i) => {
    const x = X0 + i * 4.14;
    box(s, x, 1.4, 3.85, 5.25);
    icon(s, q[0], x + 0.25, 1.65, 0.8, q[3]);
    T(s, q[1], { x: x + 0.25, y: 2.65, w: 3.4, h: 0.85, fontSize: 18, bold: true, color: C.text2 });
    T(s, q[2], { x: x + 0.25, y: 3.55, w: 3.4, h: 3.0, fontSize: 14 });
  });

  // =====================================================================
  section("02", "The data", "Six real datasets plus a synthetic order book: where they came from, what they look like, and how they were cleaned, labelled and split without leaking the future.");

  // 12. Datasets
  s = content("Six real datasets and one synthetic book",
    "Each dataset has a job. Bitcoin is the main real testbed. Ethereum provides cross-asset features. LOBSTER gives every order event on five Nasdaq stocks, which lets us test sub-second horizons and compare large- and small-tick stocks. FI-2010 is the standard academic benchmark used by the DeepLOB paper. Two public datasets test regression problems (volatility and closing auctions). The synthetic book has a planted signal, so we know the right answer.");
  table(s, [
    ["Dataset", "What it contains", "Size used", "Role in the project"],
    ["Coinbase BTC-USD", "15-level book every second, trade volumes, per-level add / trade / cancel flows (public CC0 dataset)", "1.03 M snapshots, 7–19 Apr 2021, 4 segments of ≈3 days", "Main real testbed"],
    ["Coinbase ETH-USD", "Same format and dates", "1.03 M snapshots", "Cross-asset features"],
    ["LOBSTER (Nasdaq)", "Every order event and the 10-level book after it: INTC, MSFT, AAPL, AMZN, GOOG", "1 full day per stock, resampled to 100 ms (≈234 k rows each)", "Sub-second horizons; tick size; market making"],
    ["FI-2010", "5 Finnish stocks, 10 days in 2010, normalised, 144 features + labels", "Train days 1–7, test days 8–10", "Compare with published DeepLOB"],
    ["Realised volatility", "Two-level book and trades, 10-min windows, ≈112 stocks", "Full set, 4 forward folds", "Regression; leakage case study"],
    ["Closing auction", "Nasdaq closing-auction imbalance, 200 stocks", "481 days, 5.2 M rows", "Regression under drift"],
    ["Synthetic", "Simulated 10-level book with a planted “informed trader”", "5 sessions × 60,000 events", "Controlled tests; latency sweep"],
  ], { x: X0, y: 1.3, w: CW, colW: [2.15, 4.85, 2.95, 2.18], fontSize: 12, rowH: 0.6, hl: [1] });

  // 13. Collection + compute
  s = content("How the data was collected, and where the computing ran",
    "Crypto and the public regression datasets were downloaded by a script using an API with credentials read from the environment. LOBSTER's free academic samples were downloaded by hand; they cannot be redistributed, so they are not in the repository. The synthetic book is generated on the fly. Tabular models and simulations ran on a laptop CPU; all deep-learning training ran on NVIDIA A100 GPUs on Google Colab, with checkpoints synced through git so interrupted runs could resume.");
  const coll = [
    ["cloud", "Crypto (BTC, ETH)", "Downloaded programmatically by scripts/fetch_datasets.py; credentials read from the environment or Colab secrets. 1.8 GB of raw CSV per coin.", C.accent1],
    ["uni", "LOBSTER (Nasdaq)", "Free academic sample files (message file + book file per stock) from lobsterdata.com, downloaded manually. Not redistributable.", C.accent5],
    ["book", "FI-2010 and regression sets", "Downloaded with the same fetcher script; the public datasets require accepting their data rules once.", C.accent6],
    ["random", "Synthetic book", "Generated by src/data/synthetic_generator.py: $100 price, 1¢ tick, event every ≈10 ms. An “informed trader” switches on at random and pushes 85% of market orders one way.", C.accent3],
  ];
  coll.forEach((c, i) => {
    const y = 1.35 + i * 1.33;
    icon(s, c[0], X0, y + 0.1, 0.7, c[3]);
    T(s, c[1], { x: X0 + 0.95, y, w: 5.6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
    T(s, c[2], { x: X0 + 0.95, y: y + 0.42, w: 5.6, h: 0.85, fontSize: 13 });
  });
  box(s, 7.45, 1.35, 5.28, 5.3, C.text2, "compute card");
  icon(s, "server", 7.75, 1.6, 0.75, C.accent3);
  T(s, "Compute", { x: 8.7, y: 1.6, w: 3.8, h: 0.75, fontSize: 22, bold: true, color: C.background1, valign: "middle" });
  T(s, [
    { text: "Deep learning on Google Colab A100 GPUs", options: { bold: true, color: "F2C46D", breakLine: true } },
    { text: "LSTM, DeepLOB and TLOB were trained with PyTorch + CUDA on NVIDIA A100 GPUs in Google Colab. A full Bitcoin run of the three deep models took a few hours; DeepLOB alone ≈18 minutes per segment.", options: { breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "Everything else on a laptop CPU", options: { bold: true, color: "F2C46D", breakLine: true } },
    { text: "Feature extraction, tabular models and simulations: 12 cores, 14 GB RAM.", options: { breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "Resumable runs", options: { bold: true, color: "F2C46D", breakLine: true } },
    { text: "Long experiments checkpoint after each segment and sync through a git branch, so a run cut off on one machine continues on the other." },
  ], { x: 7.75, y: 2.6, w: 4.7, h: 3.95, fontSize: 14, color: C.background1 });

  // 14. EDA
  s = content("What the data looks like (exploratory analysis)",
    "Three findings from exploration shaped the method. Most moves are tiny, so labels need a threshold and plain accuracy is misleading. The book is informative, much more so for large ticks. And the data is non-stationary: volatility and volume differ between days, which is why every Bitcoin experiment is repeated on four separate segments.");
  table(s, [
    ["", "Bitcoin (BTC-USD)", "INTC, MSFT (large tick)", "AAPL, AMZN, GOOG (small tick)"],
    ["Clock", "1 s", "100 ms", "100 ms"],
    ["Tick as a share of price", "0.0017 bp", "3.3–3.7 bp", "0.17–0.45 bp"],
    ["Spread at exactly one tick", "80% of the time", "77%", "under 1%"],
    ["Typical size at the best price", "a few BTC", "≈13,000 shares", "≈100 shares"],
    ["Mid-price changes per minute", "–", "8–10", "60–165"],
    ["Mean |20 s move|", "3.3 bp", "–", "–"],
    ["Correlation (L1 imbalance, next move)", "0.14 (20 s)", "0.24 (INTC, 5 s)", "0.04 (AMZN, 5 s)"],
  ], { x: X0, y: 1.3, w: 7.6, colW: [2.65, 1.65, 1.65, 1.65], fontSize: 12, rowH: 0.5 });
  card(s, 8.5, 1.3, 4.23, 1.65, "1. Most moves are tiny", "59% of 1-second Bitcoin mid changes are exactly zero. Labels need a threshold; accuracy alone misleads.", { size: 13 });
  card(s, 8.5, 3.1, 4.23, 1.65, "2. Large ticks are easier", "Long, slow queues make imbalance informative (0.24 vs 0.04 correlation).", { size: 13 });
  card(s, 8.5, 4.9, 4.23, 1.75, "3. Markets change day to day", "Volatility and volume shift between days, so Bitcoin is split into 4 segments, each a full repeat of the experiment.", { size: 13 });

  // 15. Preprocessing flow
  s = content("Preprocessing: from raw files to training rows",
    "Every dataset passes through the same five steps so that one feature pipeline and one set of models work on all of them. The conversion details differ by source. Bitcoin stores each level as a distance from the mid and a dollar amount, so prices are rebuilt and rounded to the 1-cent tick. LOBSTER stores prices as dollars times 10,000 and is resampled onto a 100 ms clock. FI-2010 stores features as rows, so it is transposed.");
  const steps = [
    ["1", "Convert", "Every source → one 40-column book format + timestamps + a trade tape"],
    ["2", "Clean", "Fix units, rebuild prices, mark empty levels, resample LOBSTER to 100 ms, NaN/∞ → 0"],
    ["3", "Segment", "Bitcoin cut into 4 blocks of 257,682 rows (≈3 days), each a full repeat"],
    ["4", "Label", "UP / FLAT / DOWN from the future mid, threshold set on training rows only"],
    ["5", "Split", "Chronological 70 / 15 / 15 with purged gaps; transforms fitted on train only"],
  ];
  const stw = 2.2, stg = 0.28;
  steps.forEach((st, i) => {
    const x = X0 + i * (stw + stg);
    box(s, x, 1.45, stw, 2.75);
    s.addShape(pres.shapes.OVAL, { x: x + stw / 2 - 0.32, y: 1.6, w: 0.64, h: 0.64, fill: { color: C.accent1 }, line: { type: "none" } });
    T(s, st[0], { x: x + stw / 2 - 0.32, y: 1.6, w: 0.64, h: 0.64, fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle" });
    T(s, st[1], { x: x + 0.12, y: 2.35, w: stw - 0.24, h: 0.4, fontSize: 17, bold: true, color: C.text2, align: "center" });
    T(s, st[2], { x: x + 0.15, y: 2.8, w: stw - 0.3, h: 1.35, fontSize: 13 });
    if (i < 4) arrowR(s, x + stw + 0.01, 2.65, stg - 0.02);
  });
  card(s, X0, 4.45, 3.9, 2.25, "Bitcoin / Ethereum", "Prices rebuilt as mid × (1 + distance) and rounded to 1¢; size = dollar amount ÷ price. First 10 of 15 levels kept. Cached as a 370 MB NumPy file so later runs load in seconds.", { size: 13 });
  card(s, X0 + 4.12, 4.45, 3.9, 2.25, "LOBSTER", "Each 100 ms row is the book as it stood at that instant, so “5 s ahead” is always 50 rows. Trades between rows become the trade tape; adds, trades and cancels per level become flow features.", { size: 13 });
  card(s, X0 + 8.24, 4.45, 3.89, 2.25, "FI-2010", "Files store time as columns, so they are transposed. The five label rows (horizons 10, 20, 30, 50, 100 events) are separated from the 144 features.", { size: 13 });

  // 16. Labels & class balance
  s = content("Labels: UP, FLAT or DOWN, and why accuracy misleads",
    "The label compares a smoothed future mid price with today's mid. If it rose by more than a threshold theta it is UP, if it fell by more than theta it is DOWN, otherwise FLAT. On Bitcoin, theta is chosen on training rows only so that one third of labels are FLAT, which makes random guessing score 0.333. At short horizons on large-tick stocks almost every label is FLAT, so a model that always says FLAT looks extremely accurate and is useless. That is why macro-F1, which weights all three classes equally, is the main metric, and why all models use balanced class weights.");
  box(s, X0, 1.35, 5.7, 2.35);
  T(s, [
    { text: "Future return  r = (smoothed mid in k rows − mid now) / mid now", options: { fontFace: "Cambria", breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "UP ", options: { bold: true, color: C.accent1 } }, { text: "if r > θ     ", options: {} },
    { text: "DOWN ", options: { bold: true, color: C.accent2 } }, { text: "if r < −θ     ", options: {} },
    { text: "FLAT ", options: { bold: true, color: C.accent4 } }, { text: "otherwise", options: { breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Bitcoin: θ set on training rows so ⅓ of labels are FLAT (random = 0.333). Synthetic: θ = 1.5 bp. FI-2010: its published labels.", options: { fontSize: 13, color: C.accent4 } },
  ], { x: X0 + 0.25, y: 1.5, w: 5.25, h: 2.1, fontSize: 15 });
  plain(s, X0, 3.9, 5.7, 2.8, "On INTC at 100 ms, 99% of labels are FLAT. A “model” that always answers FLAT is 99% accurate and completely useless. Macro-F1 averages the score of each class, so ignoring UP and DOWN is punished.", "Why not just accuracy?");
  T(s, "Share of FLAT labels in the test period", { x: 6.65, y: 1.35, w: 6.0, h: 0.35, fontSize: 14, bold: true, color: C.text2 });
  table(s, [
    ["Nasdaq horizon", "100 ms", "500 ms", "1 s", "5 s", "10 s", "30 s"],
    ["INTC (large tick)", "99%", "97%", "95%", "80%", "66%", "42%"],
    ["MSFT (large tick)", "99%", "96%", "93%", "74%", "60%", "37%"],
    ["AMZN (small tick)", "93%", "75%", "61%", "20%", "19%", "29%"],
  ], { x: 6.65, y: 1.8, w: 6.08, colW: [1.88, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7], fontSize: 12, rowH: 0.42, align: "center" });
  table(s, [
    ["Event / second horizon", "k = 10", "k = 20", "k = 50", "k = 100"],
    ["Synthetic (events)", "88%", "70%", "39%", "22%"],
    ["Bitcoin (seconds)", "29%", "30%", "31%", "31%"],
    ["FI-2010 (events)", "71%", "62%", "47%", "34%"],
  ], { x: 6.65, y: 3.75, w: 6.08, colW: [2.28, 0.95, 0.95, 0.95, 0.95], fontSize: 12, rowH: 0.42, align: "center" });
  T(s, "FI-2010 training share is lower (60% and 49% at k = 10, 20): the test days are calmer, which matters later.", { x: 6.65, y: 5.6, w: 6.08, h: 0.7, fontSize: 12, color: C.accent4, italic: true });

  // 17. Purged split (drawn)
  s = content("Keeping the future out: chronological, purged splits",
    "Shuffling rows randomly would put rows from the same minute in both training and test sets. Neighbouring rows are almost identical, so the model would be tested on data it has effectively seen. All splits are therefore in time order: 70% train, 15% validation, 15% test. Because each label looks k rows into the future, the last k rows before each boundary would overlap the next split, so they are dropped (purged). Every fitted quantity is estimated on training rows only. The relative-spread leak on the right was caught this way.");
  const sy = 1.75, sh = 0.75, sx0 = X0 + 0.1, sW = 11.9;
  const seg = (a, b, col) => s.addShape(pres.shapes.RECTANGLE, { x: sx0 + a * sW, y: sy, w: (b - a) * sW, h: sh, fill: { color: col }, line: { type: "none" } });
  seg(0, 0.70, HX.accent1); seg(0.70, 0.85, HX.accent3); seg(0.85, 1.0, HX.accent2);
  seg(0.665, 0.70, "2B3640"); seg(0.815, 0.85, "2B3640"); seg(0.965, 1.0, "2B3640");
  T(s, "Train (70%)", { x: sx0, y: sy, w: 0.66 * sW, h: sh, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle" });
  T(s, "Validation", { x: sx0 + 0.70 * sW, y: sy, w: 0.115 * sW, h: sh, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle" });
  T(s, "Test", { x: sx0 + 0.85 * sW, y: sy, w: 0.115 * sW, h: sh, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle" });
  T(s, "dark gaps = purged k rows", { x: sx0 + 0.52 * sW, y: sy - 0.42, w: 3.0, h: 0.35, fontSize: 12, color: C.text1, italic: true });
  s.addShape(pres.shapes.LINE, { x: sx0, y: sy + sh + 0.2, w: sW, h: 0, line: { color: "6B7A88", width: 1.5, endArrowType: "triangle" } });
  T(s, "time →", { x: sx0 + sW - 0.9, y: sy + sh + 0.25, w: 0.9, h: 0.3, fontSize: 12, color: C.accent4, align: "right" });
  card(s, X0, 3.25, 3.9, 3.4, "Fitted on train only", "Label threshold θ\nPCA loadings\nFeature scaler\nDeep-model normalising constants\nVPIN bucket size\n\nEvery model is scored on exactly the same test rows.", { icon: "check", size: 14 });
  card(s, X0 + 4.12, 3.25, 3.9, 3.4, "Purging", "Row t's label uses rows up to t + k. The last k rows before each boundary therefore “see” into the next split and are dropped (López de Prado).\n\nSequence models also get a 99-row warm-up so they see a full window.", { icon: "cut", iconFill: C.accent5, size: 14 });
  card(s, X0 + 8.24, 3.25, 3.89, 3.4, "A leak we caught", "An early feature, spread ÷ mid, took only two values with a fixed tick, so it really encoded the price level. LightGBM ranked it top for the wrong reason. The spread is now measured in whole ticks.", { icon: "warn", iconFill: C.accent2, size: 14 });

  // 18. Normalisation
  s = content("Normalising inputs for each kind of model",
    "Different models need different scaling. Logistic regression needs standardised features, trees need none. The deep models read raw book windows, so prices are turned into small relative distances and sizes are log-scaled, using constants from the first 100 training rows only. TLOB also has a learned bilinear normalisation layer, which was designed for non-stationary order-book data.");
  card(s, X0, 1.4, 3.9, 3.0, "Logistic regression", "Each feature standardised with the training mean and standard deviation.\n\nTree models (Random Forest, LightGBM) need no scaling: they only compare values.", { icon: "scale", size: 14 });
  card(s, X0 + 4.12, 1.4, 3.9, 3.0, "Deep models", "Price → (p − P₀) / P₀\nSize → log(1 + q / V₀)\n\nP₀ and V₀ come from the first 100 training rows, so prices become tiny relative distances and sizes sit around 1.", { icon: "brain", iconFill: C.accent2, size: 14 });
  card(s, X0 + 8.24, 1.4, 3.89, 3.0, "Inside TLOB: BiN", "Bilinear normalisation (Tran et al.) normalises across time and across features inside the network; a causal rolling z-score is also implemented.", { icon: "layers", iconFill: C.accent5, size: 14 });
  plain(s, X0, 4.7, CW, 1.95, "Numbers in an order book come in very different units: a price of $60,000 next to a size of 0.3 BTC. Normalising puts them on similar scales so a model can learn from both. The constants must come from training data only, otherwise the model would get a peek at the test period.");

  // =====================================================================
  section("03", "Turning the book into features", "The 27 hand-made signals that the tabular models read, what each one means, and how they relate to each other. The deep models skip this step and read the raw book.");

  // 19. Features explained
  s = content("Six families of features, in plain words",
    "All 27 features are causal: row t only uses rows up to t. Queue imbalance and the micro-price come from the best quotes. Order flow imbalance, from Cont, Kukanov and Stoikov, measures net buying pressure from one snapshot to the next, and its multi-level version is compressed with PCA fitted on training rows. VPIN, from Easley, Lopez de Prado and O'Hara, measures how one-sided recent trading has been. Book slopes measure how thin the book is. Lags add the last few seconds of history.");
  const feats = [
    ["scale", "Queue imbalance (4)", "Bid vs ask size at level 1, and over the first 3, 5 and 10 levels. Range −1 to +1.", C.accent1],
    ["target", "Micro-price deviation (1)", "A “fair price” that leans towards the side with less waiting: micro − mid = imbalance × spread / 2.", C.accent1],
    ["chart", "Order flow imbalance (9)", "Net new buying pressure since the last snapshot: bids added minus asks added (and cancellations). Per level, plus 3 PCA components.", C.accent5],
    ["random", "VPIN (1)", "How one-sided recent trades were, measured in equal-volume buckets. High = “toxic”, informed-looking flow.", C.accent3],
    ["ruler", "Book shape (3)", "Spread in whole ticks, and the slope of each side: price distance per unit of depth. Steep = thin book.", C.accent6],
    ["hourglass", "Lags (9)", "Level-1 OFI, imbalance and micro-price deviation 1, 2 and 5 rows earlier: the last few seconds of trend.", C.accent2],
  ];
  feats.forEach((f, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    card(s, X0 + col * 4.14, 1.35 + row * 2.68, 3.85, 2.45, f[1], f[2], { icon: f[0], iconFill: f[3], size: 14 });
  });

  // 20. Formulas
  s = content("The key formulas",
    "These are the exact definitions used. OFI compares the best bid and ask between two snapshots: if the bid price rose, all its size counts as new demand; if it stayed the same, only the change in size counts; if it fell, the old size counts as lost demand. The ask side is mirrored. PCA on the 10 per-level OFI values keeps 3 components, which hold about 60% of the variance on Bitcoin. VPIN uses 50 buckets of equal traded volume, and each snapshot gets the VPIN of the last completed bucket, so it never sees the future.");
  const fx = [
    ["Queue imbalance", "I = (q_bid − q_ask) / (q_bid + q_ask)", "Between −1 and +1; also over 3, 5, 10 levels."],
    ["Micro-price", "P_micro = (q_bid · p_ask + q_ask · p_bid) / (q_bid + q_ask)", "So P_micro − P_mid = I × spread / 2."],
    ["Order flow imbalance", "OFI = ΔW (bid demand change) − ΔV (ask supply change)", "Same formula at each of 10 levels; PCA keeps 3 components (36%, 15%, 9% of variance)."],
    ["VPIN", "VPIN = (1/50) Σ |buy volume − sell volume| / V", "50 equal-volume buckets; V = average volume per minute in training."],
    ["Book slope", "(p_bid,1 − p_bid,10) / Σ bid sizes", "And the mirror for the ask side."],
  ];
  fx.forEach((f, i) => {
    const y = 1.35 + i * 1.07;
    box(s, X0, y, CW, 0.92);
    T(s, f[0], { x: X0 + 0.25, y, w: 2.6, h: 0.92, fontSize: 15, bold: true, color: C.text2, valign: "middle" });
    T(s, f[1], { x: X0 + 2.9, y, w: 5.6, h: 0.92, fontSize: 15, fontFace: "Cambria", valign: "middle" });
    T(s, f[2], { x: X0 + 8.6, y, w: 3.35, h: 0.92, fontSize: 12, color: C.accent4, valign: "middle" });
  });

  // 21. Correlation
  s = content("How the 27 features relate to each other",
    "The heat map shows correlations between all 27 features on training data. The imbalance features are strongly related to each other (0.79 to 0.93) and the first OFI principal component is close to level-1 OFI (0.91), but the median absolute correlation is only 0.06, so most features carry different information. Trees handle correlated groups well; for logistic regression standardisation and the L2 penalty handle them. One honest note: mlofi_lvl_1 equals ofi_l1, so 27 columns carry 26 distinct features.");
  fig(s, "feature_corr", X0, 1.25, 5.9, 5.65);
  card(s, 7.0, 1.35, 5.73, 1.55, "Correlated groups", "Imbalance at 1/3/5/10 levels: 0.79–0.93. First OFI PCA component vs level-1 OFI: 0.91.", { size: 14 });
  card(s, 7.0, 3.05, 5.73, 1.35, "Mostly different information", "Median absolute correlation across all pairs is just 0.06.", { size: 14 });
  card(s, 7.0, 4.55, 5.73, 2.1, "The deep models skip all this", "LSTM, DeepLOB and TLOB read a raw 100 × 40 window (the last 100 snapshots) and must learn their own features. Later slides show they end up looking at the same things.", { size: 14, icon: "brain", iconFill: C.accent2 });

  // =====================================================================
  section("04", "Models and validation", "What kinds of learning problem we solved, the ladder of models from naive guesses to deep networks, and how we made sure the test results are honest.");

  // 22. Tasks
  s = content("Four kinds of learning problem",
    "Classification predicts a category; regression predicts a number. The main task is three-class classification of price direction. We also regress the size of the future move, predict volatility and auction moves, and do binary classification of whether a market maker's fill will be toxic, meaning it loses money 30 seconds later.");
  const tasks = [
    ["tag", "3-class classification", "Will the mid price be UP, FLAT or DOWN k rows ahead?", "Synthetic, Bitcoin, LOBSTER, FI-2010", "Macro-F1, accuracy, ROC-AUC, log-loss, MCC", C.accent1],
    ["chart", "Regression: price change", "How many basis points will the mid move? (also: micro-price change)", "Bitcoin, LOBSTER", "Information coefficient (IC), R², hit rate", C.accent5],
    ["clock", "Regression: volatility, auction", "Next 10-minute volatility; 60 s closing-auction price move", "Volatility and auction data", "RMSPE, MAE", C.accent6],
    ["shield", "Binary classification", "Will a resting quote fill, and will the fill be “toxic”?", "Bitcoin, LOBSTER", "ROC-AUC, markout", C.accent2],
  ];
  tasks.forEach((t, i) => {
    const x = X0 + (i % 2) * 6.17, y = 1.35 + Math.floor(i / 2) * 2.7;
    box(s, x, y, 5.96, 2.5);
    icon(s, t[0], x + 0.22, y + 0.22, 0.6, t[5]);
    T(s, t[1], { x: x + 1.0, y: y + 0.22, w: 4.8, h: 0.6, fontSize: 17, bold: true, color: C.text2, valign: "middle" });
    T(s, [
      { text: "Target: ", options: { bold: true } }, { text: t[2], options: { breakLine: true } },
      { text: "Data: ", options: { bold: true } }, { text: t[3], options: { breakLine: true } },
      { text: "Metrics: ", options: { bold: true } }, { text: t[4] },
    ], { x: x + 0.22, y: y + 0.95, w: 5.5, h: 1.45, fontSize: 14, paraSpaceAfter: 4 });
  });

  // 23. Model ladder
  s = content("The model ladder: from no learning to deep learning",
    "Models form a ladder of increasing complexity. A model only counts as good if it beats the simpler ones on exactly the same test rows. Tier 0 uses no features at all: momentum just repeats the last label we could have known. Logistic regression is linear. Random Forest and LightGBM are tree ensembles on the 27 features. The LSTM, DeepLOB and TLOB read raw 100 by 40 windows of the book.");
  const tiers = [
    ["0", "Naive baselines", "Momentum repeats the last known label; flat always says FLAT", "past labels", C.accent4],
    ["1", "Logistic regression", "Linear model, softmax over 3 classes", "27 features", C.accent1],
    ["2", "Random Forest", "100 trees, depth 10, voting", "27 features", C.accent1],
    ["3", "LightGBM", "Gradient-boosted trees, ≤ 200 rounds", "27 features", C.accent1],
    ["4", "LSTM", "Recurrent network, 62,675 parameters", "100 × 40 window", C.accent2],
    ["5", "DeepLOB · TLOB", "CNN + Inception + LSTM (213,987) · transformer (88,795)", "100 × 40 window", C.accent2],
  ];
  const tw0 = 1.92, tg = 0.12;
  tiers.forEach((t, i) => {
    const x = X0 + i * (tw0 + tg), hgt = 2.35 + i * 0.48, y = 6.7 - hgt;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: tw0, h: hgt, rectRadius: 0.08, fill: { color: t[4] }, line: { type: "none" } });
    T(s, "Tier " + t[0], { x: x + 0.12, y: y + 0.12, w: tw0 - 0.24, h: 0.3, fontSize: 12, bold: true, color: "E8EEF2" });
    T(s, t[1], { x: x + 0.12, y: y + 0.42, w: tw0 - 0.24, h: 0.65, fontSize: 15, bold: true, color: C.background1 });
    T(s, t[2], { x: x + 0.12, y: y + 1.07, w: tw0 - 0.24, h: hgt - 1.55, fontSize: 12, color: C.background1 });
    T(s, t[3], { x: x + 0.12, y: y + hgt - 0.45, w: tw0 - 0.24, h: 0.35, fontSize: 11, italic: true, color: "E8EEF2" });
  });
  T(s, [
    { text: "Grey", options: { bold: true, color: C.accent4 } }, { text: " = no learning   ", options: {} },
    { text: "Teal", options: { bold: true, color: C.accent1 } }, { text: " = tabular ML on hand-made features   ", options: {} },
    { text: "Coral", options: { bold: true, color: C.accent2 } }, { text: " = deep learning on the raw book (trained on Colab A100 GPUs)", options: {} },
  ], { x: X0, y: 1.35, w: CW, h: 0.4, fontSize: 14 });
  T(s, "Every learned model weights each class by n / (3 nᶜ), so rare UP / DOWN mistakes cost as much as FLAT ones.", { x: X0, y: 1.8, w: 6.5, h: 0.65, fontSize: 13, color: C.accent4, italic: true });

  // 24. Tabular models explained
  s = content("How the tabular models work",
    "Logistic regression computes a weighted sum of the features for each class and turns the three scores into probabilities with the softmax function; it can only draw straight boundaries, so it measures how much of the signal is linear. A Random Forest grows many decision trees on random samples of rows and features and lets them vote; the randomness makes their errors cancel out. LightGBM adds trees one at a time, each one correcting the errors of the current ensemble. Early stopping on validation loss chose 109 rounds on Bitcoin, which is 327 trees with 20,601 leaves in total.");
  card(s, X0, 1.35, 3.9, 3.6, "Logistic regression", "A weighted sum of features for each class, turned into probabilities by the softmax.\n\nCan only draw straight-line boundaries, so it tells us how much of the signal is linear.\n\nL-BFGS, C = 1, L2 penalty.", { icon: "scale", size: 14 });
  card(s, X0 + 4.12, 1.35, 3.9, 3.6, "Random Forest", "Many decision trees, each grown on a random sample of rows and choosing among random features at each split. The trees vote.\n\nRandomness decorrelates errors and reduces variance.\n\n100 trees, depth 10.", { icon: "tree", iconFill: C.accent6, size: 14 });
  card(s, X0 + 8.24, 1.35, 3.89, 3.6, "LightGBM (gradient boosting)", "Trees are added one at a time; each new tree fixes the mistakes of the ones before.\n\nF(x) = Σ η·fₘ(x), learning rate η = 0.05\n\n63 leaves, depth 8, early stopping after 15 rounds.", { icon: "bolt", iconFill: C.accent3, size: 14 });
  plain(s, X0, 5.15, CW, 1.55, "A decision tree asks a chain of yes/no questions (“Is the imbalance above 0.3? Was OFI positive a second ago?”) and gives an answer at the end. One tree is weak; a forest averages many independent trees; boosting builds a team of trees where each new one specialises in the cases the team still gets wrong.");

  // 25. Deep models + DeepLOB diagram
  s = content("How the deep models work",
    "DeepLOB treats the 100 by 40 window like an image. The first convolution combines each price with its size, the second combines bid with ask at each level, where something like a micro-price can form, and the third combines all levels. Inception modules then look at several time scales in parallel, and an LSTM tracks how things evolve. The LSTM alone reads the window one row at a time with a gated memory. TLOB is a transformer: self-attention lets every time step weigh every other one. Our TLOB attends over time only, so it is a simplified version of the published model. All three were trained with weighted cross-entropy and AdamW on Colab A100 GPUs, with early stopping.");
  const dl = [
    ["Input", "100 × 40 book window", C.accent4],
    ["Conv 1×2", "price + size → 100 × 20", C.accent2],
    ["Conv 1×2", "bid + ask → 100 × 10", C.accent2],
    ["Conv 1×10", "all levels → 94 × 1, 32 ch.", C.accent2],
    ["2 × Inception", "3×1, 5×1, max-pool, 192 ch.", C.accent5],
    ["LSTM", "64 units, last step", C.accent1],
    ["Output", "softmax: U / F / D", C.text2],
  ];
  T(s, "DeepLOB as implemented (213,987 parameters)", { x: X0, y: 1.3, w: CW, h: 0.35, fontSize: 15, bold: true, color: C.text2 });
  const dw = 1.5, dg = 0.27;
  dl.forEach((d, i) => {
    const x = X0 + i * (dw + dg);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.75, w: dw, h: 1.45, rectRadius: 0.08, fill: { color: d[2] }, line: { type: "none" } });
    T(s, d[0], { x: x + 0.08, y: 1.83, w: dw - 0.16, h: 0.42, fontSize: 14, bold: true, color: C.background1, align: "center" });
    T(s, d[1], { x: x + 0.08, y: 2.27, w: dw - 0.16, h: 0.85, fontSize: 12, color: C.background1, align: "center" });
    if (i < dl.length - 1) arrowR(s, x + dw + 0.02, 2.32, dg - 0.04);
  });
  card(s, X0, 3.4, 3.9, 1.95, "LSTM", "Reads the window one row at a time, keeping a gated memory; the last state is classified. 2 layers × 64 units.", { size: 14 });
  card(s, X0 + 4.12, 3.4, 3.9, 1.95, "TLOB (transformer)", "BiN normalisation, then self-attention: every time step can weigh every other. 2 blocks, 4 heads. Simplified: attends over time only.", { size: 14 });
  card(s, X0 + 8.24, 3.4, 3.89, 1.95, "Training recipe", "Weighted cross-entropy, AdamW, lr 5×10⁻⁴, batch 128, ≤ 8–15 epochs, early stopping. PyTorch + CUDA on Colab A100.", { size: 14, icon: "server", iconFill: C.accent3 });
  plain(s, X0, 5.5, CW, 1.2, "Instead of being told which signals matter, deep models look at the raw book and learn their own. That needs far more data and compute, which is why they were trained on cloud GPUs.");

  // 26. Pipeline
  s = content("How it all fits together",
    "One pipeline serves every dataset. Raw data is converted to the common book format, labelled and split. Tabular models read the 27 features; deep models read raw windows. Predictions go to the evaluation stage for classification metrics, and to the execution simulator, which turns them into trades against the recorded book with spread, fees and latency. The trained LightGBM is also exported to C++ to measure how fast a production version would be.");
  const pipe = [
    ["Raw data", "Bitcoin, Ethereum, LOBSTER, FI-2010, synthetic", C.accent4],
    ["Common format", "40-column book, timestamps, trade tape", C.accent4],
    ["Labels + split", "Train-only θ, purged 70/15/15", C.accent4],
    ["Features / windows", "27 causal features, or raw 100×40 windows", C.accent1],
    ["Models", "Tabular (CPU) and deep (A100 GPU)", C.accent2],
  ];
  const pw0 = 2.2, pg = 0.28;
  pipe.forEach((p, i) => {
    const x = X0 + i * (pw0 + pg);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.5, w: pw0, h: 1.5, rectRadius: 0.08, fill: { color: p[2] }, line: { type: "none" } });
    T(s, p[0], { x: x + 0.1, y: 1.6, w: pw0 - 0.2, h: 0.4, fontSize: 15, bold: true, color: C.background1, align: "center" });
    T(s, p[1], { x: x + 0.1, y: 2.05, w: pw0 - 0.2, h: 0.85, fontSize: 12, color: C.background1, align: "center" });
    if (i < pipe.length - 1) arrowR(s, x + pw0 + 0.01, 2.09, pg - 0.02);
  });
  s.addShape(pres.shapes.DOWN_ARROW, { x: X0 + 4 * (pw0 + pg) + pw0 / 2 - 0.2, y: 3.08, w: 0.4, h: 0.45, fill: { color: "9FB0BE" }, line: { type: "none" } });
  const outs = [
    ["check", "Evaluation", "Macro-F1, AUC, log-loss, MCC, IC; block bootstrap, McNemar; SHAP, ablation", C.accent6],
    ["dollar", "Execution simulator", "Spread crossing, walking the book, fees, latency, passive fills", C.accent2],
    ["code", "C++ export", "Trained LightGBM compiled to plain C++ to measure real inference speed", C.accent5],
  ];
  outs.forEach((o, i) => {
    card(s, X0 + i * 4.14, 3.65, 3.85, 1.95, o[1], o[2], { icon: o[0], iconFill: o[3], size: 14 });
  });
  box(s, X0, 5.8, CW, 0.85, C.text2);
  T(s, "Final outputs: net P&L, edge decomposition, break-even accuracy, and dollars lost per millisecond of delay", { x: X0 + 0.3, y: 5.8, w: CW - 0.6, h: 0.85, fontSize: 16, bold: true, color: C.background1, valign: "middle", align: "center" });

  // 27. Validation
  s = content("Making sure the results are honest",
    "Financial ML is notorious for results that disappear in real use, usually because of leakage. Six safeguards were used. Note the block bootstrap: neighbouring rows are not independent, so ordinary confidence intervals would be too narrow; resampling whole 5-minute blocks keeps the dependence. McNemar's test looks only at rows where exactly one of two models is right. Negative controls train on shuffled targets and must score at chance, and they do.");
  const val = [
    ["road", "Chronological, purged splits", "Train → validation → test in time order, with gaps; every transform fitted on train only."],
    ["target", "Identical test rows", "Deep-model windows end exactly on the tabular test rows, so all models answer the same questions."],
    ["layers", "Purged K-fold with embargo", "5 folds with a 1% embargo, used in the robustness report."],
    ["random", "Negative controls", "Shuffled-target models must score at chance: IC −0.0005 over 37 cases; MCC −0.026."],
    ["flask", "Block bootstrap + McNemar", "Resample 5-minute blocks 1,000 times for confidence intervals; McNemar compares rows only one model gets right."],
    ["check", "40 automated tests", "Feature formulas, causality (changing the future must not change past features), purge gap, simulator fills, planted-signal end-to-end."],
  ];
  val.forEach((v, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = X0 + col * 6.17, y = 1.35 + row * 1.8;
    icon(s, v[0], x, y + 0.1, 0.65, i < 2 ? C.accent1 : i < 4 ? C.accent5 : C.accent6);
    T(s, v[1], { x: x + 0.85, y, w: 5.1, h: 0.45, fontSize: 16, bold: true, color: C.text2 });
    T(s, v[2], { x: x + 0.85, y: y + 0.47, w: 5.1, h: 1.15, fontSize: 14 });
  });

  // 28. Metrics
  s = content("The scorecard: what each metric tells you",
    "Accuracy is misleading when FLAT dominates, so the main metric is macro-F1, the average F1 score across the three classes. ROC-AUC asks whether the model ranks a true UP above a non-UP, independently of any threshold. Log-loss and Brier score judge the probabilities themselves. MCC and Cohen's kappa correct for chance. IC and R-squared are for regression. The last three rows are economic: hit rate, net profit and latency.");
  table(s, [
    ["Metric", "What it means in plain words", "Reference value"],
    ["Accuracy", "Share of rows predicted correctly", "Misleading when FLAT dominates"],
    [{ text: "Macro-F1 (main)", options: { bold: true } }, "Average over the 3 classes of F1 (balance of precision and recall); each class counts equally", "Random = 0.333"],
    ["Balanced accuracy", "Average recall per class", "Random = 0.333"],
    ["ROC-AUC", "Chance a random true case is scored above a random false one", "0.5 = no skill, 1 = perfect"],
    ["Log-loss · Brier", "How good the predicted probabilities are (lower is better)", "Uniform guess 1.099 · 0.667"],
    ["MCC · Cohen's κ", "Agreement between prediction and truth, corrected for chance", "0 = chance"],
    ["IC · R²", "Correlation of forecast with the real move · share of variance explained", "0 = no skill; tiny R² is normal"],
    ["RMSPE · MAE", "Percentage and absolute error of a regression", "Lower is better"],
    ["Hit rate", "Share of directional calls with the right sign", "Break-even depends on cost"],
    ["Net P&L, edge (bp)", "Profit after spread, walking the book and fees", "Must exceed costs"],
    ["Latency P50 / P99", "Median and 99th-percentile time for one prediction", "Must be well inside the horizon"],
  ], { x: X0, y: 1.3, w: CW, colW: [2.6, 6.3, 3.23], fontSize: 12, rowH: 0.44 });

  // =====================================================================
  section("05", "Results: how predictable is the book?", "Model comparison on Bitcoin and the synthetic book, probability metrics, significance tests, how fast the signal fades, the FI-2010 benchmark, and what the models actually look at.");

  // 29. BTC model comparison native chart
  s = content("Bitcoin, 20 s ahead: tree models are best",
    "Macro-F1 on Bitcoin at a 20-second horizon, averaged over four test segments. Random guessing scores 0.333; naive momentum 0.413. LightGBM is best at 0.467, with Random Forest almost identical. The deep sequence models do worse here: DeepLOB and the LSTM trail even momentum, and TLOB is level with logistic regression. With at most 8 epochs they may be under-trained, but they also see only one-second snapshots, where hand-made flow features summarise the useful history compactly.");
  const mlabels = ["Naive flat", "Random guess", "LSTM", "DeepLOB", "Naive momentum", "Logistic regression", "TLOB", "Random Forest", "LightGBM"];
  const mvals = [0.152, 0.333, 0.397, 0.401, 0.413, 0.448, 0.449, 0.465, 0.467];
  const mcols = [HX.accent4, "B8C3CC", HX.accent2, HX.accent2, HX.accent4, HX.accent1, HX.accent2, HX.accent1, HX.accent3];
  s.addChart(pres.charts.BAR, [{ name: "Macro-F1", labels: mlabels, values: mvals }], Object.assign({
    x: X0, y: 1.3, w: 7.4, h: 5.45, barDir: "bar", chartColors: mcols, valAxisMinVal: 0, valAxisMaxVal: 0.5, valAxisMajorUnit: 0.1,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valGridLine: { color: "E1E7EC", size: 0.75 }, catGridLine: { style: "none" },
    showLegend: false, showTitle: true, title: "Test macro-F1, BTC-USD, 20 s horizon (mean of 4 segments)", barGapWidthPct: 45, valAxisLabelFormatCode: "0.0",
  }, chartText));
  T(s, [
    { text: "Teal", options: { bold: true, color: C.accent1 } }, { text: " tabular ML   ", options: {} },
    { text: "Coral", options: { bold: true, color: C.accent2 } }, { text: " deep learning   ", options: {} },
    { text: "Grey", options: { bold: true, color: C.accent4 } }, { text: " baselines", options: {} },
  ], { x: 8.3, y: 1.35, w: 4.4, h: 0.4, fontSize: 13 });
  bullets(s, [
    ["Real but small signal.", "Best models add ≈0.05 macro-F1 over momentum and 0.13 over random guessing."],
    ["Trees win.", "LightGBM 0.467 ± 0.021; Random Forest 0.465. Also best on the synthetic book (0.584)."],
    ["Deep models trail here.", "DeepLOB 0.401 and LSTM 0.397 are below momentum; TLOB 0.449 ≈ logistic."],
    ["Accuracy misleads.", "On synthetic data, “always FLAT” has the top accuracy (0.698) and the worst macro-F1 (0.274)."],
  ], { x: 8.3, y: 1.9, w: 4.43, h: 4.8, fontSize: 14, gap: 10 });

  // 30. Model ladder fig (synthetic + BTC)
  s = content("Same ranking on the synthetic book and on real Bitcoin",
    "Left: the synthetic book over 5 seeds, where we planted the signal so the right answer is known. Right: real Bitcoin over 4 segments. Error bars are one standard deviation. On both, gradient-boosted trees on microstructure features come out on top. The synthetic result confirms the pipeline can find a signal when one exists; the shuffled-target controls confirm it does not invent one.");
  fig(s, "model_ladder", X0, 1.3, CW, 4.3);
  plain(s, X0, 5.7, CW, 1.1, "Bars further right are better. The pattern holds on both the controlled synthetic book and the real market, so it is not a fluke of one dataset.");

  // 31. Full metrics table
  s = content("Every classification metric tells the same story",
    "To report probability-based measures, the tabular models were retrained on each Bitcoin segment with identical settings and their probabilities kept. The ranking is the same in every column: LightGBM, then Random Forest, then logistic regression, then momentum. AUC for UP vs DOWN on rows that actually moved is 0.717, much higher than the overall AUC: the models are better at telling the direction than at telling whether a move will be big enough to count.");
  table(s, [
    ["Model", "Acc.", "Bal. acc.", "Macro-F1", "MCC", "κ", "ROC-AUC", "AUC (UP vs DOWN)", "Log-loss", "Brier"],
    ["Naive momentum", "0.415", "0.413", "0.413", "0.118", "0.118", "0.559", "0.571", "1.440", "0.879"],
    ["Logistic regression", "0.463", "0.452", "0.448", "0.187", "0.186", "0.641", "0.710", "1.041", "0.626"],
    ["Random Forest", "0.474", "0.467", "0.465", "0.207", "0.206", "0.655", "0.717", "1.029", "0.619"],
    ["LightGBM", "0.476", "0.470", "0.467", "0.211", "0.209", "0.658", "0.717", "1.026", "0.617"],
    [{ text: "Uniform guess", options: { italic: true } }, "0.333", "0.333", "0.333", "0", "0", "0.500", "0.500", "1.099", "0.667"],
  ], { x: X0, y: 1.3, w: CW, colW: [2.45, 0.95, 1.0, 1.05, 0.9, 0.85, 1.05, 1.68, 1.1, 1.1], fontSize: 13, rowH: 0.48, hl: [4], align: "center" });
  T(s, "BTC-USD, 20 s horizon, mean over 4 segments. Higher is better except log-loss and Brier.", { x: X0, y: 4.25, w: CW, h: 0.3, fontSize: 12, italic: true, color: C.accent4 });
  stat(s, X0, 4.65, 2.8, "0.658", "LightGBM ROC-AUC (one-vs-rest)", C.accent1);
  stat(s, X0 + 3.05, 4.65, 2.8, "0.717", "AUC for UP vs DOWN on rows that moved", C.accent5);
  stat(s, X0 + 6.1, 4.65, 2.8, "1.026", "Log-loss vs 1.099 for a uniform guess", C.accent3);
  stat(s, X0 + 9.15, 4.65, 2.95, "0.211", "MCC: real but weak association (0 = chance)", C.accent2);

  // 32. ROC / calibration
  s = content("ROC curves and calibration: the probabilities can be trusted",
    "Left two panels: ROC curves for UP versus rest and DOWN versus rest. A curve above the diagonal means real skill; LightGBM and Random Forest are highest. Right: the reliability diagram. When LightGBM says it is 64% sure, it is right 63% of the time. Only the most confident predictions are slightly over-confident, 83% predicted versus 79% observed. This matters for trading, because confidence is what decides which signals to trade.");
  fig(s, "btc_roc_calibration", X0, 1.3, CW, 4.2);
  card(s, X0, 5.6, 3.9, 1.15, "Ranking skill", "Curves well above the diagonal; AUC 0.66 overall.", { size: 13 });
  card(s, X0 + 4.12, 5.6, 3.9, 1.15, "Well calibrated", "Says 64% → right 63% of the time.", { size: 13 });
  card(s, X0 + 8.24, 5.6, 3.89, 1.15, "Slight over-confidence at the top", "83% predicted vs 79% observed.", { size: 13 });

  // 33. Significance
  s = content("Is LightGBM's lead statistically real?",
    "Because rows close in time are correlated, we used a moving-block bootstrap that resamples whole 5-minute blocks 1,000 times. The 95% confidence interval for the macro-F1 difference excludes zero for every comparison, and McNemar's test agrees. The lead over Random Forest is tiny (0.004) but consistent in 3 of 4 segments. So the gain is statistically certain, though, as we will see, that does not make it economically useful.");
  table(s, [
    ["LightGBM vs", "Δ macro-F1", "95% block-bootstrap CI", "Segments better", "McNemar p"],
    ["Naive momentum", "+0.059", "[0.050, 0.068]", "4 / 4", "< 10⁻⁴"],
    ["Logistic regression", "+0.019", "[0.015, 0.024]", "4 / 4", "< 10⁻⁴"],
    ["Random Forest", "+0.004", "[0.001, 0.007]", "3 / 4", "0.015"],
    ["DeepLOB", "+0.058", "[0.049, 0.068]", "4 / 4", "< 10⁻⁴"],
  ], { x: X0, y: 1.35, w: 7.4, colW: [2.1, 1.25, 1.75, 1.2, 1.1], fontSize: 14, rowH: 0.55, align: "center" });
  plain(s, X0, 4.4, 7.4, 2.3, "If a confidence interval does not include zero, the improvement is very unlikely to be luck. All four intervals are above zero. A tiny p-value from McNemar's test says the same thing: on the rows where the two models disagree, LightGBM is right far more often.");
  card(s, 8.3, 1.35, 4.43, 2.55, "Block bootstrap", "Rows seconds apart are nearly identical, so treating them as independent overstates certainty. Resampling 300-row (5-minute) blocks within each segment keeps that dependence.", { icon: "layers", size: 14 });
  card(s, 8.3, 4.1, 4.43, 2.6, "McNemar's test", "Ignores rows both models get right or both get wrong; asks whether, when they disagree, one wins more often than chance would allow.", { icon: "scale", iconFill: C.accent5, size: 14 });

  // 34. Confusion
  s = content("Where the models make mistakes",
    "Row-normalised confusion matrices on Bitcoin at 20 seconds, pooled over segments. F is FLAT, U is UP, D is DOWN. Each row shows what the model predicted when the truth was that class. FLAT is the hardest class for every model. The tree models spread their mistakes evenly. DeepLOB leans one way: it calls 41% of FLAT rows DOWN and recalls DOWN (54%) much better than UP (42%).");
  fig(s, "btc_confusion", X0, 1.4, CW, 3.6);
  plain(s, X0, 5.2, 7.5, 1.5, "Read each row as “when the truth was X, how often did the model say F, U or D?”. A perfect model would have dark squares only on the diagonal.", "How to read it");
  card(s, 8.35, 5.2, 4.38, 1.5, "Finding", "FLAT is hardest for all. DeepLOB is biased towards DOWN (41% of FLAT rows called DOWN).", { size: 13 });

  // 35. Signal decay
  s = content("How fast does the signal fade?",
    "Left: Bitcoin skill against horizon. LightGBM macro-F1 is 0.540 at 10 s, 0.467 at 20 s, 0.395 at 50 s and 0.350 at 100 s, where it no longer beats momentum. Right: on Nasdaq, the best learned model's macro-F1 minus momentum. Large-tick stocks peak at 1 to 5 seconds; at 100 ms nothing is predictable because 99% of labels are FLAT. Small-tick stocks are hard to beat: their last move is a strong baseline. So useful information lives at roughly 1 to 10 seconds, which sets the latency budget for trading.");
  fig(s, "signal_decay", X0, 1.3, CW, 4.2);
  card(s, X0, 5.6, 3.9, 1.15, "Bitcoin", "0.540 at 10 s → 0.350 at 100 s (momentum 0.358).", { size: 13 });
  card(s, X0 + 4.12, 5.6, 3.9, 1.15, "Large-tick Nasdaq", "Peak at 1–5 s. INTC at 5 s: 0.457 vs momentum 0.344.", { size: 13 });
  card(s, X0 + 8.24, 5.6, 3.89, 1.15, "Small-tick Nasdaq", "Mostly lose to momentum; order-flow features add +0.011.", { size: 13 });

  // 36. AUC by horizon (native chart)
  s = content("Skill falls with horizon in every measure",
    "ROC-AUC by horizon for the three tabular models on Bitcoin. LightGBM goes from 0.732 at 10 seconds to 0.584 at 50 seconds. Macro-F1 falls in the same way. The ordering of models is stable at every horizon, with Random Forest and LightGBM almost tied.");
  s.addChart(pres.charts.BAR, [
    { name: "Logistic regression", labels: ["10 s", "20 s", "50 s"], values: [0.694, 0.641, 0.578] },
    { name: "Random Forest", labels: ["10 s", "20 s", "50 s"], values: [0.722, 0.655, 0.585] },
    { name: "LightGBM", labels: ["10 s", "20 s", "50 s"], values: [0.732, 0.658, 0.584] },
  ], Object.assign({
    x: X0, y: 1.3, w: 7.4, h: 5.45, barDir: "col", barGrouping: "clustered", chartColors: [HX.accent5, HX.accent6, HX.accent3],
    valAxisMinVal: 0.5, valAxisMaxVal: 0.76, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
    valGridLine: { color: "E1E7EC", size: 0.75 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", showTitle: true, title: "One-vs-rest ROC-AUC on BTC (0.5 = no skill)", barGapWidthPct: 60,
  }, chartText));
  table(s, [
    ["Horizon", "Logistic", "Random Forest", "LightGBM"],
    ["10 s", "0.500", "0.532", "0.540"],
    ["20 s", "0.448", "0.465", "0.467"],
    ["50 s", "0.382", "0.396", "0.395"],
  ], { x: 8.3, y: 1.75, w: 4.43, colW: [1.1, 1.05, 1.23, 1.05], fontSize: 13, rowH: 0.45, align: "center" });
  T(s, "Macro-F1 by horizon", { x: 8.3, y: 1.35, w: 4.43, h: 0.35, fontSize: 14, bold: true, color: C.text2 });
  plain(s, 8.3, 3.75, 4.43, 2.95, "The further ahead you look, the less the current book tells you. The order book is a short-lived signal: it describes pressure that is resolved within seconds.");

  // 37. FI-2010 native chart
  s = content("Benchmark: our models vs published DeepLOB on FI-2010",
    "FI-2010 is the standard academic order-book benchmark used by the DeepLOB paper. Our DeepLOB reproduces the published accuracy to within 0.7 to 2.3 points, which validates the implementation. At short horizons, a plain logistic regression that is given the same book history DeepLOB sees wins: 86.1% versus 83.1% at k equals 10. So DeepLOB's short-horizon advantage comes from seeing history, not from its architecture. At long horizons DeepLOB wins clearly, by 12 points at k equals 50. LightGBM's poor k equals 20 result is distribution shift: the test days are calmer, and validation data cannot detect it.");
  s.addChart(pres.charts.BAR, [
    { name: "Naive flat", labels: ["k = 10", "k = 20", "k = 50", "k = 100"], values: [70.7, 62.1, 47.3, 34.4] },
    { name: "Logistic + history", labels: ["k = 10", "k = 20", "k = 50", "k = 100"], values: [86.1, 78.3, 66.5, 60.2] },
    { name: "LightGBM + history", labels: ["k = 10", "k = 20", "k = 50", "k = 100"], values: [78.0, 56.6, 64.4, 75.7] },
    { name: "DeepLOB (ours)", labels: ["k = 10", "k = 20", "k = 50", "k = 100"], values: [83.1, 74.1, 78.2, 77.5] },
  ], Object.assign({
    x: X0, y: 1.3, w: 7.6, h: 5.45, barDir: "col", barGrouping: "clustered", chartColors: [HX.accent4, HX.accent1, HX.accent3, HX.accent2],
    valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 20, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", dataLabelFontSize: 10,
    valGridLine: { color: "E1E7EC", size: 0.75 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", showTitle: true, title: "FI-2010 Setup 2 test accuracy (%), mean of 3 seeds", barGapWidthPct: 50,
  }, chartText));
  bullets(s, [
    ["Reproduced.", "Our DeepLOB is within 0.7–2.3 points of the paper (83.1 vs 84.5 at k = 10)."],
    ["History beats architecture at short horizons.", "Logistic regression + book history: 86.1% vs DeepLOB 83.1% at k = 10."],
    ["Deep wins at long horizons.", "DeepLOB 78.2% vs 66.5% at k = 50."],
    ["Distribution shift.", "Test days are calmer (62% FLAT vs 49% in training); validation accuracy (86–88%) cannot see it."],
  ], { x: 8.45, y: 1.35, w: 4.28, h: 5.4, fontSize: 14, gap: 10 });

  // 37b. Simpler beats deep
  s = content("Simpler models beat deep networks, even in accuracy",
    "A common expectation is that deep networks reading the raw book are more accurate and only pay for it in speed. At the short horizons studied here the opposite holds. On Bitcoin at 20 seconds LightGBM is more accurate than DeepLOB, the LSTM and TLOB, in both plain accuracy and macro-F1, and the lead over DeepLOB is significant on every segment. The same holds on the synthetic book, and on FI-2010 at k equals 10 and 20 a logistic regression given the same book history beats DeepLOB. Three reasons: the signal is short-lived and sits at the top of the book, which a few hand-made features capture directly; a network with 214 thousand parameters overfits noise easily at this signal-to-noise ratio; and seeing history matters more than the architecture. Deep networks still win at longer FI-2010 horizons, and they had limited tuning, so the gap could narrow with more search.");
  s.addChart(pres.charts.BAR, [
    { name: "LightGBM", labels: ["BTC accuracy", "BTC macro-F1", "Synthetic macro-F1"], values: [0.476, 0.467, 0.584] },
    { name: "TLOB", labels: ["BTC accuracy", "BTC macro-F1", "Synthetic macro-F1"], values: [0.455, 0.449, 0.492] },
    { name: "LSTM", labels: ["BTC accuracy", "BTC macro-F1", "Synthetic macro-F1"], values: [0.448, 0.397, 0.521] },
    { name: "DeepLOB", labels: ["BTC accuracy", "BTC macro-F1", "Synthetic macro-F1"], values: [0.429, 0.401, 0.559] },
  ], Object.assign({
    x: X0, y: 1.3, w: 7.0, h: 3.9, barDir: "col", barGrouping: "clustered", chartColors: [HX.accent3, HX.accent5, HX.accent4, HX.accent2],
    valAxisMinVal: 0.3, valAxisMaxVal: 0.62, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 10,
    valGridLine: { color: "E1E7EC", size: 0.75 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", showTitle: true, title: "Same test rows: the simpler model scores highest", barGapWidthPct: 50,
  }, chartText));
  table(s, [
    ["", "LightGBM", "DeepLOB"],
    ["Accuracy (BTC, 20 s)", "0.476", "0.429"],
    ["Model size", "327 trees", "213,987 parameters"],
    ["Training per segment", "≈9 s, laptop CPU", "≈18 min, A100 GPU"],
    ["One prediction", "10.5 µs (C++)", "2.4–5.7 ms (GPU)"],
  ], { x: 7.95, y: 1.35, w: 4.78, colW: [1.9, 1.33, 1.55], fontSize: 12, rowH: 0.45 });
  card(s, 7.95, 3.85, 4.78, 1.35, "FI-2010 caveat", "Logistic + history wins at k = 10–20 (86.1% vs 83.1%); DeepLOB wins at k = 50–100.", { size: 13 });
  const why = [
    ["Short-lived signal at the top of the book", "A few hand-made features (imbalance, OFI, lags) capture it directly."],
    ["Too little signal for a big network", "214 k parameters overfit noise; LightGBM stops early on validation loss."],
    ["History matters more than architecture", "Give a linear model DeepLOB's past 100 rows and it wins at short horizons."],
  ];
  why.forEach((w, i) => card(s, X0 + i * 4.14, 5.4, 3.85, 1.35, w[0], w[1], { size: 13, headSize: 14 }));

  // 38. What drives predictions
  s = content("What drives the predictions? Ablation and SHAP",
    "Left: LightGBM macro-F1 as feature groups are added. The level-1 snapshot alone, four features, already reaches 0.436. Adding the nine lags of level-1 signals gives the largest single gain, plus 0.024. Everything deeper adds less than 0.01. Right: TreeSHAP, which splits each prediction exactly into contributions from each feature, computed on 20,000 test rows per segment. Deeper-book shape gets 39%, level-1 lags 35%, the level-1 snapshot 14%, VPIN 7%. The usual split-gain importance under-weights the lags, which is why gain is not a reliable explanation. Adding Ethereum's book helps Bitcoin a little at short horizons, consistent with Ethereum leading by a few seconds.");
  fig(s, "ablation_shap", X0, 1.3, CW, 4.3);
  card(s, X0, 5.7, 3.9, 1.05, "Top of book + recent history", "L1 alone: 0.436; + lags: +0.024.", { size: 13 });
  card(s, X0 + 4.12, 5.7, 3.9, 1.05, "SHAP shares", "Deep shape 39%, L1 lags 35%, L1 14%, VPIN 7%.", { size: 13 });
  card(s, X0 + 8.24, 5.7, 3.89, 1.05, "Cross-asset (ETH)", "+0.016 at 10 s, +0.012 at 20 s, +0.007 at 50 s.", { size: 13 });

  // 39. DeepLOB attribution
  s = content("The deep network learns to look at the same things",
    "Gradient times input attribution for DeepLOB on 2,000 windows per segment. It shows how much each part of the input moved the prediction. Level 1 counts twice as much as any deeper level, and 57% of the weight falls on the most recent 10 of its 100 snapshots. So the deep network, without being given hand-made features, ends up using the same information as LightGBM: the top of the book and its last few seconds. The price versus size split is partly a scaling effect of this attribution method.");
  fig(s, "deeplob_attribution", X0, 1.3, 8.3, 5.0);
  stat(s, 9.3, 1.5, 3.4, "2×", "Weight on level 1 versus any deeper level", C.accent2);
  stat(s, 9.3, 3.2, 3.4, "57%", "Of the weight on the most recent 10 of 100 snapshots", C.accent2);
  plain(s, 9.3, 4.95, 3.43, 1.8, "Hand-made features already capture what the network discovers, with far less data and compute.", "Takeaway");

  // 40. Regression micro-price
  s = content("Regression: a one-number rule rivals LightGBM on large ticks",
    "Here the target is the size of the future mid change, scored by the information coefficient, the correlation of forecast with reality. On large-tick stocks the mid price closes the gap to the micro-price within about 10 seconds (slope 0.93), so a one-parameter rule, beta times the micro-price gap, matches or beats LightGBM from 5 seconds on and gets the sign right 64 to 82% of the time. That matters for speed: the rule costs a few arithmetic operations. On Bitcoin, LightGBM regression reaches IC 0.263 and out-of-sample R-squared 0.067 at 10 seconds; ridge reaches 0.241 and 0.057. As a training target, the micro-price is never better than the mid.");
  fig(s, "microprice", X0, 1.3, 7.2, 2.9);
  table(s, [
    ["IC with future mid change", "Large tick 5 s", "Large tick 10 s", "Small tick 1 s", "BTC 10 s"],
    ["Micro-price rule (1 parameter)", "0.290", "0.278", "0.062", "0.024"],
    ["Momentum (past return)", "0.037", "0.012", "0.042", "0.074"],
    ["Ridge regression", "–", "–", "–", "0.241"],
    ["LightGBM (target: mid change)", "0.209", "0.170", "0.155", "0.263"],
    ["LightGBM (target: micro change)", "0.080", "0.009", "−0.007", "0.260"],
    ["Shuffled-target control", "−0.003", "0.005", "−0.003", "0.002"],
  ], { x: X0, y: 4.35, w: 7.2, colW: [2.6, 1.15, 1.2, 1.15, 1.1], fontSize: 12, rowH: 0.34, align: "center" });
  card(s, 8.1, 1.35, 4.63, 1.75, "Micro-price rule wins on large ticks", "IC 0.29 vs LightGBM 0.21 at 5 s; right sign 64–82% of the time.", { size: 14, icon: "target", iconFill: C.accent1 });
  card(s, 8.1, 3.25, 4.63, 1.6, "LightGBM wins elsewhere", "Small ticks (0.155) and Bitcoin (IC 0.263, R² 0.067).", { size: 14, icon: "bolt", iconFill: C.accent3 });
  card(s, 8.1, 5.0, 4.63, 1.7, "Controls stay at zero", "Shuffled targets give IC ≈ 0, so the skill is real.", { size: 14, icon: "check", iconFill: C.accent6 });

  // 41. Volatility / toxic fills
  s = content("Volatility is easier, and leakage can fake it",
    "Volatility is far more predictable than direction. On the realised-volatility data LightGBM cuts the naive error by about 30%, RMSPE 0.235 versus 0.337. The best public solution added nearest-neighbour features averaged over similar time windows, including future ones. Restricting the neighbours to the past keeps only about 59% of their gain, and shuffled K-fold cross-validation makes a plain model look as good as the leaky one. On closing-auction data, LightGBM improves MAE over predicting zero by 2.4%. For market making, toxic fills are only weakly predictable: AUC 0.55 to 0.58 on Bitcoin.");
  fig(s, "optiver_leakage", X0, 1.3, 6.6, 3.8);
  plain(s, X0, 5.3, 6.6, 1.4, "Letting a model see the future, even indirectly, can make a weak model look brilliant. How you validate changes the conclusion.", "Lesson");
  card(s, 7.55, 1.35, 5.18, 1.65, "Realised volatility", "LightGBM RMSPE 0.235 vs naive 0.337 (≈30% better). Past-only neighbours keep ≈59% of the leaky gain.", { size: 14, icon: "chart", iconFill: C.accent5 });
  card(s, 7.55, 3.15, 5.18, 1.65, "Closing auction", "481 days. MAE 5.848 vs 5.994 for predicting zero (2.4% better); retraining every 12 days helps only slightly.", { size: 14, icon: "clock", iconFill: C.accent6 });
  card(s, 7.55, 4.95, 5.18, 1.75, "Toxic fills", "Is a market maker's fill going to lose money? AUC 0.55–0.58 (BTC), 0.39–0.53 (Nasdaq). VPIN does not sort fills by toxicity.", { size: 14, icon: "shield", iconFill: C.accent2 });

  // =====================================================================
  section("06", "From prediction to profit", "A prediction is only worth something if a trading system can act on it in time and after paying to trade. Here accuracy, speed and cost meet.");

  // 42. Edge model
  s = content("A simple model of the trading edge",
    "For one round trip, the expected net edge is the accuracy edge, minus what is lost by acting late, minus the cost of crossing the spread, minus fees on both legs. The machine-learning model controls the first term, the system's speed the second, and the market and exchange the last two. If moves have average size m and the model is right with probability p, the accuracy term is about (2p minus 1) times m, so the break-even hit rate is one half plus C over 2m. If the cost C is bigger than the move m, the required hit rate is above 100%, and no model can win.");
  const terms = [
    ["Accuracy edge", "E[s·Δm] over the horizon", "Set by the ML model", C.accent1, "+"],
    ["Latency loss", "Edge lost by acting L later", "Set by system speed", C.accent5, "−"],
    ["Crossing cost", "Spread + walking the book", "Set by the market", C.accent3, "−"],
    ["Fees", "2 × fee per round trip", "Set by the exchange", C.accent2, "−"],
  ];
  T(s, "net edge  =", { x: X0, y: 1.6, w: 1.6, h: 1.3, fontSize: 20, bold: true, color: C.text2, valign: "middle", fontFace: "Cambria" });
  terms.forEach((t, i) => {
    const x = X0 + 1.65 + i * 2.65;
    if (i > 0) T(s, t[4], { x: x - 0.38, y: 1.6, w: 0.35, h: 1.3, fontSize: 28, bold: true, color: C.text2, align: "center", valign: "middle" });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.4, w: 2.3, h: 1.7, rectRadius: 0.08, fill: { color: t[3] }, line: { type: "none" } });
    T(s, t[0], { x: x + 0.1, y: 1.5, w: 2.1, h: 0.45, fontSize: 16, bold: true, color: C.background1, align: "center" });
    T(s, t[1], { x: x + 0.1, y: 1.95, w: 2.1, h: 0.55, fontSize: 12, color: C.background1, align: "center" });
    T(s, t[2], { x: x + 0.1, y: 2.55, w: 2.1, h: 0.4, fontSize: 12, italic: true, bold: true, color: C.background1, align: "center" });
  });
  box(s, X0, 3.45, 5.9, 3.25, C.text2);
  T(s, "Break-even hit rate", { x: X0 + 0.3, y: 3.6, w: 5.3, h: 0.45, fontSize: 18, bold: true, color: "F2C46D" });
  T(s, "p* = ½ + C / (2m)", { x: X0 + 0.3, y: 4.1, w: 5.3, h: 0.9, fontSize: 36, bold: true, color: C.background1, fontFace: "Cambria", valign: "middle" });
  T(s, "p = share of calls with the right direction\nm = average size of the move\nC = total round-trip cost", { x: X0 + 0.3, y: 5.1, w: 5.3, h: 1.45, fontSize: 14, color: "D5E3EA" });
  card(s, 6.75, 3.45, 5.98, 3.25, "Worked example (Bitcoin, 20 s)", "Typical move m ≈ 4.3 bp. Taker fee 4 bp per side, so C ≥ 8 bp.\n\np* = 0.5 + 8 / (2 × 4.3) = 1.43\n\nA hit rate of 143% is impossible: even a perfect forecast loses money. With C = 1 bp, p* would be 62%, which good signals can reach.", { icon: "flask", iconFill: C.accent3, size: 14 });

  // 43. P&L regimes
  s = content("The edge exists, but costs remove it",
    "The same predictions under three execution assumptions. Trading at the mid price with no costs, which several papers report, LightGBM earns 8,055 dollars per Bitcoin test window. As an aggressive taker with a one-second delay it loses 36,394 dollars, because fees alone, 37,631 dollars, are 4.7 times the cost-free profit. On the synthetic book, crossing the spread and paying fees removes 75% of LightGBM's cost-free profit and 94% of DeepLOB's. DeepLOB's signals flip more often, so it trades more and pays more friction.");
  fig(s, "pnl_regimes", X0, 1.3, CW, 4.1);
  stat(s, X0, 5.4, 3.0, "+$8,055", "BTC, trading at the mid with no costs (as many papers report)", C.accent1);
  stat(s, X0 + 3.1, 5.4, 3.0, "−$36,394", "Same signals as a taker with 1 s delay and real fees", C.accent2);
  stat(s, X0 + 6.2, 5.4, 2.9, "4.7×", "Fees ($37,631) relative to the cost-free profit", C.accent3);
  stat(s, X0 + 9.2, 5.4, 2.93, "75% / 94%", "Of cost-free P&L removed by costs (synthetic): LightGBM / DeepLOB", C.accent5);

  // 44. Waterfall + table
  s = content("Where the edge goes: one round trip, step by step",
    "Each Bitcoin round trip was split into the edge at the decision mid, the cost of reacting one snapshot later, and the cost of crossing the spread and walking the book. For the most confident 2% of signals the edge is 2.23 basis points; a one-second delay costs 0.75, crossing costs 1.01, leaving 0.47 gross. Then fees of 8 basis points turn it deeply negative. The table shows that trading only confident signals doubles the edge and raises the hit rate from 61% to 72%.");
  fig(s, "edge_waterfall", X0, 1.3, 6.0, 3.6);
  table(s, [
    ["Signals traded", "Trades", "Edge", "1 s delay", "Crossing", "Gross", "Hit rate"],
    ["All", "6,359", "1.01", "−0.27", "−0.73", "0.01", "60.7%"],
    ["Top 10%", "3,776", "1.97", "−0.65", "−0.89", "0.43", "68.9%"],
    ["Top 2%", "1,590", "2.23", "−0.75", "−1.01", "0.47", "71.8%"],
  ], { x: 6.85, y: 1.35, w: 5.88, colW: [1.3, 0.8, 0.7, 0.8, 0.8, 0.7, 0.78], fontSize: 12, rowH: 0.45, align: "center", hl: [3] });
  T(s, "bp per round trip, LightGBM, 20 s hold. Break-even hit rate for the top 2%: 70% at 0 bp fee, 94% at 1 bp, over 100% at 4 bp.", { x: 6.85, y: 3.25, w: 5.88, h: 0.75, fontSize: 12, italic: true, color: C.accent4 });
  const four = [
    ["Better accuracy doubles the edge", "Confident signals only: hit rate 61% → 72%, edge 1.0 → 2.2 bp."],
    ["Better signals are more delay-sensitive", "1 s delay costs 0.27 bp on all signals, 0.75 bp on the top 2%."],
    ["Without fees, the bar is cleared", "72% achieved vs 70% needed at 0 bp fee."],
    ["At the real 4 bp fee, nothing is enough", "Mean |move| ≈ 4.3 bp < 8 bp round-trip fee, so p* > 1."],
  ];
  four.forEach((f, i) => {
    const x = i % 2 === 0 ? X0 : 6.85, y = i < 2 ? 4.25 : 5.5;
    card(s, x, y, i % 2 === 0 ? 6.0 : 5.88, 1.15, f[0], f[1], { size: 13, headSize: 15 });
  });

  // 45. Hit rate vs confidence
  s = content("Confidence filtering: trade less, be right more often",
    "Left: the share of UP or DOWN calls with the right sign, on rows that actually moved, as each model keeps only its most confident predictions. Every tabular model improves from about 69% when it trades every row to 86 to 87% for the top 2%. These numbers are higher than in the previous table because only rows with a label beyond the threshold count. Right: the hit rate needed to break even against round-trip cost, with LightGBM's achieved hit rates. Good calibration is what makes this filtering work.");
  fig(s, "hit_rate_breakeven", X0, 1.3, CW, 4.3);
  table(s, [
    ["Hit rate on moved rows", "All rows", "Top 50%", "Top 20%", "Top 10%", "Top 5%", "Top 2%"],
    ["LightGBM", "68.5%", "74.3%", "79.2%", "82.1%", "84.3%", "86.5%"],
    ["Random Forest", "68.6%", "74.6%", "79.6%", "82.6%", "85.0%", "87.0%"],
    ["Logistic regression", "68.0%", "72.9%", "78.5%", "81.3%", "84.2%", "85.6%"],
  ], { x: X0, y: 5.7, w: CW, colW: [2.83, 1.55, 1.55, 1.55, 1.55, 1.55, 1.55], fontSize: 11, rowH: 0.27, align: "center" });

  // 46. Latency native chart
  s = content("How fast is each model?",
    "Time to make one prediction, on a logarithmic scale where each grid line is ten times slower. Logistic regression and LightGBM from Python take about a tenth of a millisecond. Random Forest from Python is slow, 42.6 ms, because of thread start-up on every call. DeepLOB on a GPU with batch size one takes 2.4 to 5.7 ms. LightGBM compiled to plain C++ takes 10.5 microseconds, with exactly the same predictions as the Python model on 20,000 test rows. A full C++ pipeline of features plus a 50-tree ensemble plus order logic takes 1.9 microseconds, about 399,000 snapshots per second. For reference, the Bitcoin data arrives once per second.");
  s.addChart(pres.charts.BAR, [{ name: "Median latency (µs)", labels: ["Random Forest (Python)", "DeepLOB (GPU, BTC)", "DeepLOB (GPU, synthetic)", "Logistic reg. (Python)", "LightGBM (Python)", "LightGBM → C++"], values: [42570, 5700, 2400, 137, 109, 10.5] }], Object.assign({
    x: X0, y: 1.3, w: 7.4, h: 5.45, barDir: "bar", chartColors: [HX.accent1, HX.accent2, HX.accent2, HX.accent1, HX.accent1, HX.accent3],
    valAxisLogScaleBase: 10, valAxisMinVal: 1, valAxisMaxVal: 100000, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0.#",
    valGridLine: { color: "E1E7EC", size: 0.75 }, catGridLine: { style: "none" }, showLegend: false, showTitle: true, title: "Median time for one prediction, microseconds (log scale)", barGapWidthPct: 45, valAxisLabelFormatCode: "#,##0",
  }, chartText));
  table(s, [
    ["Model / step", "P50", "P99"],
    ["LightGBM → C++ (327 trees)", "10.5 µs", "17.9 µs"],
    ["C++ features + 50 trees + order", "1.9 µs", "6.3 µs"],
    ["LightGBM (Python, 1 thread)", "109 µs", "217 µs"],
    ["Logistic regression (Python)", "137 µs", "264 µs"],
    ["DeepLOB (GPU, batch 1)", "2.4–5.7 ms", "–"],
    ["Random Forest (Python)", "42.6 ms", "47.5 ms"],
    ["Micro-price rule", "a few ns", "–"],
  ], { x: 8.25, y: 1.35, w: 4.48, colW: [2.58, 1.0, 0.9], fontSize: 12, rowH: 0.4, hl: [1] });
  T(s, "The C++ model is bit-identical to Python (max difference 0, 100% agreement on 20,000 rows). For scale: Bitcoin data arrives every 1 s; the Nasdaq signal peaks 1–5 s ahead.", { x: 8.25, y: 4.85, w: 4.48, h: 1.8, fontSize: 13, color: C.accent4, italic: true });

  // 47. Speed vs accuracy
  s = content("No trade-off: the most accurate model is also one of the fastest",
    "Accuracy against inference speed on Bitcoin. The usual assumption, that deep models are more accurate but slower, does not hold at these horizons. LightGBM is the most accurate model and, compiled to C++, 230 to 540 times faster than DeepLOB on a GPU without changing a single prediction. Random Forest is accurate but slow from Python.");
  fig(s, "speed_vs_accuracy", X0, 1.3, 7.8, 5.4);
  stat(s, 8.8, 1.5, 3.9, "230–540×", "LightGBM in C++ is this much faster than DeepLOB on a GPU", C.accent3);
  stat(s, 8.8, 3.25, 3.9, "+0.066", "And more accurate: macro-F1 0.467 vs 0.401", C.accent1);
  plain(s, 8.8, 5.0, 3.93, 1.7, "Top-left is best: high accuracy, little time. LightGBM sits there; the deep models sit lower and to the right.");

  // 48. Latency sweep native scatter
  s = content("Turning milliseconds into dollars",
    "To measure the value of speed directly, the synthetic book, with events about 10 ms apart and a 200 ms horizon, was traded as an aggressive taker with extra delay added to every order. Both models lose about the same per millisecond: 3.4 dollars for LightGBM, 3.6 for DeepLOB. DeepLOB's extra 2.3 ms of GPU inference therefore costs about 10 dollars, while LightGBM's better signal is worth about 350 dollars at zero delay. Choosing the more accurate model is worth about 35 times more than choosing the faster one. Accuracy also buys latency budget: DeepLOB's profit is gone by about 30 ms; LightGBM's lasts beyond 100 ms. On Bitcoin as a taker, latency barely matters because fees dominate.");
  s.addChart(pres.charts.SCATTER, [
    { name: "Added latency (ms)", values: [0, 1, 10, 25, 50, 100] },
    { name: "LightGBM", values: [508, 457, 427, 369, 290, 119] },
    { name: "DeepLOB", values: [155, 104, 71, 14, -76, -247] },
  ], Object.assign({
    x: X0, y: 1.3, w: 7.4, h: 5.45, chartColors: [HX.accent1, HX.accent2], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 8,
    valAxisMinVal: -300, valAxisMaxVal: 600, valAxisMajorUnit: 100, catAxisMinVal: 0, catAxisMaxVal: 100, catAxisMajorUnit: 25,
    showValAxisTitle: true, valAxisTitle: "Net P&L per test window ($)", showCatAxisTitle: true, catAxisTitle: "Added latency (ms)",
    valAxisTitleFontSize: 12, catAxisTitleFontSize: 12, valAxisTitleColor: "4A5866", catAxisTitleColor: "4A5866", valAxisTitleFontFace: "+mn-lt", catAxisTitleFontFace: "+mn-lt",
    valGridLine: { color: "E1E7EC", size: 0.75 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", showTitle: true, title: "Synthetic latency sweep (mean of 5 seeds)",
  }, chartText));
  stat(s, 8.3, 1.35, 2.15, "≈$3.5", "Lost per extra ms (LightGBM $3.4, DeepLOB $3.6)", C.accent5, 34);
  stat(s, 10.6, 1.35, 2.13, "35×", "Accuracy worth more than speed ($350 vs $10)", C.accent3, 34);
  stat(s, 8.3, 3.15, 2.15, "≈30 ms", "DeepLOB's profit is gone", C.accent2, 34);
  stat(s, 10.6, 3.15, 2.13, ">100 ms", "LightGBM still profitable (break-even ≈134 ms)", C.accent1, 34);
  plain(s, 8.3, 4.95, 4.43, 1.75, "A more accurate model can afford a slower system. On Bitcoin as a taker, going from 1 s to 0 s delay improves P&L by only ≈5%: fees dominate.");

  // 49. Latency figure (both panels)
  s = content("Latency sweep: synthetic book vs real Bitcoin",
    "Left: the synthetic book, five seeds, with a shaded band of plus or minus one standard error. Right: Bitcoin over four segments, where latency is in whole seconds because that is the data's resolution. On Bitcoin the lines are deep in the negative and nearly flat: going from 1 s to 0 s improves P&L from minus 36.4 to minus 34.5 thousand dollars, and even 30 s of delay only moves it to minus 41.0 thousand.");
  fig(s, "latency_pnl", X0, 1.3, CW, 4.5);
  card(s, X0, 5.9, 6.0, 0.85, "Synthetic: speed matters, accuracy matters more", "", { size: 13 });
  card(s, X0 + 6.13, 5.9, 6.0, 0.85, "Bitcoin taker: fees swamp everything", "", { size: 13 });

  // 50. Passive trading
  s = content("Passive trading: when speed is the whole edge",
    "A market maker posts resting orders and earns the spread instead of paying it, but its orders tend to fill just before the price moves against it, which is called adverse selection. On Bitcoin there is no spread to earn, and naive two-sided quoting loses about 10,400 dollars a day; using the toxic-fill model to quote only when the expected value is positive cuts losses by 86% or more, but breaks even only under optimistic fill assumptions. On large-tick Nasdaq stocks, the result depends on queue position, which is a pure speed race: a fill at the front of the queue earns money, at the back it loses, and with 13,000 shares already queued a new order is almost always at the back.");
  card(s, X0, 1.35, 5.9, 2.4, "Bitcoin", "No spread to earn (0.0017 bp tick). Naive two-sided quoting loses ≈$10,400 a day. Quoting only when the toxic-fill model predicts positive value cuts losses by ≥86%, but breaks even only under optimistic fills.", { icon: "btc", iconFill: C.accent3, size: 14 });
  card(s, X0, 3.95, 5.9, 2.75, "Large-tick Nasdaq", "Queue position decides everything. With ≈13,000 shares already waiting, a new order is almost always at the back of the queue. Prediction helps decide when to quote, but latency decides whether quoting pays.", { icon: "building", iconFill: C.accent5, size: 14 });
  T(s, "Value of a fill 30 s later (bp)", { x: 6.85, y: 1.35, w: 5.88, h: 0.4, fontSize: 15, bold: true, color: C.text2 });
  s.addChart(pres.charts.BAR, [
    { name: "Front of queue", labels: ["INTC", "MSFT"], values: [1.24, 0.70] },
    { name: "Back of queue", labels: ["INTC", "MSFT"], values: [-2.22, -1.78] },
  ], Object.assign({
    x: 6.85, y: 1.8, w: 5.88, h: 4.9, barDir: "col", barGrouping: "clustered", chartColors: [HX.accent1, HX.accent2], valAxisMinVal: -2.5, valAxisMaxVal: 1.5, valAxisMajorUnit: 0.5,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.00;−0.00", valGridLine: { color: "E1E7EC", size: 0.75 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", barGapWidthPct: 60, valAxisLabelFormatCode: "0.0",
  }, chartText));

  // 51. Balancing rule
  s = content("The balancing rule: cost, then accuracy, then speed",
    "This is the practical recipe that comes out of the project. First check feasibility: if the typical move over the horizon is not clearly larger than the round-trip cost, nothing will work, so change the instrument, the horizon or the execution style. Then choose the most accurate model whose inference fits well inside the horizon. Then spend on latency in proportion to how fast the signal decays, keeping latency a small fraction of the horizon, and smaller for better signals. For passive trading, queue position, which is speed, dominates; prediction is the filter.");
  const rules = [
    ["1", "Check feasibility first", "Is the typical move m over horizon h clearly larger than the round-trip cost C? If not, no model or system will work: change the instrument, horizon or execution style.", C.accent2],
    ["2", "Choose the model by accuracy", "Pick the most accurate model whose inference fits well inside h. At seconds every model fits; near 100 ms only microsecond models do; at long horizons deep models become affordable.", C.accent1],
    ["3", "Spend on speed in proportion to decay", "Keep latency L a small fraction of h, smaller for better signals, because their edge is concentrated in the first moments.", C.accent5],
    ["4", "Passive trading is a speed race", "For resting orders, queue position dominates; prediction decides when to quote.", C.accent3],
  ];
  rules.forEach((r, i) => {
    const y = 1.35 + i * 1.35;
    box(s, X0, y, CW, 1.18);
    s.addShape(pres.shapes.OVAL, { x: X0 + 0.25, y: y + 0.26, w: 0.66, h: 0.66, fill: { color: r[3] }, line: { type: "none" } });
    T(s, r[0], { x: X0 + 0.25, y: y + 0.26, w: 0.66, h: 0.66, fontSize: 22, bold: true, color: C.background1, align: "center", valign: "middle" });
    T(s, r[1], { x: X0 + 1.2, y: y, w: 3.5, h: 1.18, fontSize: 17, bold: true, color: C.text2, valign: "middle" });
    T(s, r[2], { x: X0 + 4.8, y: y, w: 7.1, h: 1.18, fontSize: 14, valign: "middle" });
  });

  // =====================================================================
  section("07", "What we learned", "What the results mean for machine learning on order-book data, the limits of this study, and where it could go next.");

  // 52. Discussion
  s = content("What this means for machine learning on order books",
    "Four lessons. Feature engineering still beats raw-input deep learning at short horizons on this data: the hand-made features encode exactly what DeepLOB's attribution shows it learns to look for. The input matters more than the architecture: giving a linear model the same history DeepLOB sees closes and reverses the gap at short horizons. Validation design changes the answer: shuffled cross-validation, full-sample normalisation, a relative-spread feature and look-ahead neighbours each produced convincing but false improvements. And classification metrics are not economic metrics.");
  const less = [
    ["cogs", "Features still beat raw deep learning at short horizons", "OFI, imbalance and lags encode what DeepLOB learns to look at, with far less data and compute.", C.accent1],
    ["eye", "The input matters more than the architecture", "On FI-2010, giving logistic regression the same history as DeepLOB reverses the gap at short horizons.", C.accent5],
    ["shield", "Validation design changes the answer", "Shuffled CV, full-sample normalising, a relative-spread feature and look-ahead neighbours all produced false gains.", C.accent2],
    ["dollar", "Statistical is not economic", "+0.05 macro-F1 is statistically certain, but worthless at a 4 bp fee and valuable at 0 bp.", C.accent3],
  ];
  less.forEach((l, i) => {
    const x = X0 + (i % 2) * 6.17, y = 1.35 + Math.floor(i / 2) * 2.7;
    card(s, x, y, 5.96, 2.5, l[1], l[2], { icon: l[0], iconFill: l[3], size: 15 });
  });

  // 53. Limitations & future
  s = content("Limitations and future work",
    "Be upfront about the limits. The real samples are small: 12 days of Bitcoin from one exchange and one day per Nasdaq stock, so LOBSTER results are a case study. Bitcoin data is one snapshot per second, so its one-second delay is the data's resolution, not a real system latency. Queue position is not observed, so passive results are bounds. Our orders do not have lasting price impact. The latency sweep is synthetic, so its dollar figures show the mechanism, not real magnitudes. The deep models had limited tuning, and the C++ timing covers inference only. Future work addresses each of these.");
  T(s, "Limitations", { x: X0, y: 1.3, w: 6.0, h: 0.45, fontSize: 18, bold: true, color: C.accent2 });
  bullets(s, [
    ["Small real samples:", "12 days of BTC from one exchange; one day per Nasdaq stock."],
    ["Coarse BTC data:", "1 s snapshots, so “1 s delay” is the data's resolution."],
    ["Queue position unobserved:", "passive fills reported as bounds."],
    ["No lasting price impact", "of our own orders in the simulator."],
    ["Synthetic latency sweep:", "shows the mechanism, not real-market dollar sizes."],
    ["Limited deep-model tuning", "(8–15 epochs) and a simplified TLOB."],
    ["Timing covers inference only,", "not network or feed handling."],
  ], { x: X0, y: 1.85, w: 5.9, h: 4.9, fontSize: 14, gap: 8 });
  T(s, "Future work", { x: 6.85, y: 1.3, w: 5.88, h: 0.45, fontSize: 18, bold: true, color: C.accent1 });
  const fut = [
    ["building", "Passive execution on large-tick stocks with a queue-position model and many days of LOBSTER data."],
    ["brain", "Deep models on sub-second Nasdaq data: does “deep wins at long horizons” hold outside FI-2010?"],
    ["chart", "Volatility-normalised history features, and online / retrained models for changing markets."],
    ["gear", "Wire-to-order latency: book engine, feed parser and fast networking around the C++ model."],
  ];
  fut.forEach((f, i) => {
    const y = 1.9 + i * 1.2;
    icon(s, f[0], 6.85, y + 0.05, 0.6, C.accent1);
    T(s, f[1], { x: 7.65, y, w: 5.08, h: 1.05, fontSize: 14, valign: "middle" });
  });

  // 54. Conclusion (dark)
  s = pres.addSlide({ masterName: "DARK", sectionTitle: curSection });
  s.addText("Conclusion", { placeholder: "title" });
  s.addNotes("Close with the recipe: pick a market and execution style where the typical move exceeds the cost; pick the most accurate model whose inference fits well inside the signal's horizon, which at short horizons is a simple model compiled to fast code; and invest in speed in proportion to how concentrated the signal is in its first moments.");
  const concl = [
    ["The book is predictable, modestly", "LightGBM macro-F1 0.467 and AUC 0.658 on Bitcoin at 20 s, significantly better than every other model."],
    ["Simpler beats deep, even in accuracy", "LightGBM is more accurate than DeepLOB on Bitcoin (0.476 vs 0.429); DeepLOB wins only at long FI-2010 horizons."],
    ["Costs set the bar", "On Bitcoin the 8 bp round trip exceeds the 4.3 bp typical move, so no model profits as a taker."],
    ["Accuracy first, then speed", "Accuracy was worth 35× more than speed, and LightGBM in C++ (10.5 µs) is both faster and more accurate."],
  ];
  concl.forEach((c, i) => {
    const x = X0 + (i % 2) * 6.17, y = 2.75 + Math.floor(i / 2) * 1.95;
    box(s, x, y, 5.96, 1.7, "1D4361", "conclusion card");
    T(s, c[0], { x: x + 0.25, y: y + 0.2, w: 5.5, h: 0.42, fontSize: 17, bold: true, color: "F2C46D" });
    T(s, c[1], { x: x + 0.25, y: y + 0.68, w: 5.5, h: 0.9, fontSize: 15, color: C.background1 });
  });
  T(s, "Trading costs set the bar, accuracy decides whether a model clears it, and latency matters in proportion to how quickly the signal decays.", { x: X0, y: 1.4, w: CW, h: 1.0, fontSize: 20, italic: true, color: "CFE3E6" });

  // 55. References
  s = content("References",
    "The main references behind the features, models and validation methods. The full list of 17 is in the written report.");
  const refs = [
    "R. Cont, A. Kukanov, S. Stoikov. The price impact of order book events. J. Financial Econometrics, 2014.",
    "D. Easley, M. López de Prado, M. O'Hara. Flow toxicity and liquidity in a high-frequency world. Rev. Financial Studies, 2012.",
    "S. Stoikov. The micro-price: a high-frequency estimator of future prices. Quantitative Finance, 2018.",
    "Z. Zhang, S. Zohren, S. Roberts. DeepLOB: deep convolutional neural networks for limit order books. IEEE TSP, 2019.",
    "A. Ntakaris et al. Benchmark dataset for mid-price forecasting of limit order book data (FI-2010). J. Forecasting, 2018.",
    "L. Berti, G. Kasneci. TLOB: a transformer with dual attention for LOB price trend prediction. arXiv:2502.15757, 2025.",
    "D. T. Tran et al. Data normalization for bilinear structures in high-frequency financial time-series. ICPR, 2021.",
    "G. Ke et al. LightGBM: a highly efficient gradient boosting decision tree. NeurIPS, 2017.",
    "M. López de Prado. Advances in Financial Machine Learning. Wiley, 2018.",
    "S. Lundberg, S.-I. Lee. A unified approach to interpreting model predictions (SHAP). NeurIPS, 2017.",
    "L. Breiman. Random forests. Machine Learning, 2001.  ·  S. Hochreiter, J. Schmidhuber. Long short-term memory. 1997.",
    "R. Huang, T. Polak. LOBSTER: limit order book reconstruction system. 2011.  ·  H. Künsch. Block bootstrap. Ann. Stat., 1989.",
  ];
  T(s, refs.map((r, i) => ({ text: `[${i + 1}]  ` + r, options: { breakLine: i < refs.length - 1, paraSpaceAfter: 7 } })), { x: X0, y: 1.35, w: CW, h: 5.4, fontSize: 13 });

  await pres.writeFile({ fileName: OUT });
  if (SKILL_THEME) {
    const { applyTheme } = require(SKILL_THEME);
    await applyTheme(OUT, THEME);
  }
  console.log("wrote", OUT);
}
main().catch((e) => { console.error(e); process.exit(1); });

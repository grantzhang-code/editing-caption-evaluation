"use strict";

const byId = (id) => document.getElementById(id);
const numeric = (id) => Number(byId(id).value);
const fixed = (n, digits = 3) => n.toFixed(digits);

function quality(p, r, critical) {
  return critical || p + r === 0 ? 0 : 2 * p * r / (p + r);
}

function edgeScores(target, source, c, o, j, ep, eg) {
  const matched = target >= 0.5 && source >= 0.5;
  const w = matched ? Math.min(target, source) : 0;
  return { matched, w, grounded: 2 * w * c / (ep + eg),
    joint: 2 * w * c * o * j / (ep + eg) };
}

function update() {
  const p = numeric("precision"), r = numeric("recall");
  const critical = byId("critical").checked;
  const q = quality(p, r, critical);
  byId("q-value").textContent = fixed(q);
  byId("q-fill").style.width = (q * 100) + "%";
  byId("q-detail").textContent = critical
    ? "critical_error = true → Q 直接归零"
    : p + r === 0 ? "p + r = 0 → Q = 0"
    : "2 × " + fixed(p, 2) + " × " + fixed(r, 2) + " ÷ " + fixed(p + r, 2) + " = " + fixed(q);

  const target = numeric("target-iou"), source = numeric("source-iou");
  byId("target-iou-value").textContent = fixed(target, 2);
  byId("source-iou-value").textContent = fixed(source, 2);
  const ep = numeric("ep"), eg = numeric("eg");
  if (![ep, eg].every((n) => Number.isInteger(n) && n >= 1 && n <= 20)) {
    byId("grounded-value").textContent = "—";
    byId("joint-value").textContent = "—";
    byId("grounded-detail").textContent = "";
    byId("joint-detail").textContent = "";
    byId("joint-explanation").textContent = "请输入 1–20 之间的整数连接数。本示例固定一条候选连接。";
    byId("match-status").textContent = "请检查连接数";
    byId("match-status").classList.add("fail");
    return;
  }
  const c = Number(byId("correspondence").checked);
  const o = Number(byId("origin-correct").checked);
  const keep = byId("origin").value === "keep";
  const j = keep ? o : q;
  const result = edgeScores(target, source, c, o, j, ep, eg);
  byId("grounded-value").textContent = fixed(result.grounded);
  byId("joint-value").textContent = fixed(result.joint);
  byId("grounded-detail").textContent = "2 × " + fixed(result.w, 2) + " × " + c + " / (" + ep + " + " + eg + ")";
  byId("joint-detail").textContent = "2 × " + fixed(result.w, 2) + " × " + c + " × " + o + " × " + fixed(j) + " / (" + ep + " + " + eg + ")";
  byId("match-status").textContent = result.matched
    ? "几何匹配成立 · 两端 IoU ≥ 0.50"
    : "几何匹配未成立 · 至少一端 IoU < 0.50";
  byId("match-status").classList.toggle("fail", !result.matched);
  byId("joint-explanation").textContent = !result.matched
    ? "这条连接没有通过匹配门槛，两个指标的分子均为零。未匹配连接仍计入分母。"
    : keep
    ? "GT 为 keep：J = O，忽略上方指令 Q。由于 O 是 0/1，O × O = O。"
    : "GT 为 edit：J = 上方的 Q_instruction = " + fixed(q) + "。来源、类型或指令质量任一为零，Joint F1 就为零。";
}

document.querySelectorAll("#guide-panel input, #guide-panel select").forEach((el) => el.addEventListener("input", update));
byId("reset-joint").addEventListener("click", () => {
  const defaults = { precision: ".75", recall: ".5", "target-iou": ".8", "source-iou": ".7", origin: "edit", ep: "1", eg: "1" };
  for (const id in defaults) byId(id).value = defaults[id];
  byId("critical").checked = false;
  byId("correspondence").checked = true;
  byId("origin-correct").checked = true;
  update();
});
byId("print-page").addEventListener("click", () => window.print());
let closedDetails = [];
window.addEventListener("beforeprint", () => {
  closedDetails = Array.from(document.querySelectorAll("details:not([open])"));
  closedDetails.forEach((el) => { el.open = true; });
});
window.addEventListener("afterprint", () => closedDetails.forEach((el) => { el.open = false; }));

const navLinks = Array.from(document.querySelectorAll(".sidebar nav a"));
const tracked = navLinks.map((link) => document.querySelector(link.getAttribute("href"))).filter(Boolean);
function activeSection() {
  let current = tracked[0];
  for (const section of tracked) {
    if (section.getBoundingClientRect().top <= 150) current = section;
  }
  for (const link of navLinks) {
    const active = link.getAttribute("href") === "#" + current.id;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  }
}
let scheduled = false;
window.addEventListener("scroll", () => {
  if (!scheduled) {
    scheduled = true;
    requestAnimationFrame(() => { activeSection(); scheduled = false; });
  }
}, { passive: true });
update();
activeSection();

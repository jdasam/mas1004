'use strict';
/* For each preset candidate, update rule and learning rate: over seeds 1..30, how many runs reach
   the target accuracy within 3,000 steps, the median step, the final training and test accuracy.
   Run from the demo directory:  node test/calibrate.js [blobs|moons|xor|spirals ...] */
const LB = require('./load.js')();

const CANDIDATES = [
  { data: 'blobs',   act: 'tanh',  layers: 0, target: 1,    lrs: { gd: [0.3, 1, 3],      adam: [0.02, 0.05, 0.1] } },
  { data: 'moons',   act: 'tanh',  layers: 1, target: null, lrs: { gd: [0.3, 1, 3],      adam: [0.02, 0.05] } },
  { data: 'moons',   act: 'tanh',  layers: 3, target: 1,    lrs: { gd: [0.3, 0.5, 1, 2], adam: [0.02] } },
  { data: 'xor',     act: 'tanh',  layers: 3, target: 1,    lrs: { gd: [0.5, 1, 2],      adam: [0.02] } },
  { data: 'spirals', act: 'leaky', layers: 4, target: 0.99, lrs: { gd: [0.3, 1],         adam: [0.05] } },
];
const SEEDS = 30, MAX_STEPS = 3000;
const median = a => { if (!a.length) return null; const b = a.slice().sort((x, y) => x - y); return b[b.length >> 1]; };

function runOne(c, data, test, rule, lr, seed) {
  const net = LB.initNet(c.layers, c.act, seed), opt = LB.makeOptimizer(net);
  let at = null;
  for (let s = 0; s < MAX_STEPS; s++) {
    const r = LB.trainStep(net, opt, data, lr, rule);
    if (!Number.isFinite(r.loss)) break;
    if (c.target !== null && r.acc >= c.target) { at = s; break; }
  }
  return { seed, at, acc: LB.evaluate(net, data).acc, test: LB.evaluate(net, test).acc };
}

const only = process.argv.slice(2);
for (const c of CANDIDATES) {
  if (only.length && !only.includes(c.data)) continue;
  const data = LB.makeData(c.data), test = LB.makeTestData(c.data);
  for (const rule of LB.RULES) for (const lr of c.lrs[rule]) {
    const rs = [];
    for (let seed = 1; seed <= SEEDS; seed++) rs.push(runOne(c, data, test, rule, lr, seed));
    const ok = rs.filter(r => r.at !== null);
    console.log(JSON.stringify({
      data: c.data, act: c.act, layers: c.layers, rule, lr, target: c.target,
      reach: c.target === null ? null : ok.length + '/' + SEEDS, medianSteps: median(ok.map(r => r.at)),
      acc: { min: Math.min(...rs.map(r => r.acc)), median: median(rs.map(r => r.acc)), max: Math.max(...rs.map(r => r.acc)) },
      testMedian: median(rs.map(r => r.test)), seeds123: rs.slice(0, 3).map(r => [r.at, r.acc]),
      /* without a target, every seed's final training accuracy, sorted */
      accs: c.target === null ? rs.map(r => r.acc).sort((x, y) => x - y) : undefined
    }));
  }
}

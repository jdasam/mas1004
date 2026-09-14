'use strict';
/* Measures how many full-batch Adam steps the network needs to classify every training point correctly,
   once with the real labels and once with the shuffled labels, starting from the same knobs for each seed.
   Prints one JSON line per run, then one summary line per width, learning rate and label set.
   `reached` is the step count after which all 200 training points are correct (null if not within maxSteps);
   trainAcc and testAcc are measured when the run stops (test accuracy uses the real labels of the test points).
   Run from the demo directory:
     node test/calibrate.js <width> <lr,lr,...> [seedFrom=1] [seedTo=30] [maxSteps=6000]
     node test/calibrate.js --summary a.jsonl b.jsonl ...     (one summary over runs saved by several processes) */
const fs = require('fs');
const MS = require('./load.js')();

function median(a){
  if (!a.length) return null;
  const b = a.slice().sort((x, y) => x - y), h = b.length >> 1;
  return +(b.length % 2 ? b[h] : (b[h - 1] + b[h]) / 2).toFixed(4);
}

function summarize(runs){
  const groups = new Map();
  for (const r of runs) {
    const key = r.width + '|' + r.lr + '|' + r.labels;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }
  const order = { real: 0, shuffled: 1 };
  return Array.from(groups.values()).map(rs => {
    const steps = rs.filter(r => r.reached !== null).map(r => r.reached);
    const ms = rs.reduce((s, r) => s + r.ms, 0), n = rs.reduce((s, r) => s + r.steps, 0);
    return {
      summary: true, width: rs[0].width, lr: rs[0].lr, labels: rs[0].labels,
      reached: steps.length + '/' + rs.length,
      medianSteps: median(steps), maxReachedSteps: steps.length ? Math.max(...steps) : null,
      medianTrainAcc: median(rs.map(r => r.trainAcc)), medianTestAcc: median(rs.map(r => r.testAcc)),
      msPerStep: +(ms / n).toFixed(2)
    };
  }).sort((a, b) => a.width - b.width || a.lr - b.lr || order[a.labels] - order[b.labels]);
}

function runAll(H, lrs, seedFrom, seedTo, maxSteps){
  const train = MS.makeMoons('train'), test = MS.makeMoons('test');
  const sets = { real: train, shuffled: { n: train.n, X: train.X, y: MS.shuffleLabels(train.y, MS.SHUFFLE_SEED) } };
  const runs = [];
  for (const lr of lrs) for (let seed = seedFrom; seed <= seedTo; seed++) for (const labels of ['real', 'shuffled']) {
    const data = sets[labels], net = MS.initNet(H, seed), st = MS.makeAdam(net);
    let reached = null, steps = 0;
    const t0 = process.hrtime.bigint();
    while (steps < maxSteps) {
      const r = MS.adamStep(net, st, data, lr);   // loss and accuracy before this update
      if (r.acc >= 1) { reached = steps; steps++; break; }
      steps++;
    }
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;
    const run = { width: H, lr, seed, labels, reached, steps,
                  trainAcc: MS.evaluate(net, data).acc, testAcc: MS.evaluate(net, test).acc, ms: +ms.toFixed(1) };
    runs.push(run);
    console.log(JSON.stringify(run));
  }
  return runs;
}

const args = process.argv.slice(2);
if (args[0] === '--summary') {
  const runs = [];
  for (const f of args.slice(1)) {
    for (const line of fs.readFileSync(f, 'utf8').split('\n')) {
      if (!line.startsWith('{')) continue;
      const r = JSON.parse(line);
      if (!r.summary) runs.push(r);
    }
  }
  summarize(runs).forEach(s => console.log(JSON.stringify(s)));
} else {
  const H = parseInt(args[0], 10), lrs = String(args[1] || '').split(',').map(Number);
  if (!(H > 0) || !lrs.every(lr => lr > 0)) {
    console.error('usage: node test/calibrate.js <width> <lr,lr,...> [seedFrom=1] [seedTo=30] [maxSteps=6000]');
    process.exit(1);
  }
  const seedFrom = parseInt(args[2] || '1', 10), seedTo = parseInt(args[3] || '30', 10), maxSteps = parseInt(args[4] || '6000', 10);
  summarize(runAll(H, lrs, seedFrom, seedTo, maxSteps)).forEach(s => console.log(JSON.stringify(s)));
}

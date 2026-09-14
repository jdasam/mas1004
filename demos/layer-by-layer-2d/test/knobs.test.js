'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const LB = require('./load.js')();
const knobs = net => LB.knobList(net).map(id => LB.getKnob(net, id));

test('evaluate agrees with lossGrad, and pointLoss is -log p(correct)', () => {
  const d = LB.makeData('moons'), net = LB.initNet(2, 'tanh', 4);
  const e = LB.evaluate(net, d), r = LB.lossGrad(net, d);
  assert.ok(Math.abs(e.loss - r.loss) < 1e-12 && e.acc === r.acc);
  assert.ok(Math.abs(LB.lossOnly(net, d) - r.loss) < 1e-12);
  assert.equal(e.z.length, d.n);
  assert.ok(Math.abs(e.z[5] - LB.logit(net, d.X[10], d.X[11])) < 1e-12);
  const p = 1 / (1 + Math.exp(-0.8));
  assert.ok(Math.abs(LB.pointLoss(0.8, 1) + Math.log(p)) < 1e-12);
  assert.ok(Math.abs(LB.pointLoss(0.8, 0) + Math.log(1 - p)) < 1e-12);
  assert.ok(Number.isFinite(LB.pointLoss(800, 0)) && Number.isFinite(LB.pointLoss(-800, 1)));
});

test('knobList: 6 per layer then 3 for the readout; get and set reach the right numbers', () => {
  const net = LB.initNet(2, 'tanh', 1), ids = LB.knobList(net);
  assert.equal(ids.length, 15); assert.equal(LB.knobCount(net), 15);
  assert.deepEqual(ids.slice(0, 6).map(i => i.name), ['w11', 'w12', 'w21', 'w22', 'b1', 'b2']);
  assert.deepEqual(ids.slice(12).map(i => [i.layer, i.name]), [[-1, 'v1'], [-1, 'v2'], [-1, 'c']]);
  assert.equal(LB.knobLabel(ids[1]), 'w₁₂'); assert.equal(LB.knobLabel(ids[14]), 'c');
  LB.setKnob(net, { layer: 1, k: 2 }, 0.25); assert.equal(net.layers[1].W[2], 0.25);
  LB.setKnob(net, { layer: 0, k: 5 }, -0.5); assert.equal(net.layers[0].b[1], -0.5);
  LB.setKnob(net, { layer: -1, k: 1 }, 1.5); assert.equal(net.v[1], 1.5);
  LB.setKnob(net, { layer: -1, k: 2 }, 0.75); assert.equal(net.c, 0.75);
  assert.equal(LB.getKnob(net, { layer: 1, k: 2 }), 0.25);
  assert.equal(LB.knobCount(LB.initNet(0, 'tanh', 1)), 3);
});

test('forward-difference slopes at delta 0.01 are within 1e-2 of backprop, and cost knobCount + 1 losses', () => {
  for (const [name, layers, act, seed] of [['blobs', 0, 'tanh', 1], ['moons', 1, 'tanh', 1], ['moons', 3, 'tanh', 1], ['spirals', 4, 'leaky', 1]]) {
    const d = LB.makeData(name), net = LB.initNet(layers, act, seed), before = knobs(net);
    const num = LB.numericGradient(net, d, 0.01), bp = LB.flatGradient(net, LB.lossGrad(net, d).g);
    assert.equal(num.evals, LB.knobCount(net) + 1);
    assert.ok(Math.abs(num.base - LB.lossOnly(net, d)) < 1e-12);
    for (let j = 0; j < bp.length; j++) assert.ok(Math.abs(num.slopes[j] - bp[j]) <= 1e-2, name + ' knob ' + j + ': ' + num.slopes[j] + ' vs ' + bp[j]);
    assert.deepEqual(knobs(net), before, 'numericGradient must put every knob back');
  }
});

test('nudge and lossCurve measure without moving the knob', () => {
  const d = LB.makeData('moons'), net = LB.initNet(1, 'tanh', 1), id = { layer: 0, k: 0 }, old = LB.getKnob(net, id);
  const r = LB.nudge(net, d, id, 0.1);
  assert.equal(LB.getKnob(net, id), old);
  assert.ok(Math.abs(r.slope - (r.after - r.before) / 0.1) < 1e-12);
  LB.setKnob(net, id, old + 0.1); assert.ok(Math.abs(LB.lossOnly(net, d) - r.after) < 1e-12); LB.setKnob(net, id, old);
  const c = LB.lossCurve(net, d, id, 2, 61);
  assert.equal(c.xs.length, 61); assert.equal(c.ys.length, 61);
  assert.ok(Math.abs(c.xs[0] - (old - 2)) < 1e-12 && Math.abs(c.xs[60] - (old + 2)) < 1e-12 && Math.abs(c.xs[30] - old) < 1e-12);
  assert.ok(Math.abs(c.ys[30] - r.before) < 1e-12);
  assert.equal(LB.getKnob(net, id), old);
});

test('a step moves every knob by -lr times its slope; gdStep is a step along backprop', () => {
  const d = LB.makeData('xor'), net = LB.initNet(3, 'tanh', 2), a = LB.cloneNet(net);
  const slopes = LB.flatGradient(net, LB.lossGrad(net, d).g), before = knobs(net);
  LB.stepAlong(net, slopes, 0.3);
  knobs(net).forEach((w, j) => assert.ok(Math.abs(w - (before[j] - 0.3 * slopes[j])) < 1e-12));
  const r = LB.gdStep(a, d, 0.3);
  assert.deepEqual(knobs(a), knobs(net));
  assert.ok(Math.abs(r.loss - LB.lossOnly(LB.cloneNet(LB.initNet(3, 'tanh', 2)), d)) < 1e-12);
  const b = LB.initNet(3, 'tanh', 2), c = LB.initNet(3, 'tanh', 2), opt = LB.makeOptimizer(c);
  LB.trainStep(b, null, d, 0.3, 'gd'); assert.deepEqual(knobs(b), knobs(net));
  LB.trainStep(c, opt, d, 0.02, 'adam'); assert.equal(opt.head.t, 1);
  assert.deepEqual(LB.RULES, ['gd', 'adam']);
});

test('resetOptimizerFor clears one layer or the readout only', () => {
  const d = LB.makeData('moons'), net = LB.initNet(2, 'tanh', 1), opt = LB.makeOptimizer(net);
  for (let i = 0; i < 5; i++) LB.adamStep(net, opt, d, 0.02);
  LB.resetOptimizerFor(opt, 1);
  assert.equal(opt.layers[1].t, 0); assert.equal(opt.layers[0].t, 5); assert.equal(opt.head.t, 5);
  LB.resetOptimizerFor(opt, -1);
  assert.equal(opt.head.t, 0); assert.ok(opt.head.m.every(v => v === 0));
});

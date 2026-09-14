'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const LB = require('./load.js')();

test('decompose then compose gives W back, including det < 0 and near-isotropic', () => {
  const rng = LB.rngOf('decomp');
  const mats = [[1,0,0,1], [0,-1,1,0], [2,0,0,-3], [1,1,1,1], [0,0,0,0], [-1,0,0,-1], [1,0,0,-1], [0,2,2,0]];
  for (let i = 0; i < 1000; i++) mats.push([0,0,0,0].map(() => 4*rng() - 2));
  for (const m of mats) {
    const d = LB.decompose(Float64Array.from(m)), w = LB.compose(d);
    for (let k = 0; k < 4; k++) assert.ok(Math.abs(w[k] - m[k]) < 1e-9, m.join(',') + ' -> ' + Array.from(w).join(','));
    assert.ok(d.s1 >= 0 && d.s2 >= 0, 'nonnegative stretch ' + m.join(','));
    assert.equal(d.flip, m[0]*m[3] - m[1]*m[2] < 0, 'flip iff det < 0 ' + m.join(','));
    assert.ok(Math.abs(d.theta1) <= Math.PI + 1e-12 && Math.abs(d.theta2) <= Math.PI + 1e-12);
  }
});

test('a pure rotation is one rotation, not two halves', () => {
  const a = 40 * Math.PI / 180;
  const d = LB.decompose(Float64Array.from([Math.cos(a), -Math.sin(a), Math.sin(a), Math.cos(a)]));
  assert.ok(Math.abs(d.theta1) < 1e-9 && Math.abs(d.theta2 - a) < 1e-9 && !d.flip, JSON.stringify(d));
  const s = LB.decompose(Float64Array.from([2, 0, 0, 0.5]));
  assert.ok(Math.abs(s.theta1) < 1e-9 && Math.abs(s.theta2) < 1e-9 && Math.abs(s.s1 - 2) < 1e-9 && Math.abs(s.s2 - 0.5) < 1e-9, JSON.stringify(s));
});

test('the chosen split has the least total rotation among the 90-degree variants', () => {
  const rng = LB.rngOf('minrot');
  for (let i = 0; i < 300; i++) {
    const m = Float64Array.from([0,0,0,0].map(() => 4*rng() - 2)), d = LB.decompose(m);
    const vs = LB._variants(m);
    assert.ok(vs.length >= 4);
    for (const alt of vs) {
      const w = LB.compose(alt);
      for (let k = 0; k < 4; k++) assert.ok(Math.abs(w[k] - m[k]) < 1e-9, 'variant must also rebuild W');
      assert.ok(Math.abs(d.theta1) + Math.abs(d.theta2) <= Math.abs(alt.theta1) + Math.abs(alt.theta2) + 1e-9);
    }
  }
});

test('every step at t = 1 in order equals the forward pass; t = 0 leaves points alone', () => {
  for (const act of ['tanh', 'leaky', 'relu']) {
    const net = LB.initNet(4, act, 3), steps = LB.buildSteps(net);
    assert.equal(steps[steps.length - 1].kind, 'readout');
    const perLayer = steps.filter(s => s.layer === 0).map(s => s.kind);
    assert.deepEqual(perLayer.filter(k => k !== 'flip'), ['rotate', 'stretch', 'rotate', 'shift', 'act']);
    const d = LB.makeData('spirals'), tr = LB.trace(steps, d.X), last = tr[tr.length - 1];
    assert.equal(tr.length, steps.length + 1);
    for (let i = 0; i < 200; i++) {
      const h = LB.forward(net, d.X[2*i], d.X[2*i+1]);
      assert.ok(Math.abs(last[2*i] - h[0]) < 1e-9 && Math.abs(last[2*i+1] - h[1]) < 1e-9, act);
    }
    for (const s of steps) {
      const p = LB.applyStep(s, 0, 0.3, -0.7);
      assert.ok(Math.abs(p[0] - 0.3) < 1e-12 && Math.abs(p[1] + 0.7) < 1e-12, s.kind);
    }
  }
});

test('flip appears exactly for layers whose W has a negative determinant', () => {
  const net = LB.initNet(8, 'tanh', 11), steps = LB.buildSteps(net);
  net.layers.forEach((L, i) => {
    const det = L.W[0]*L.W[3] - L.W[1]*L.W[2];
    assert.equal(steps.some(s => s.layer === i && s.kind === 'flip'), det < 0);
  });
});

test('describe() reads the numbers the way the side panel shows them', () => {
  assert.equal(LB.describe({ kind: 'rotate', angle: 31.7 * Math.PI / 180 }), 'rotate 31.7° counterclockwise');
  assert.equal(LB.describe({ kind: 'rotate', angle: -12.4 * Math.PI / 180 }), 'rotate 12.4° clockwise');
  assert.equal(LB.describe({ kind: 'rotate', angle: 0.0004 }), 'rotate 0.0°');
  assert.equal(LB.describe({ kind: 'flip' }), 'flip across the horizontal axis');
  assert.equal(LB.describe({ kind: 'stretch', s1: 1.62, s2: 0.83 }), 'stretch ×1.62 along x, ×0.83 along y');
  assert.equal(LB.describe({ kind: 'shift', bx: 0.3, by: -0.1 }), 'shift by (0.30, −0.10)');
  assert.equal(LB.describe({ kind: 'act', act: 'tanh' }), 'tanh: squash each coordinate into (−1, 1)');
});

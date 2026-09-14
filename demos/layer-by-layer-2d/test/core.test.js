'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const LB = require('./load.js')();

test('datasets: 200 points, 100 per class, inside [-1.5, 1.5], u in [0, 1]', () => {
  for (const name of ['moons', 'xor', 'spirals']) {
    const d = LB.makeData(name);
    assert.equal(d.n, 200);
    assert.equal(d.y.reduce((s, v) => s + v, 0), 100);
    for (let i = 0; i < 400; i++) assert.ok(Math.abs(d.X[i]) < 1.5, name + ' ' + d.X[i]);
    for (let i = 0; i < 200; i++) assert.ok(d.u[i] >= 0 && d.u[i] <= 1, name + ' u ' + d.u[i]);
  }
  assert.deepEqual(LB.makeData('moons').X, LB.makeData('moons').X);   // seeded
});

test('backprop gradient matches a central difference', () => {
  for (const act of ['tanh', 'leaky', 'relu']) {
    const d = LB.makeData('spirals'), net = LB.initNet(3, act, 7);
    const g = LB.lossGrad(net, d).g, h = 1e-6;
    const check = (arr, ga, k, what) => {
      const old = arr[k];
      arr[k] = old + h; const lp = LB.lossGrad(net, d).loss;
      arr[k] = old - h; const lm = LB.lossGrad(net, d).loss;
      arr[k] = old;
      const num = (lp - lm) / (2*h);
      assert.ok(Math.abs(num - ga[k]) < 1e-5, act + ' ' + what + '[' + k + '] ' + num + ' vs ' + ga[k]);
    };
    net.layers.forEach((L, i) => {
      for (let k = 0; k < 4; k++) check(L.W, g.layers[i].W, k, 'W' + i);
      for (let k = 0; k < 2; k++) check(L.b, g.layers[i].b, k, 'b' + i);
    });
    for (let k = 0; k < 2; k++) check(net.v, g.v, k, 'v');
    const box = [net.c], old = net.c;
    net.c = old + h; const lp = LB.lossGrad(net, d).loss;
    net.c = old - h; const lm = LB.lossGrad(net, d).loss;
    net.c = old;
    assert.ok(Math.abs((lp - lm) / (2*h) - g.c) < 1e-5, act + ' c');
    assert.equal(box.length, 1);
  }
});

test('adam lowers the loss on moons', () => {
  const net = LB.initNet(2, 'tanh', 1), opt = LB.makeOptimizer(net), d = LB.makeData('moons');
  const first = LB.lossGrad(net, d).loss;
  let r;
  for (let i = 0; i < 300; i++) r = LB.adamStep(net, opt, d, 0.03);
  assert.ok(r.loss < 0.5 * first, first + ' -> ' + r.loss);
});

test('each preset reaches its target accuracy from its own seed with its default update rule', () => {
  assert.deepEqual(Object.keys(LB.PRESETS), ['blobs', 'moons', 'xor', 'spirals']);
  for (const name of Object.keys(LB.PRESETS)) {
    const p = LB.PRESETS[name], d = LB.makeData(name);
    assert.ok(LB.RULES.includes(p.opt) && p.lr.gd > 0 && p.lr.adam > 0, name + ' preset shape');
    const net = LB.initNet(p.layers, p.act, p.seed), opt = LB.makeOptimizer(net);
    let reached = null;
    for (let s = 0; s < p.steps && reached === null; s++) if (LB.trainStep(net, opt, d, p.lr[p.opt], p.opt).acc >= p.target) reached = s;
    assert.ok(reached !== null, name + ' did not reach ' + p.target + ' in ' + p.steps + ' steps with ' + p.opt);
    console.log('# preset', name, p.opt, 'lr', p.lr[p.opt], 'reached', p.target, 'at step', reached);
  }
});

test('add layer keeps old weights and grows the optimizer; remove undoes it', () => {
  const net = LB.initNet(2, 'tanh', 1), opt = LB.makeOptimizer(net), d = LB.makeData('moons');
  for (let i = 0; i < 20; i++) LB.adamStep(net, opt, d, 0.03);
  const W0 = Array.from(net.layers[0].W);
  LB.addLayer(net, 5); LB.syncOptimizer(net, opt);
  assert.equal(net.layers.length, 3); assert.equal(opt.layers.length, 3);
  assert.deepEqual(Array.from(net.layers[0].W), W0);
  const W2 = net.layers[2].W;
  assert.ok(Math.abs(W2[0] - 1) < 0.2 && Math.abs(W2[1]) < 0.2 && Math.abs(W2[2]) < 0.2 && Math.abs(W2[3] - 1) < 0.2);
  const r = LB.adamStep(net, opt, d, 0.03);
  assert.ok(Number.isFinite(r.loss));
  LB.removeLayer(net); LB.syncOptimizer(net, opt);
  assert.equal(net.layers.length, 2); assert.equal(opt.layers.length, 2);
});

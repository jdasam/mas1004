'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('crypto');
const LB = require('./load.js')();
const sha = a => crypto.createHash('sha256').update(Buffer.from(a.buffer, a.byteOffset, a.byteLength)).digest('hex');

test('training points of moons, xor, spirals are unchanged', () => {
  const Y = 'b93d3105e4863a1e3cab2571bdff746d1364bf6c65a4d8ab043389c43ed622a2';
  const want = {
    moons:   ['703953bbc206686e45c4c80bb0b3c9caae7ef1952e6f6f509cddd5a9db75369c', '37051d1306f25c01bbc7a20911b39e02da51952668ea4ab908e6b84b3b26a336'],
    xor:     ['033f57c6c0ce4077461ddf230df516c061cb456c9edecea7972313224f1af980', '55b4bf9f56f9c7430ab4593c45a67ca5f7202cb5af58a5899a7c8a79bf492da2'],
    spirals: ['4f7ef444232983e077e30c29511656dccb72bc392d0aa0bc96ef652938f75069', 'd45236c8b9376d3b6151ff4c36614e04bdebda719aa999325c1233e272b3f358'],
  };
  for (const [name, [x, u]] of Object.entries(want)) {
    const d = LB.makeData(name);
    assert.equal(sha(d.X), x, name + ' X'); assert.equal(sha(d.u), u, name + ' u'); assert.equal(sha(d.y), Y, name + ' y');
  }
});

test('blobs come first, are seeded, and the rule x > 0 gets at least 97%', () => {
  assert.deepEqual(Object.keys(LB.DATASETS), ['blobs', 'moons', 'xor', 'spirals']);
  const rule = d => { let ok = 0; for (let i = 0; i < d.n; i++) if ((d.X[2*i] > 0) === (d.y[i] === 1)) ok++; return ok / d.n; };
  const d = LB.makeData('blobs'), t = LB.makeTestData('blobs');
  assert.equal(d.n, 200); assert.equal(d.y.reduce((s, v) => s + v, 0), 100);
  assert.deepEqual(d.X, LB.makeData('blobs').X);
  for (let i = 0; i < 400; i++) assert.ok(Math.abs(d.X[i]) < 1.5);
  for (let i = 0; i < 200; i++) assert.ok(d.u[i] >= 0 && d.u[i] <= 1);
  assert.ok(rule(d) >= 0.97, 'train ' + rule(d)); assert.ok(rule(t) >= 0.97, 'test ' + rule(t));
  console.log('# rule x > 0 on blobs: train', rule(d), 'test', rule(t), '| on moons', rule(LB.makeData('moons')));
});

test('test points: 50 per class, seeded, never on top of a training point', () => {
  for (const name of Object.keys(LB.DATASETS)) {
    const d = LB.makeData(name), t = LB.makeTestData(name);
    assert.equal(t.n, 100); assert.equal(t.y.reduce((s, v) => s + v, 0), 50);
    for (let i = 0; i < t.n; i++) assert.equal(t.y[i], i % 2);
    assert.deepEqual(t.X, LB.makeTestData(name).X);
    for (let i = 0; i < t.n; i++) {
      let best = Infinity;
      for (let j = 0; j < d.n; j++) best = Math.min(best, Math.hypot(t.X[2*i] - d.X[2*j], t.X[2*i+1] - d.X[2*j+1]));
      assert.ok(best > 1e-6, name + ' test point ' + i + ' repeats a training point');
    }
  }
});

test('zero layers: the readout acts on the input, one readout step, remove stops at 0', () => {
  const net = LB.initNet(0, 'tanh', 1), d = LB.makeData('blobs');
  assert.equal(net.layers.length, 0);
  assert.deepEqual(LB.forward(net, 0.3, -0.7), [0.3, -0.7]);
  assert.ok(Math.abs(LB.logit(net, 0.3, -0.7) - (net.v[0]*0.3 - net.v[1]*0.7 + net.c)) < 1e-12);
  let want = 0;
  for (let i = 0; i < d.n; i++) {
    const p = 1 / (1 + Math.exp(-(net.v[0]*d.X[2*i] + net.v[1]*d.X[2*i+1] + net.c)));
    want -= Math.log(d.y[i] ? p : 1 - p);
  }
  assert.ok(Math.abs(LB.lossGrad(net, d).loss - want / d.n) < 1e-9);
  const steps = LB.buildSteps(net);
  assert.deepEqual(steps.map(s => s.kind), ['readout']);
  assert.equal(LB.trace(steps, d.X).length, 2);
  LB.addLayer(net, 3); assert.equal(net.layers.length, 1);
  LB.removeLayer(net); LB.removeLayer(net); assert.equal(net.layers.length, 0);
});

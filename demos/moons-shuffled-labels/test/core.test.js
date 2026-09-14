'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const load = require('./load.js');
const MS = load();
const LB = load.loadFrom(path.join(__dirname, '..', '..', 'layer-by-layer-2d', 'index.html'), 'lb-core', 'LB');

test('training points are the layer-by-layer demo moons', () => {
  const tr = MS.makeMoons('train'), a = LB.makeData('moons');
  assert.equal(tr.n, 200);
  assert.deepEqual(tr.X, a.X); assert.deepEqual(tr.y, a.y);
});

/* 층별 데모의 테스트 점(과제 A1)이 아직 합쳐지지 않은 작업 사본에서만 건너뛴다. 합친 뒤에는 반드시 돈다. */
test('test points are the layer-by-layer demo test points',
  { skip: typeof LB.makeTestData !== 'function' && 'layer-by-layer demo has no makeTestData yet' }, () => {
  const te = MS.makeMoons('test'), b = LB.makeTestData('moons');
  assert.equal(te.n, 100);
  assert.deepEqual(te.X, b.X); assert.deepEqual(te.y, b.y);
});

test('shuffled labels: seeded, same class counts, really shuffled', () => {
  const y = MS.makeMoons('train').y, s = MS.shuffleLabels(y, MS.SHUFFLE_SEED);
  assert.deepEqual(s, MS.shuffleLabels(y, MS.SHUFFLE_SEED));
  assert.equal(s.reduce((t, v) => t + v, 0), 100);
  let moved = 0; for (let i = 0; i < y.length; i++) if (s[i] !== y[i]) moved++;
  assert.ok(moved >= 60, 'only ' + moved + ' labels changed');
  assert.notDeepEqual(MS.shuffleLabels(y, 2), s);
  assert.equal(y[1], 1, 'the original labels are untouched');
});

test('backprop matches a central difference for every parameter', () => {
  const tr = MS.makeMoons('train'), data = { n: tr.n, X: tr.X, y: MS.shuffleLabels(tr.y, 1) };
  const net = MS.initNet(5, 3), g = MS.lossGrad(net, data).g, h = 1e-6;
  MS.params(net).forEach((p, q) => {
    for (let k = 0; k < p.length; k++) {
      const old = p[k];
      p[k] = old + h; const lp = MS.evaluate(net, data).loss;
      p[k] = old - h; const lm = MS.evaluate(net, data).loss;
      p[k] = old;
      assert.ok(Math.abs((lp - lm) / (2*h) - g[q][k]) < 1e-5, 'param ' + q + '[' + k + ']');
    }
  });
});

test('both panels start from the same knobs, and Adam lowers the loss', () => {
  const a = MS.initNet(16, 7), b = MS.initNet(16, 7);
  MS.params(a).forEach((p, q) => assert.deepEqual(p, MS.params(b)[q]));
  const tr = MS.makeMoons('train'), st = MS.makeAdam(a), first = MS.evaluate(a, tr).loss;
  let r; for (let i = 0; i < 200; i++) r = MS.adamStep(a, st, tr, MS.PRESET.lr);
  assert.ok(r.loss < 0.5 * first, first + ' -> ' + r.loss);
});

test('en and ko have the same keys and the same {placeholders}', () => {
  const en = MS.STRINGS.en, ko = MS.STRINGS.ko;
  assert.deepEqual(Object.keys(ko).sort(), Object.keys(en).sort());
  const holes = s => (s.match(/\{\w+\}/g) || []).sort();
  for (const k of Object.keys(en)) {
    assert.equal(typeof ko[k], 'string', k);
    assert.deepEqual(holes(ko[k]), holes(en[k]), k);
    assert.ok(!/\u2014/.test(en[k] + ko[k]), 'no em dash in ' + k);
  }
});

test('t() fills placeholders and falls back to English', () => {
  assert.equal(MS.t('en', 'note.max', { n: '13,000' }), 'Paused after 13,000 steps.');
  assert.equal(MS.t('ko', 'note.max', { n: '13,000' }), '13,000걸음에서 멈췄다.');
  assert.equal(MS.t('ko', 'panel.shuf'), '섞은 라벨');
  assert.equal(MS.t('fr', 'ctl.train'), 'Train');
  assert.equal(MS.t('ko', 'no.such.key'), 'no.such.key');
});

test('width 64 fits real and shuffled labels; width 4 cannot fit shuffled labels', () => {
  const tr = MS.makeMoons('train'), shuf = { n: tr.n, X: tr.X, y: MS.shuffleLabels(tr.y, MS.SHUFFLE_SEED) };
  const fit = (H, data, steps) => {
    const net = MS.initNet(H, 1), st = MS.makeAdam(net);
    for (let s = 0; s < steps; s++) if (MS.adamStep(net, st, data, MS.PRESET.lr).acc >= 1) return s;
    return null;
  };
  const real64 = fit(64, tr, MS.PRESET.steps), shuf64 = fit(64, shuf, MS.PRESET.steps), shuf4 = fit(4, shuf, MS.PRESET.steps);
  console.log('# width 64 real', real64, 'shuffled', shuf64, '| width 4 shuffled', shuf4);
  assert.ok(real64 !== null && shuf64 !== null); assert.equal(shuf4, null);
});

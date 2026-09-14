'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const LB = require('./load.js')();

test('en and ko have the same keys and the same {placeholders}', () => {
  const en = LB.STRINGS.en, ko = LB.STRINGS.ko;
  assert.deepEqual(Object.keys(ko).sort(), Object.keys(en).sort());
  const holes = s => (s.match(/\{\w+\}/g) || []).sort();
  for (const k of Object.keys(en)) {
    assert.equal(typeof ko[k], 'string', k);
    assert.deepEqual(holes(ko[k]), holes(en[k]), k);
    assert.ok(!/—/.test(en[k] + ko[k]), 'no em dash in ' + k);
  }
});

test('t() fills placeholders and falls back to English', () => {
  assert.equal(LB.t('en', 'cap.layer', { n: 2, L: 3 }), 'Layer 2 of 3');
  assert.equal(LB.t('ko', 'cap.layer', { n: 2, L: 3 }), '층 2 / 3');
  assert.equal(LB.t('fr', 'tab.knobs'), 'Knobs');
  assert.equal(LB.t('ko', 'no.such.key'), 'no.such.key');
});

test('describe() reads the steps in Korean', () => {
  const deg = d => d * Math.PI / 180;
  assert.equal(LB.describe({ kind: 'rotate', angle: deg(38.4) }, 'ko'), '반시계 방향으로 38.4° 회전');
  assert.equal(LB.describe({ kind: 'rotate', angle: deg(-6.6) }, 'ko'), '시계 방향으로 6.6° 회전');
  assert.equal(LB.describe({ kind: 'rotate', angle: 0.0004 }, 'ko'), '0.0° 회전');
  assert.equal(LB.describe({ kind: 'flip' }, 'ko'), '가로축을 기준으로 뒤집기');
  assert.equal(LB.describe({ kind: 'stretch', s1: 1.45, s2: 1.02 }, 'ko'), 'x 방향 ×1.45, y 방향 ×1.02 늘이기');
  assert.equal(LB.describe({ kind: 'shift', bx: 0.3, by: -0.1 }, 'ko'), '(0.30, −0.10)만큼 이동');
  assert.equal(LB.describe({ kind: 'act', act: 'tanh' }, 'ko'), 'tanh: 좌표마다 (−1, 1)로 누르기');
  assert.equal(LB.describe({ kind: 'readout' }, 'ko'), '두 종류를 나누는 직선 v·h + c = 0 긋기');
  assert.equal(LB.describe({ kind: 'rotate', angle: deg(31.7) }), 'rotate 31.7° counterclockwise');
});

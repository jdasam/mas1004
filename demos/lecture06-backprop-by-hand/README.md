# 6차시 데모: Backprop by Hand (역전파를 손으로 따라가기)

MAS1004 6차시 수업용 인터랙티브 웹 데모. 5차시 Neuron Patch의 작은 프리셋(1 neuron, 1 ReLU, 3 ReLU)을 곱하기, 더하기, ReLU, 빼기, 제곱의 낱개 연산으로 펼친 계산 그래프에서,
데이터 점 하나에 대해 Forward로 값을 한 칸씩 채우고 Backward로 손실의 slope를 한 칸씩 채운다. 학생은 다음 칸을 먼저 암산하고 버튼을 눌러 확인한다.
4차시의 노브 하나씩 밀어 재기(Nudge each knob)를 옆에 두어 숫자가 거의 같고 손실 계산 횟수가 다른 것을 본다.
설계 문서: `docs/superpowers/specs/2026-09-17-lecture06-demos-design.md` 3절.

의존성 없는 단일 파일이다. `index.html` 하나를 브라우저로 열면 되고(`file://` 포함), 빌드 단계도 외부 네트워크 요청도 없다. UI는 영어다.

## 학생에게 배포하는 법

1. GitHub Pages (기본): `mas1004-2026/tools/publish_demo.sh lecture06-backprop-by-hand`를 실행하면 이 폴더가 서브모듈 `mas1004/demos/`로 복사되고 push까지 된다(`test/`는 복사에서 빠진다).
   학생용 링크: https://jdasam.github.io/mas1004/demos/lecture06-backprop-by-hand/ (데모 목록: https://jdasam.github.io/mas1004/demos/).
2. LMS에 첨부: `index.html`만 올린다.
3. 수업 중 임시 서버: 이 폴더에서 `python -m http.server 8000`.

## 수업에서 쓰는 법

| 순서 | 프리셋 | 하는 일 |
| --- | --- | --- |
| 1 | 1 neuron | ŷ = w·x + b, 점 (2, 3). Forward 네 번: wx = −1, ŷ = −1, e = −4, loss = 16. Backward 다섯 번: loss 1, e −8, ŷ −8, w −16, b −8. 규칙 표의 ², +, × 행이 차례로 켜진다 |
| 2 | 1 ReLU | z₁ = 2, h₁ = 2, ŷ = 0, loss = 9. Backward: e −6, ŷ −6, w₂ −12, b₂ −6, h₁ 3, z₁ 3, w₁ 6, b₁ 3. 학생이 다음 칸을 먼저 말하게 한다 |
| 3 | 3 ReLU | z₂ = −2라 N2의 ReLU가 닫혀 z₂, w₂, b₂의 slope가 0이다(죽은 뉴런). loss = 0.01, slope가 작다 |
| 4 | 아무 프리셋 | Nudge each knob: 노브마다 Δ = 0.001만큼 밀어 잰 slope가 backprop 열과 거의 같다. 손실 계산 횟수는 노브 수 + 1 대 forward 1번 + backward 1번 |
| 5 | 아무 프리셋 | Take a step: 모든 노브를 w ← w − lr × slope로 옮긴다. 값과 slope 칸은 모두 ?로 돌아가므로 Forward를 다시 눌러 새 값을 계산하고(손실이 내려간 것을 본다) Backward로 새 slope를 잰다. Forward all, Backward all, Take a step을 반복한다 |

- 노브 상자를 위아래로 끌면 0.1 단위(Shift로 0.01), 누르면 입력. 노브나 점을 바꾸면 값은 바로 다시 계산되고 slope는 모두 지워진다(4차시 규칙).
- 플롯의 데이터 점을 누르면 그 점이 고른 점이 된다. x, y 상자로 직접 입력할 수도 있다. 기본 점 (2, 3)은 데이터의 x = 2 근처 점(y ≈ 3.1)과 가깝다.
- 손실은 점 하나의 (ŷ − y)²이다. Neuron Patch의 손실은 이것을 120점에서 평균한 것이다.

## URL 옵션

`?preset=relu1&x=2&y=3&lr=0.02`

| 옵션 | 값 |
| --- | --- |
| preset | line, relu1, relu3 (기본 relu1) |
| x, y | 고른 점 |
| lr | learning rate (기본 0.02. 0.05는 3 ReLU 프리셋의 기본 점에서 목표를 지나쳐 손실이 는다) |

## 구조와 테스트

한 파일 안에 스크립트가 둘이다. `<script id="bp-core">`(전역 `BP`)는 데이터(4차시, 5차시와 같은 봉우리 120점), 프리셋 그래프, forward와 backward(부분 계산 포함), 설명 문장, nudge, 한 걸음을 갖고 DOM을 건드리지 않는다. `<script id="bp-ui">`(전역 `BPUI`)는 SVG 그래프, 플롯, 숫자 상자, 버튼, 표, URL 옵션을 맡는다.

캡처용 `window.BPUI.hooks`: `loadPreset(key)`, `forwardStep()`, `forwardAll()`, `backwardStep()`, `backwardAll()`, `nudge()`, `takeStep()`, `setKnob(id, v)`, `setPoint(x, y)`, `reset()`, `draw()`, `state()`. 노브 id는 `w`, `b` (1 neuron), `w1`, `b1`, `w2`, `b2` (1 ReLU), `w1`…`w3`, `b1`…`b3`, `v1`…`v3`, `c` (3 ReLU). 뷰포트 1600×1000, `device_scale_factor=2`를 권한다.

```bash
node --test test/*.test.js                       # 코어 (Neuron Patch 코어와 데이터 대조 포함)
uv run --with playwright python test/smoke.py    # 브라우저 스모크, 캡처는 test/shots/
```

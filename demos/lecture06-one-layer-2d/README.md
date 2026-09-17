# 6차시 데모: One Layer in 2D (행렬, 편향, 활성화가 평면에 하는 일)

MAS1004 6차시 수업용 인터랙티브 웹 데모. 2×2 행렬 W의 숫자 넷을 돌리면 평면 전체가 한꺼번에 옮겨지는 것을 격자와 글자 F로 본다.
행 벡터 표기 `[x y] × W`를 쓰므로 (1, 0)이 가는 곳은 W의 1행, (0, 1)이 가는 곳은 2행이다.
`+ b`를 켜면 평면이 통째로 옮겨지고, `act( )`를 켜면 ReLU가 두 축에서 접거나 tanh가 (−1, 1) 상자 안으로 누른다.
설계 문서: `docs/superpowers/specs/2026-09-17-lecture06-demos-design.md` 2절.

의존성 없는 단일 파일이다. `index.html` 하나를 브라우저로 열면 되고(`file://` 포함), 빌드 단계도 외부 네트워크 요청도 없다. UI는 영어다.

## 학생에게 배포하는 법

1. GitHub Pages (기본): `mas1004-2026/tools/publish_demo.sh lecture06-one-layer-2d`를 실행하면 이 폴더가 서브모듈 `mas1004/demos/`로 복사되고 push까지 된다(`test/`는 복사에서 빠진다).
   학생용 링크: https://jdasam.github.io/mas1004/demos/lecture06-one-layer-2d/ (데모 목록: https://jdasam.github.io/mas1004/demos/).
2. LMS에 첨부: `index.html`만 올린다.
3. 수업 중 임시 서버: 이 폴더에서 `python -m http.server 8000`.

## 수업에서 쓰는 법

1. Identity에서 시작해 Rotate 30°, Rotate 90°, Stretch, Squeeze, Shear, Flip을 차례로 누른다. 격자가 곧고 평행하고 원점이 그대로인 것, 주황 화살표 끝의 숫자가 W의 1행과 같은 것을 본다.
2. 숫자 상자를 위아래로 끌어(0.1 단위, Shift로 0.01) 격자가 따라오는 것을 본다. 누르면 숫자를 직접 입력할 수 있다.
3. 속 빈 점 p를 끌면 오른쪽 아래 계산 패널의 숫자가 바로 바뀐다. 5차시의 행 × 열 계산 그대로다.
4. `+ b`를 켜고 b를 돌린다. 원점이 b로 옮겨진다.
5. `act( )`를 켜고 ReLU를 고른다. 가로축 아래로 내려간 F의 세로 획이 축 위로 접혀 붙는다. tanh는 격자를 (−1, 1) 상자 안으로 누른다. Linear는 아무것도 하지 않는다.
6. ▶ Play(또는 Space)는 원래 격자에서 시작해 켜진 단계를 차례로 재생한다(× W 1.2초, + b 0.6초, act 0.8초).

## 프리셋

| 프리셋 | W (행 순서) | 보이는 것 |
| --- | --- | --- |
| Identity | [[1, 0], [0, 1]] | 그대로 |
| Rotate 30° | [[0.87, 0.5], [−0.5, 0.87]] | 반시계 30° |
| Rotate 90° | [[0, 1], [−1, 0]] | 반시계 90° |
| Stretch | [[2, 0], [0, 1]] | x 방향 2배 |
| Squeeze | [[0.5, 0], [0, 0.5]] | 절반 |
| Shear | [[1, 0], [1, 1]] | 위쪽이 오른쪽으로 밀린다 |
| Flip | [[1, 0], [0, −1]] | y가 −y로 |
| Random | 누를 때마다 다음 시드 | 성분 N(0, 0.8)을 0.1 단위로, 행렬식 절댓값 0.3 이상 |

## URL 옵션

`?preset=shear&stage=act&act=relu&p=2,1`

| 옵션 | 값 |
| --- | --- |
| preset | identity, rotate30, rotate90, stretch, squeeze, shear, flip |
| w | `w11,w12,w21,w22` (preset보다 우선) |
| b | `b1,b2` |
| stage | w, b, act (기본 w) |
| act | linear, relu, tanh (기본 relu) |
| p | `x,y` (기본 2,1) |

## 구조와 테스트

한 파일 안에 스크립트가 둘이다. `<script id="ol-core">`(전역 `OL`)는 프리셋, 무작위 행렬, 단계별 변환 `f(p; W, b, act, tW, tb, tact)`, 격자와 F 샘플링, 계산 패널 문자열을 갖고 DOM을 건드리지 않는다. `<script id="ol-ui">`(전역 `OLUI`)는 캔버스, 숫자 상자, 칩, 재생, URL 옵션을 맡는다.

캡처용 `window.OLUI.hooks`: `setW([[..],[..]])`, `setB([b1,b2])`, `setAct(key)`, `setStage(key)`, `loadPreset(key)`, `setPoint(x, y)`, `play()`, `seek(tW, tb, tact)`, `draw()`, `state()`. 뷰포트 1600×1000, `device_scale_factor=2`를 권한다.

```bash
node --test test/*.test.js                       # 코어
uv run --with playwright python test/smoke.py    # 브라우저 스모크, 캡처는 test/shots/
```

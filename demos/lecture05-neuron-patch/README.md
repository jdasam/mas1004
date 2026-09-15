# 5차시 데모: Neuron Patch (뉴런을 이어 붙여 만드는 모델)

MAS1004 5차시(뉴런으로 만든 모델) 수업용 인터랙티브 웹 데모. Max/MSP 패치처럼 뉴런 상자를 캔버스에 놓고
x에서 ŷ까지 선으로 이어 1차원 회귀 신경망을 만든다. 데이터는 4차시와 같은 봉우리 8개 데이터(점 120개)이고,
선 하나가 weight 노브 하나, 뉴런 하나가 bias 노브 하나다. 4차시의 Measure the gradient, Take a step, Run을
그대로 써서 손으로 맞추던 노브를 경사하강으로 맞춘다.

의존성 없는 단일 파일이다. `index.html` 하나를 브라우저로 열면 되고(`file://` 포함), GitHub Pages에 그 파일만 올려도 동작한다.
빌드 단계도 외부 네트워크 요청도 없다. UI는 영어다.

## 학생에게 배포하는 법

파일 하나라서 방법이 셋이다. 어느 쪽이든 학생은 링크를 열거나 파일을 더블클릭하면 된다.

1. **LMS에 첨부**: `index.html`을 사이버캠퍼스 자료실에 올린다. 학생은 내려받아 브라우저로 연다.
   `file://`로 열려도 모든 기능이 동작한다.
2. **GitHub Pages** (기본): 공개 저장소 `jdasam/mas1004`의 2026 브랜치가 Pages로 열려 있다.
   `mas1004-2026/tools/publish_demo.sh lecture05-neuron-patch`를 실행하면 이 폴더가 서브모듈 `mas1004/demos/`로 복사되고
   push까지 된다 (`test/`는 복사에서 빠진다). 학생용 링크: https://jdasam.github.io/mas1004/demos/lecture05-neuron-patch/
   (데모 목록: https://jdasam.github.io/mas1004/demos/). 이 상위 저장소(`sogang-course`)는 비공개라 Pages를 쓸 수 없다.
3. **수업 중 임시 서버**: 교실 PC에서 `python -m http.server 8000`을 이 폴더에서 띄우고 같은 네트워크의 학생에게
   `http://<교수 PC IP>:8000/`을 알려 준다.

## 수업에서 쓰는 법

위쪽 "Start from" 프리셋 버튼을 수업 순서대로 누른다. 프리셋은 누구에게나 같은 값으로 시작한다.
프리셋을 누르면 learning rate도 그 프리셋에 맞춘 값으로 바뀐다. Reset은 지금 프리셋의 시작 상태로 되돌린다.

| 순서 | 프리셋 | 학생이 하는 일 |
| --- | --- | --- |
| 1 | 1 neuron | x → ŷ 선 하나, ŷ는 Linear. weight가 기울기, ŷ의 bias가 절편이다. 3차시에 맞추던 직선과 같은 모델이라는 것을 확인한다 (노브 2개) |
| 2 | 1 ReLU | ReLU 뉴런 하나를 끼운다. N1의 bias를 돌리면 꺾이는 점이 좌우로 움직이고, weight 부호를 바꾸면 꺾이는 방향이 바뀐다. 뉴런 상자 안의 작은 곡선으로 확인한다 |
| 3 | 3 ReLU | 꺾이는 점이 −1.5, 0, 1.5에 있는 ReLU 세 개 (노브 10개). 손으로 노브를 돌려 손실을 가장 낮춘 사람을 뽑는다. "Show neurons"를 켜면 ŷ에 들어가는 뉴런별 기여(w × h(x))가 흐린 선으로 보인다 |
| 4 | 3 Linear | 3 ReLU와 같은 weight에서 활성화만 Linear. 뉴런을 몇 개 쌓아도 결과가 여전히 직선이라는 것을 본다. 인스펙터에서 한 뉴런만 ReLU로 바꾸면 바로 꺾인다 |
| 5 | 3 ReLU, 2 layers, Empty | Measure the gradient와 Take a step을 번갈아 누르고, Run으로 자동 반복한다. 뉴런을 더하거나 지우고 선을 다시 이어 가며 손실이 어디까지 내려가는지 본다. 2 layers는 tanh 4개씩 두 층 (노브 33개) |

- Measure the gradient를 누르면 노브마다 빨간 호와 slope 숫자가, 선의 숫자 상자 위에는 작은 빨간 slope가 붙는다.
  노브, 숫자 상자, 활성화, 연결 중 무엇이든 바꾸면 slope가 회색으로 바뀌고 Take a step이 꺼진다 (4차시와 같은 규칙).
- Run 중에도 노브를 돌리거나 선을 바꿀 수 있다. 바뀐 값에서 이어서 내려간다.
- 손실이 유한하지 않거나 10⁶을 넘으면 멈추고 "Diverged: the loss blew up"을 띄운다. learning rate를 낮추고 Reset한다.
- 선이 12개를 넘는 패치(2 layers)에서는 숫자 상자가 촘촘해서 상자 위 slope를 그리지 않는다. slope는 인스펙터 노브와 뉴런 상자의 bias 옆에 보인다.

## URL 옵션

| 옵션 | 용도 |
| --- | --- |
| `?preset=relu3` | 시작 프리셋 (`line`, `relu1`, `relu3`, `linear3`, `layers2`, `empty`). 기본은 `line` |
| `?lr=0.02` | learning rate 초기값. 프리셋 버튼을 누르면 그 프리셋의 값으로 다시 바뀐다 |
| `?opt=adam` | 숨은 옵션. 갱신 규칙을 Adam으로 바꾼다 (기본 learning rate 0.02, 프리셋을 눌러도 유지). 학생 화면에는 규칙 이름만 "update rule: Adam"으로 바뀐다 |

옵션은 `?`와 `#` 어느 쪽으로 줘도 된다: `index.html?preset=layers2#lr=0.05`.

## 데이터와 모델

- 데이터는 4차시 `GD.DATA`와 같은 점 120개다 (x ∈ [−3, 3], 숨은 함수는 봉우리 8개의 합에 표준편차 0.08 잡음,
  시드 `bump-truth|19`, `bump-data|19`, `DATA_VERSION = 1`). 난수 코드를 4차시에서 그대로 복사해 같은 숫자가 나온다.
- 모델은 방향이 있는 그래프다. 입력 `x`(bias 없음), 뉴런 `N1, N2, …`, 출력 `ŷ`(ŷ도 bias와 활성화를 가진 뉴런, 기본 Linear).
  뉴런 하나의 계산은 `z = b + Σ w·h(들어오는 노드)`, `h = f(z)`. 활성화 f는 Linear, ReLU, tanh, sigmoid, step 중 하나다.
- 노브 수 = 선 수 + (뉴런 수 + ŷ의 bias 1개).
- 새 선의 weight는 `rngOf('np|w|' + seq)`로 정해진다. 같은 순서로 이으면 모든 학생이 같은 값을 받는다 (부호 무작위, 크기 0.3 ~ 1.2).
- 고리가 생기는 연결, 같은 연결 두 번, 자기 자신, x로 들어가는 연결, ŷ에서 나가는 연결은 거절하고 짧은 메시지를 띄운다.

## 계산 방식

- 손실은 평균제곱오차 `L = (1/120) Σ (ŷ − y)²`.
- gradient는 역전파(reverse-mode accumulation)로 한 번에 계산한다. `dL/dŷ = 2(ŷ − y)/n`에서 시작해 위상 정렬의 역순으로 내려간다.
  수업에서는 이 이름을 쓰지 않는다. 화면의 slope는 4차시의 전진 차분(Δ = 0.001, `NP.numericGradient`)으로 잰 값과 거의 같다.
  코어 테스트가 두 방식을 비교한다.
- 갱신은 `w ← w − lr × slope`, bias도 같다. 내부 표준화는 없다.
- Run은 프레임마다 (gradient + 갱신)을 1, 10, 100번 반복한다. 손으로 Measure와 Take a step을 누르는 것과 같은 계산이다.
- 프리셋별 learning rate(`NP.PRESET_LR`)는 `test/calibrate.js`로 실측했다. line 0.1, relu1 0.1, relu3 0.05, linear3 0.05, layers2 0.1, empty 0.05 (Quiz B1: 0.1에서는 연결 순서에 따라 ReLU가 죽어 직선으로 끝났다).
  learning rate 슬라이더는 로그 눈금으로 10⁻⁵에서 10까지다.

## 구조

한 파일 안에 스크립트가 두 개다.

- `<script id="np-core">`: 난수, 데이터, 활성화, 그래프 편집(뉴런, 선, 연결 검사), 순전파, 손실, gradient, 수치 gradient,
  경사하강, Adam, 프리셋, 실측 상수. DOM을 건드리지 않는다. 전역 `NP`.
- `<script id="np-ui">`: SVG 패치(뉴런 상자, 선, 숫자 상자, 드래그 연결), 인스펙터 노브(4차시 노브 위젯), 캔버스 산점도와 손실 곡선,
  학습 막대. 전역 `NPUI` (`NPUI.hooks`는 테스트용).

코어는 `index.html` 안에 인라인되어 있다. 테스트(`test/load.js`)와 `test/calibrate.js`는 이 스크립트를 꺼내 Node에서 실행한다.

코어만 Node에서 실행하려면:

```js
const fs = require('fs');
const html = fs.readFileSync('index.html', 'utf8');
const NP = (0, eval)(html.match(/<script id="np-core">([\s\S]*?)<\/script>/)[1] + '\n;NP');
const g = NP.buildPreset('relu3');
NP.knobCount(g);                  // 10
NP.trainSteps(g, NP.PRESET_LR.relu3, 1000);   // { loss, steps: 1000, diverged: false }
```

## 테스트

`test/`는 게시할 때 복사되지 않는다.

```bash
node --test test/*.test.js                        # 코어
node test/calibrate.js                            # 프리셋 learning rate 실측
uv run --with playwright python test/smoke.py     # 프리셋 6개, 마우스로 패치 만들기, 학습, 창 크기 (30초쯤)
```

`smoke.py`는 1600×1000에서 프리셋마다 스크린샷을 `test/shots/`에 남기고, 실제 마우스로 캔버스 더블클릭(뉴런 추가),
x 출력점에서 뉴런 입력점으로 드래그(선 추가), 뉴런에서 ŷ로 드래그, 숫자 상자 위로 드래그(weight 증가), 선 클릭 후 Delete(선 삭제)를 확인한다.
3 ReLU에서 Measure와 Take a step으로 손실이 내려가는지, Run 2초로 내려가는지, Space로 Run이 켜지고 꺼지는지,
`?preset=layers2`가 노브 33개인지, 1920×1080과 1100×900에서 가로 스크롤이 없는지도 본다.
여러 뉴런 선택도 실제 마우스와 키보드로 확인한다: `Shift`+클릭 세 번 후 tanh 한 번에 바꾸고 `Ctrl+Z` 한 번으로 복구, 사각형 선택,
함께 이동과 복구, Empty에서 뉴런 셋을 사각형으로 골라 ŷ와 x에 한 번에 잇기(노브 10개), 중복 건너뛰기 메시지, `Delete`로 한꺼번에 지우고 복구,
빈 곳 클릭으로 선택 해제, 더블클릭 추가, 하나만 드래그. 스크린샷은 `15-multiselect-1600.png`, `16-marquee.png`, `17-1920x1080-multiselect.png`.
테스트 훅: `NPUI.hooks.select(id 또는 [ids], additive)`, `NPUI.hooks.selection()`(고른 id 배열).

## 조작

- 뉴런 추가: 빈 캔버스 더블클릭(그 자리), 또는 `+ Neuron` 버튼(빈 자리). 새 뉴런은 ReLU, bias 0.5 (bias 0이면 ReLU 꺾이는 점이 모두 x = 0에 모여 학습 중에 죽기 쉽다)
- 연결: 상자 오른쪽 점(출력)에서 드래그해 다른 상자의 왼쪽 점(입력)이나 상자 위에 놓는다. 왼쪽 점에서 다른 상자의 오른쪽 점으로 거꾸로 드래그해도 된다
- 되돌리기: `Ctrl+Z`(맥은 `⌘Z`) 또는 `Undo` 버튼. 다시 하기: `Ctrl+Shift+Z`, `Ctrl+Y`(맥은 `⌘⇧Z`) 또는 `Redo` 버튼. 뉴런 추가·삭제, 연결, 활성화 변경, 프리셋 전환, Reset, Take a step, Run, Randomize, 노브·숫자 상자·상자 이동이 기록된다(최대 200개). learning rate는 기록하지 않는다. Run 중에 되돌리면 Run이 멈추고 Run을 누르기 전 상태로 돌아간다. 숫자 입력 칸에 글자를 치는 중에는 동작하지 않는다
- Reset: 프리셋을 그대로 쓰면 프리셋 처음 상태로, 구조를 바꾼 네트워크(Empty에서 만든 것 등)는 학습을 시작할 때의 상태로 돌아간다. 실수로 눌러도 `Ctrl+Z`로 되돌릴 수 있다
- `Randomize`: 모든 weight와 bias를 새 난수로 다시 뽑는다. 누를 때마다 시드가 1, 2, 3, ... 으로 바뀌어 수업마다 같은 순서가 나온다. weight는 부호 무작위, 크기 0.3 ~ 1.2. x만 입력으로 받는 뉴런은 꺾이는 점이 x ∈ [−2.5, 2.5] 안에 오도록 bias를 정한다
- `Randomize this neuron`: 인스펙터에서 뉴런을 고르면 나온다. 그 뉴런의 들어오는 weight, bias, 나가는 weight만 다시 뽑는다. 죽은 뉴런을 살릴 때 쓴다
- `inactive` 표시: 모든 데이터 점에서 0을 내는 ReLU(또는 step) 뉴런은 회색 점선 테두리와 `inactive` 표시가 붙고, slope가 모두 0이라는 안내가 뜬다. 편집, 한 걸음, Run 중(0.2초마다)에 갱신된다
- 이동: 상자 몸통을 드래그. 여러 개를 고른 상태에서 고른 상자 하나를 드래그하면 모두 같은 거리만큼 함께 움직인다(캔버스 밖으로 나가지 않게 멈춘다). 드래그 한 번이 되돌리기 한 번이다. 고르지 않은 상자를 그냥 드래그하면 그 상자만 골라서 움직인다
- 선택: 상자나 선, 숫자 상자를 클릭. 인스펙터에 활성화 버튼과 bias, 들어오는 weight, 나가는 weight 노브가 나온다. 선을 클릭하면 그 선만 골라진다
- 여러 뉴런 선택: `Shift`+클릭(`Ctrl`, `⌘`+클릭도 된다)으로 하나씩 더하거나 뺀다. 빈 캔버스에서 드래그하면 점선 사각형이 그려지고, 상자 가운데가 사각형 안에 든 뉴런이 골라진다(`Shift`를 누른 채 그리면 기존 선택에 더한다). 빈 캔버스를 그냥 클릭하거나 `Esc`를 누르면 선택이 풀린다. 빈 캔버스 더블클릭은 그대로 뉴런 추가다. 고른 상자 하나를 움직이지 않고 클릭하면 그 상자만 남는다. x와 ŷ도 함께 골라 이동, 연결할 수 있다
- 여러 뉴런 인스펙터: "N neurons selected" 아래 활성화 버튼(모두 같은 활성화일 때만 불이 들어온다)을 누르면 고른 뉴런 전부(ŷ 포함, x 제외)가 한 번에 바뀐다. `Randomize these neurons`는 고른 뉴런마다 `Randomize this neuron`을 한 번씩 하고, `Remove N neurons`는 고른 뉴런을 지운다(x와 ŷ는 남는다). 고른 뉴런의 bias 노브도 나온다. 어느 동작이든 되돌리기 한 번이다
- 여러 선 한 번에 잇기: 여러 개를 고른 상태에서 고른 상자의 오른쪽 점을 끌어 다른 상자에 놓으면 고른 상자 전부(ŷ 제외)에서 그 상자로 선이 생긴다. 고르지 않은 상자(x 등)의 오른쪽 점을 고른 상자에 놓으면 그 상자에서 고른 상자 전부(x 제외)로 선이 생긴다. 고른 상자의 왼쪽 점에서 거꾸로 끌어 다른 상자의 오른쪽 점에 놓아도 그 상자에서 고른 상자 전부로 잇는다. 자기 자신, 이미 있는 연결, 고리, x로 들어가는 연결, ŷ에서 나가는 연결은 조용히 건너뛰고 상태 줄에 "Connected 2 wires, skipped 1 (already connected)"처럼 개수를 띄운다. 한 번 놓아 생긴 선들은 되돌리기 한 번에 사라지고, 선택은 그대로 남는다
- 삭제: 선택한 뒤 `Delete`나 `Backspace`, 또는 인스펙터의 Remove 버튼. 여러 뉴런을 골랐으면 전부 지운다(되돌리기 한 번). 노브에 포커스가 있을 때는 동작하지 않는다. x와 ŷ는 지울 수 없다
- 숫자 상자: 위아래로 드래그하면 weight가 픽셀당 0.01씩, `Shift`를 누르면 0.001씩 바뀐다
- 노브: 4차시와 같다. 드래그, 화살표 키, 다이얼 더블클릭은 0으로. 휠은 노브를 한 번 클릭한 뒤에만 값을 바꾸고(한 칸에 0.1, `Shift`는 0.01), 그 전에는 인스펙터를 스크롤한다. 숫자 상자와 노브는 ±5 안에서 움직인다
- `Space`: Run과 Pause 전환
- 선 색: 빨강은 양수 weight, 파랑은 음수 weight. 굵을수록 |w|가 크다

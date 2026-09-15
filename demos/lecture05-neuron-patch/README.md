# 5차시 데모: Neuron Patch (뉴런을 이어 붙여 만드는 모델)

MAS1004 5차시(뉴런으로 만든 모델) 수업용 인터랙티브 웹 데모. Max/MSP 패치처럼 뉴런 상자를 캔버스에 놓고
x에서 ŷ까지 선으로 이어 1차원 회귀 신경망을 만든다. 기본 데이터는 4차시와 같은 봉우리 8개 데이터(점 120개)이고,
그래프 카드 위쪽 Data 줄에서 Zigzag, Steps, Wave, Random으로 바꿀 수 있다.
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

## 데이터 고르기

그래프 카드 맨 위 Data 줄의 칩으로 맞출 데이터를 바꾼다. 모두 점 120개, x ∈ [−3, 3], 잡음 표준편차 0.08이다.
손실은 `node test/calibrate.js --datasets`로 잰 값이다: 프리셋 learning rate(`NP.PRESET_LR`)로 경사하강 3000걸음.
잡음이 있어서 손실은 0.0064(= 0.08²) 근처보다 내려가지 않는다.

| 키 | 모양 | 보여 주는 것 | 프리셋별 손실 (3000걸음) |
| --- | --- | --- | --- |
| `bumps` (Bumps) | 봉우리 8개의 합. 4차시 데이터 그대로 (y ∈ [−0.82, 3.95]) | 기본 데이터. 매끄러운 곡선 | 2 layers 0.0078, 3 ReLU 0.224, 1 ReLU 0.347, 1 neuron 1.184 |
| `zigzag` (Zigzag) | 꺾인 직선. 꼭짓점 x = −3, −1.5, 0, 1.5, 3에서 y = 2.5, 0.5, 1, 3.5, 1. 내려가 x = −1.5에서 바닥, 0에서 한 번 더 꺾여 가파르게 올라 x = 1.5에서 꼭대기, 다시 내려간다 | 3 ReLU 프리셋이 처음 꺾이는 점(−1.5, 0, 1.5)에서 꺾인다. ReLU 3개로 거의 정확히 맞는다 | 3 ReLU 0.0053 (300걸음 0.029, 1000걸음 0.012), 2 layers 0.009, 1 ReLU 0.506, 1 neuron 0.588 |
| `steps` (Steps) | 세 층 계단 (약 0, 1.5, 3). x = −1.5와 1.5에서 뛴다: `1.5·σ(10(x+1.5)) + 1.5·σ(10(x−1.5))` | 가파른 턱. ReLU는 꺾인 직선이라 턱을 비스듬히 잘라 직선보다 조금 나은 정도다. 3 ReLU의 세 뉴런을 한꺼번에 step이나 tanh로 바꾸면 턱에 맞는다(턱이 프리셋 뉴런이 바뀌는 자리에 있다) | 2 layers(tanh) 0.011, 3 ReLU 0.104, 1 ReLU 0.120, 1 neuron 0.120. 3 ReLU 뉴런을 tanh로 0.020, step으로 0.032, sigmoid로 0.086(`?opt=adam`이면 0.007) |
| `wave` (Wave) | `1.5 + 1.5·sin(3x)`. 봉우리 셋 | 빠르게 오르내린다. 뉴런 3개로는 따라가지 못하고 2 layers는 따라간다 | 2 layers 0.025 (300걸음 0.747, 1000걸음 0.067), 3 ReLU 1.063, 1 neuron 1.077 |
| `random` (Random #N) | 번호 N으로 뽑은 매끄러운 함수: 가우스 봉우리 3개 + 사인 + 기울기. 폭 2 ~ 4로 [−1, 4] 안에 들어가게 늘이고 옮긴다 | 모두가 같은 번호로 같은 문제를 푼다 ("모두 #3으로 해 보세요") | #1: 2 layers 0.014, 3 ReLU 0.091. #2: 2 layers 0.015, 3 ReLU 0.234. #3: 2 layers 0.036, 3 ReLU 0.040 |

- Random 번호마다 난이도가 다르다. `node`로 #1 ~ #30을 잰 값(3000걸음, 1 neuron / 3 ReLU / 2 layers):
  - 3 ReLU로 잘 맞는 번호: #7 (1.164 / 0.052 / 0.008), #17 (0.675 / 0.018 / 0.006), #21 (0.515 / 0.010 / 0.010), #28 (0.937 / 0.051 / 0.009)
  - 2 layers가 필요한 번호: #8 (0.660 / 0.531 / 0.016), #19 (1.008 / 0.485 / 0.015), #10 (0.918 / 0.416 / 0.031)
  - 피할 번호: #16, #25, #22는 거의 직선이다(1 neuron 0.052, 0.109, 0.147). #11은 2 layers도 0.260에서 멈춘다
- Steps에서 step 뉴런은 slope가 늘 0이라 꺾이는 자리는 움직이지 않고 나가는 weight와 ŷ bias만 학습된다. sigmoid는 기본 경사하강으로는 3000걸음에 0.086에서 거의 멈춘다(slope ≈ 0.001).
  tanh는 Bumps에서도 ReLU보다 훨씬 낫다(0.0075 대 0.224). "S자가 턱에 맞는다"는 비교는 step으로 보이는 편이 분명하다: step은 Steps 0.032, Bumps 0.448.
- 1 neuron과 3 Linear는 어느 데이터에서나 같은 직선으로 끝난다(두 열이 같은 값). Empty는 ŷ의 bias만 있어서 데이터 평균이 된다.
- 프리셋 6개 × 데이터(고정 모양 4개와 Random #1 ~ #10)에서 3000걸음 동안 발산이 없고 손실이 내려간다 (`test/datasets.test.js`).
- 칩을 누르면 점만 바뀌고 네트워크(weight, bias, 연결, 활성화)는 그대로다. 측정한 slope는 회색이 되고 Take a step이 꺼진다.
  손실 곡선과 Adam 상태는 새로 시작하고, Run 중이면 계속 돈다. 데이터 변경은 되돌리기 한 번이다(그때의 네트워크와 데이터로 함께 돌아간다).
- `Random`은 지금 번호(처음 1)의 함수를 쓴다. `New random function`은 번호를 하나 올리고 Random으로 바꾼다(#999 다음은 #1). 버튼 옆에 `Random #3`처럼 번호가 뜬다.
  링크로 줄 때는 `?data=random&seed=3` (New random function을 두 번 누른 것과 같은 점).
- 프리셋 버튼과 Reset은 데이터를 바꾸지 않는다. 그래프의 y축 범위는 보이는 데이터의 최솟값과 최댓값에 15% 여유를 두고 맞춘다. 범례에 `data: Zigzag`처럼 지금 데이터가 적힌다.
- Zigzag 꼭짓점: 처음 요청은 y = 0, 2.5, 0.5, 3, 1(네 조각이 번갈아 오르내림)이었지만 3 ReLU 프리셋은 이것을 맞출 수 없다(3000걸음 후 0.249에서 멈춤).
  프리셋의 세 뉴런은 오른쪽, 왼쪽, 오른쪽으로 열려 있어서 `기울기(−1.5 ~ 0) = 기울기(−1.5 왼쪽) + 기울기(0 ~ 1.5)`인 모양만 정확히 맞출 수 있다. 지금 꼭짓점은 이 관계를 만족하고, 프리셋의 처음 weight 부호도 그대로 둔 채 맞춰진다.

## URL 옵션

| 옵션 | 용도 |
| --- | --- |
| `?preset=relu3` | 시작 프리셋 (`line`, `relu1`, `relu3`, `linear3`, `layers2`, `empty`). 기본은 `line` |
| `?lr=0.02` | learning rate 초기값. 프리셋 버튼을 누르면 그 프리셋의 값으로 다시 바뀐다 |
| `?opt=adam` | 숨은 옵션. 갱신 규칙을 Adam으로 바꾼다 (기본 learning rate 0.02, 프리셋을 눌러도 유지). 학생 화면에는 규칙 이름만 "update rule: Adam"으로 바뀐다 |
| `?data=wave` | 시작 데이터 (`bumps`, `zigzag`, `steps`, `wave`, `random`). 기본은 `bumps` |
| `?seed=3` | Random 함수 번호 (1 ~ 999의 정수, 기본 1. 범위 밖이나 숫자가 아니면 1). `?data=random&seed=3`은 Random #3으로 시작한다. 다른 데이터와 함께 주면 Random 칩을 눌렀을 때 쓸 번호가 된다 |

옵션은 `?`와 `#` 어느 쪽으로 줘도 된다: `index.html?preset=layers2#lr=0.05`, `index.html?preset=relu3&data=zigzag`.

## 데이터와 모델

- 봉우리 데이터(`NP.DATA`)는 4차시 `GD.DATA`와 같은 점 120개다 (x ∈ [−3, 3], 숨은 함수는 봉우리 8개의 합에 표준편차 0.08 잡음,
  시드 `bump-truth|19`, `bump-data|19`, `DATA_VERSION = 1`). 난수 코드를 4차시에서 그대로 복사해 같은 숫자가 나온다.
- 다른 데이터는 `NP.makeData(key, seed)`가 같은 형식으로 만든다: `x = −3 + 6(i + r())/120`, `y = f(x) + 0.08·gauss(r)`, 둘 다 소수 둘째 자리,
  `r = rngOf('np|data|' + key + '|' + seed)`. 고정 모양(zigzag, steps, wave)은 시드 0, Random #N은 시드 N이고 함수 자체는 `rngOf('np|fn|' + N)`에서 뽑는다
  (`NP.randomFunction(N)`, 잡음 없는 곡선은 `NP.dataFunction(key, seed)`). 목록은 `NP.DATASETS`.
- `NP.setData(points)`가 기본 데이터를 정하고 `NP.getData()`가 돌려준다. data 인자 없이 부른 `NP.loss`, `NP.gradient`, `NP.numericGradient`,
  `NP.trainSteps`, `NP.deadNeurons`, `NP.randomize`(ŷ bias = 데이터 평균)가 이 데이터를 쓴다. `NP.setData(null)`은 봉우리로 되돌린다. `NP.DATA`, `NP.XS`, `NP.YS`는 늘 봉우리다.
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

- `<script id="np-core">`: 난수, 데이터(봉우리와 Data 줄의 모양들, 기본 데이터 바꾸기), 활성화, 그래프 편집(뉴런, 선, 연결 검사), 순전파, 손실, gradient, 수치 gradient,
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
node --test test/*.test.js                        # 코어 (core, randomize, datasets)
node test/calibrate.js                            # 프리셋 learning rate 실측, 끝에 프리셋 × 데이터 표 (1분쯤)
node test/calibrate.js --datasets                 # 프리셋 × 데이터 표만 (300, 1000, 3000걸음 손실)
uv run --with playwright python test/smoke.py     # 프리셋 6개, 마우스로 패치 만들기, 학습, 데이터, 창 크기 (1분쯤)
```

`test/datasets.test.js`는 데이터마다 결정성, 점 120개, y ∈ [−1.5, 4.5](Random #1 ~ #50 포함), 번호마다 다른 점, 봉우리 = 4차시 `GD.DATA`,
Zigzag 꼭짓점, `setData`로 손실과 gradient가 바뀌는지, 데이터마다 역전파 = 중앙 차분, 프리셋 × 데이터 3000걸음 발산 없음,
3 ReLU가 Zigzag에서 손실 < 0.02, Wave에서 2 layers < 0.1이고 3 ReLU > 0.8인지, Steps에서 3 ReLU 뉴런을 step이나 tanh로 바꾸면 ReLU의 절반 이하인지,
`NP.setData`가 배열이 아닌 입력을 거절하는지, Random 번호가 1 ~ 999 밖이면 #1인지 본다.

`smoke.py`의 데이터 부분은 실제 마우스로 Data 칩 다섯 개를 눌러 점이 바뀌는지(`NPUI.hooks.data()`와 `NP.getData()` 비교), 칩마다 Run으로 손실이 내려가는지,
측정한 slope가 회색이 되는지, Run 중에 데이터를 바꿔도 계속 도는지, 프리셋이 데이터를 바꾸지 않는지, New random function 두 번이면 `Random #3`이고 #2와 점이 다른지,
`Ctrl+Z`가 이전 데이터로, `Ctrl+Shift+Z`가 다시 되돌리는지, `?data=wave`와 `?data=random&seed=3`(클릭으로 만든 #3과 같은 점),
1600×1000과 1920×1080에서 스크롤이 없고 Data 줄이 한 줄인지(Random #999 포함, 데이터를 바꿔도 그래프 카드가 움직이지 않는지),
발산한 네트워크에서 데이터를 바꿔도 "Diverged" 안내가 남는지, `?seed=1e21`이 #1인지, 같은 데이터에 다른 번호를 준 `setData`가 번호를 몰래 바꾸지 않는지 확인한다.
스크린샷: `19-random-3.png`, `18-1920x1080-random12.png`, 데이터마다 3 ReLU로 Run한 `20-bumps-relu3-run.png` ~ `24-random3-relu3-run.png`, Wave에서 2 layers로 Run한 `25-wave-layers2-run.png`.
테스트 훅: `NPUI.hooks.setData(key, seed)`(seed를 빼면 Random은 지금 번호), `NPUI.hooks.data()`(`{key, seed}`).

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
- 데이터: 그래프 카드 위 `Data` 칩(Bumps, Zigzag, Steps, Wave, Random)으로 바꾼다. `New random function`은 Random 번호를 하나 올리고 Random으로 바꾸며 옆에 `Random #N`을 띄운다. 네트워크는 그대로 남는다
- 되돌리기: `Ctrl+Z`(맥은 `⌘Z`) 또는 `Undo` 버튼. 다시 하기: `Ctrl+Shift+Z`, `Ctrl+Y`(맥은 `⌘⇧Z`) 또는 `Redo` 버튼. 뉴런 추가·삭제, 연결, 활성화 변경, 프리셋 전환, 데이터 변경, Reset, Take a step, Run, Randomize, 노브·숫자 상자·상자 이동이 기록된다(최대 200개). Run 중에 데이터를 바꾼 뒤 되돌리면 데이터를 바꾸기 직전으로 돌아간다. learning rate는 기록하지 않는다. Run 중에 되돌리면 Run이 멈추고 Run을 누르기 전 상태로 돌아간다. 숫자 입력 칸에 글자를 치는 중에는 동작하지 않는다
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

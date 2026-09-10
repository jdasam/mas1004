# 4차시 데모: Gradient Descent (기울기를 재서 내려가기)

MAS1004 4차시(경사하강법) 수업용 인터랙티브 웹 데모. 학생이 노브 하나를 Δ만큼 밀어 slope를 재고,
노브 8개의 slope 목록(gradient)을 표로 채우고, Measure와 Take a step 버튼으로 경사하강을 손으로 돌리고,
learning rate와 시작 위치를 바꿔 결과가 어떻게 갈리는지 본다. 데이터와 8노브 모델은 3차시 데모의 Mystery와 같다.
설계 문서: `docs/superpowers/specs/2026-09-10-lecture04-gradient-descent-design.md`.

의존성 없는 단일 파일이다. `index.html` 하나를 브라우저로 열면 되고(`file://` 포함), GitHub Pages에 그 파일만 올려도 동작한다.
빌드 단계도 외부 네트워크 요청도 없다. UI는 영어다.

## 학생에게 배포하는 법

파일 하나라서 방법이 셋이다. 어느 쪽이든 학생은 링크를 열거나 파일을 더블클릭하면 된다.

1. **LMS에 첨부**: `index.html`을 사이버캠퍼스 자료실에 올린다. 학생은 내려받아 브라우저로 연다.
   `file://`로 열려도 모든 기능이 동작한다.
2. **GitHub Pages** (기본): 공개 저장소 `jdasam/mas1004`의 2026 브랜치가 Pages로 열려 있다.
   `mas1004-2026/tools/publish_demo.sh lecture04-gradient-descent`를 실행하면 이 폴더가 서브모듈 `mas1004/demos/`로 복사되고
   push까지 된다 (`test/`는 복사에서 빠진다). 학생용 링크: https://jdasam.github.io/mas1004/demos/lecture04-gradient-descent/
   (데모 목록: https://jdasam.github.io/mas1004/demos/). 이 상위 저장소(`sogang-course`)는 비공개라 Pages를 쓸 수 없다.
3. **수업 중 임시 서버**: 교실 PC에서 `python -m http.server 8000`을 이 폴더에서 띄우고 같은 네트워크의 학생에게
   `http://<교수 PC IP>:8000/`을 알려 준다.

## 수업에서 쓰는 법

탭 네 개를 슬라이드 순서대로 진행한다. 탭 잠금은 없으므로 진행 통제는 구두로 한다.
탭 1, 2, 3은 8노브 모델의 노브 값을 공유한다. 탭 4는 12노브 모델을 따로 쓴다.

| 탭 | 모델 | 학생이 하는 일 |
| --- | --- | --- |
| 1 One knob at a time | 8노브 | 노브 하나를 골라 손실 곡선을 본다. Δ 슬라이더로 두 점의 slope가 Δ → 0의 값으로 수렴하는 것을 본다. 확대하면 곡선이 직선이 된다 |
| 2 Measure the gradient | 8노브 | "Measure the gradient"를 눌러 노브를 하나씩 밀어 표를 채운다. 카운터가 9 는다. 노브를 돌리면 표가 흐려진다 |
| 3 Step by step | 8노브 | Measure와 Take a step을 번갈아 누른다. Run으로 자동 반복. learning rate를 바꿔 세 가지 결과(느림, 적당, 발산)를 본다 |
| 4 Where you start | 12노브 | 주파수와 위상도 노브다. Start A, B, C에서 각각 Run하고 기록표에서 도착 손실이 다른 것을 읽는다 |

탭 3과 4의 손실 지도는 노브 두 개의 단면이다. 나머지 노브가 움직이면 지도도 바뀐다.
지도를 그리는 3,600번의 손실 계산은 카운터에 넣지 않는다.

## URL 옵션

| 옵션 | 용도 |
| --- | --- |
| `#tab=3` | 특정 탭으로 바로 열기 |
| `#lr=0.05` | learning rate 초기값 (탭 3과 4에 같이 적용) |
| `#start=B` | 탭 4의 시작 프리셋 미리 고르기 (A, B, C) |

옵션은 `?`와 `#` 어느 쪽으로 줘도 된다: `index.html?tab=4#start=B`.
3차시의 `?auto=1` 같은 교수 전용 옵션은 없다. 자동 실행(Run)이 이번 시간의 학생 기능이다.

## 데이터

- 3차시 데모의 Mystery 데이터와 점 하나까지 같다. 같은 시드(`rs|v1|data|mystery`, `rs|v1|mystery-truth`)와
  같은 `DATA_VERSION = 1`을 쓴다. 3차시 파일을 고쳐 데이터가 달라지면 이 데모도 같이 고친다.
- 8노브 모델은 3차시 탭 5의 모델 그대로다. 12노브 모델은 sin과 cos의 주파수와 위상을 노브로 푼 것이고,
  정답(w₂ = 2, w₃ = 0, w₅ = 3, w₆ = 0)을 넣으면 8노브 모델과 완전히 같은 함수가 된다.

## 계산 방식

수업에서 말하는 것과 화면의 숫자가 같다.

- slope는 전진 차분이다. `slope = (Loss(w + Δ) − Loss(w)) / Δ`, `Δ = 0.001`.
- gradient 한 번에 손실을 `k + 1`번 계산한다 (지금 자리 1번, 노브마다 1번). 카운터가 그만큼 는다.
- 갱신은 `w ← w − lr × slope`. 내부 표준화는 없다. 노브에 보이는 값을 그대로 갱신한다.
- Run은 손으로 Measure와 Step을 누르는 것과 같은 계산을 반복한다. 결과도 같다.
- 경사하강 중에는 노브 값을 범위로 자르지 않는다. 범위를 벗어나면 다이얼은 끝에 걸리고 숫자만 실제 값을 보인다.
  손실이 유한하지 않거나 10⁶을 넘으면 멈추고 "Diverged: the loss blew up"을 띄운다.

프리셋 값은 `test/calibrate.js`로 실측했다. 8노브: small 0.0002, about right 0.3, too large 0.6 (0.4부터 발산).
12노브: 0.07. 시작 A, B, C도 같은 스크립트가 고른 자리다.

## 구조

한 파일 안에 스크립트가 두 개다.

- `<script id="gd-core">`: 난수, 데이터, 모델 두 개, 손실, 수치 gradient, 갱신, 실행기, 손실 곡선과 손실 격자,
  실측 상수. DOM을 건드리지 않는다.
- `<script id="gd-ui">`: 노브 위젯, 캔버스 산점도와 손실 곡선과 손실 지도, gradient 표, 탭 네 개.

코어만 Node에서 실행하려면:

```js
const fs = require('fs');
const html = fs.readFileSync('index.html', 'utf8');
const K = (0, eval)(html.match(/<script id="gd-core">([\s\S]*?)<\/script>/)[1] + '\n;GD');
K.numericalGradient(K.MODELS.m8, K.START.m8);   // { lossNow: 2.387, slopes: [...], evals: 9, delta: 0.001 }
K.run(K.MODELS.m8, K.START.m8, K.LR.m8.good, 2000).loss;   // 0.00625
```

## 테스트

`test/`는 게시할 때 복사되지 않는다.

```bash
node --test test/*.test.js                        # 코어 (17개)
node test/calibrate.js                            # 시작 위치와 learning rate 프리셋 실측 (1분쯤)
uv run --with playwright python test/smoke.py     # 탭 4개를 열고 핵심 동작 확인 (2분쯤)
```

## 조작

- 다이얼을 위아래로 드래그. `Shift`를 누른 채 드래그하면 미세 조정
- 노브를 클릭한 뒤 화살표 키로 한 눈금씩, `Shift`+화살표로 크게, `Home`으로 초기값
- 마우스 휠, 다이얼 더블클릭(초기값), `−`/`+` 버튼, 터치 모두 동작
- 산점도 위에 마우스를 올리면 가장 가까운 점의 x, y, ŷ가 보인다
- 탭 1의 축소판을 누르면 그 노브가 큰 그래프로 온다

# 3차시 데모: Fitting a Function to Data (직선 맞추기에서 전수 탐색까지)

MAS1004 3차시(함수 근사) 수업용 인터랙티브 웹 데모. 학생이 직선을 손으로 맞추고, 손실을 보고,
컴퓨터의 전수 탐색(grid search)을 돌려 보고, 파라미터가 늘어나면 전수 탐색도 불가능해진다는 것을 겪는다.
다음 시간의 경사하강법을 위한 밑작업이다. 설계 문서: `docs/superpowers/specs/2026-09-08-lecture03-regression-design.md`.

의존성 없는 단일 파일이다. `index.html` 하나를 브라우저로 열면 되고(`file://` 포함), GitHub Pages에 그 파일만 올려도 동작한다.
빌드 단계도 외부 네트워크 요청도 없다. UI는 영어다.

## 학생에게 배포하는 법

파일 하나라서 방법이 셋이다. 어느 쪽이든 학생은 링크를 열거나 파일을 더블클릭하면 된다.

1. **LMS에 첨부**: `index.html`을 사이버캠퍼스 자료실에 올린다. 학생은 내려받아 브라우저로 연다.
   `file://`로 열려도 모든 기능이 동작한다.
2. **GitHub Pages** (기본): 공개 저장소 `jdasam/mas1004`의 2026 브랜치가 Pages로 열려 있다.
   `mas1004-2026/tools/publish_demo.sh lecture03-regression-search`를 실행하면 이 폴더가 서브모듈 `mas1004/demos/`로 복사되고
   push까지 된다. 학생용 링크: https://jdasam.github.io/mas1004/demos/lecture03-regression-search/
   (데모 목록: https://jdasam.github.io/mas1004/demos/). 이 상위 저장소(`sogang-course`)는 비공개라 Pages를 쓸 수 없다.
3. **수업 중 임시 서버**: 교실 PC에서 `python -m http.server 8000`을 이 폴더에서 띄우고 같은 네트워크의 학생에게
   `http://<교수 PC IP>:8000/`을 알려 준다. 학교 무선망이 기기 간 접속을 막으면 안 된다.

교수 화면은 `?auto=1`을 붙여 연다. 학생에게 주는 링크나 파일에는 붙이지 않는다 (URL 옵션 표 참조).

## 수업에서 쓰는 법

탭 다섯 개를 슬라이드 순서대로 진행한다. 탭 잠금은 없으므로 진행 통제는 구두로 한다.

| 탭 | 데이터 | 학생이 하는 일 |
| --- | --- | --- |
| 1 Data | Rent, Cafe, Sleep, Galton 1886 | 점만 보고 질문의 값을 눈대중으로 입력 |
| 2 A line by hand | 위와 같음 | 노브 2개로 직선을 맞춘다. "Show the error"로 잔차와 손실(MSE) 공개, 최저 기록 |
| 3 Brute force | 위와 같음 | (a, b) 격자를 10, 30, 100 스텝으로 전부 계산. 손실 지도, 평가 횟수, 경과 시간, 비용 표 |
| 4 More terms | Study | 직선이 안 맞는 데이터. 2차, 3차 모델로 바꾸고 노브 3, 4개. 격자 탐색 10, 20, 30 스텝 |
| 5 Mystery function | Mystery | sin, cos, exp, log, x⁴가 섞인 숨은 함수. 노브 8개. 격자 비용 표 |

탭 1에서 3까지는 데이터셋 칩이 공유된다. 갈턴 데이터는 부모 평균 키와 자녀 키를 평균 기준 편차(cm)로 표시한다.

## URL 옵션

| 옵션 | 용도 |
| --- | --- |
| `?auto=1` | **교수 전용.** 탭 2, 4, 5에 "Let the machine turn the knobs" 버튼이 나타난다. 경사하강이 한 스텝씩 노브를 돌린다 |
| `#tab=3` | 특정 탭으로 바로 열기 |
| `#ds=galton` | 탭 1~3의 데이터셋을 미리 고르기 (rent, cafe, sleep, galton) |

옵션은 `?`와 `#` 어느 쪽으로 줘도 된다: `index.html?auto=1#tab=5`. 학생에게 주는 링크에는 `auto`를 붙이지 않는다.

## 데이터

- 시드가 고정되어 있어서 교수 화면과 학생 화면의 데이터가 같다. 생성 로직은 `rs-core` 스크립트의 `DATASETS`에 있다.
- `DATA_VERSION`을 두었다. 수업에 한 번 쓰인 뒤 생성 로직을 고치면 이 숫자를 올린다.
- 갈턴 데이터는 HistData 패키지의 `Galton` 표(928명)를 인치에서 cm로 바꿔 파일에 내장했다.
- 숨은 함수의 가중치는 `mysteryTruth()`가 시드로 뽑고, 상수항은 함수의 평균이 0이 되도록 정한다.

## 구조

한 파일 안에 스크립트가 두 개다.

- `<script id="rs-core">`: 난수, 데이터, 모델(기저 함수의 선형 결합), 손실, 격자 탐색, 경사하강. DOM을 건드리지 않는다.
- `<script id="rs-ui">`: 노브 위젯, 캔버스 산점도와 손실 지도, 탭 다섯 개, 자동 맞추기.

코어만 Node에서 실행하려면:

```js
const fs = require('fs');
const html = fs.readFileSync('index.html', 'utf8');
const K = (0, eval)(html.match(/<script id="rs-core">([\s\S]*?)<\/script>/)[1] + '\n;RS');
K.leastSquares(K.MODELS.line, 'rent');   // { p: [1.521, 25.195], loss: 36.98 }
```

경사하강은 내부적으로 특징을 표준화한 좌표에서 돌고, 노브에는 원래 모델의 파라미터를 되돌려 표시한다.
학습률은 직선 0.2, 2차와 3차 0.15, 숨은 함수 0.06이다.

## 조작

- 다이얼을 위아래로 드래그. `Shift`를 누른 채 드래그하면 미세 조정
- 노브를 클릭한 뒤 화살표 키로 한 눈금씩, `Shift`+화살표로 크게, `Home`으로 초기값
- 마우스 휠, 다이얼 더블클릭(초기값), `−`/`+` 버튼, 터치 모두 동작
- 산점도 위에 마우스를 올리면 가장 가까운 점의 x, y, ŷ가 보인다

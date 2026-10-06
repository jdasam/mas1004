# 데모: Sliding a Kernel (CNN 커널이 이미지를 훑는 과정)

MAS1004 합성곱 수업용 인터랙티브 웹 데모. 28 × 28 입력 이미지 위로 3 × 3 커널을 한 칸씩 옮기며
출력 26 × 26의 각 픽셀이 "입력 조각 9개 × 커널 9개를 곱해 더한 값"이라는 것을 보여 준다.
의존성 없는 단일 파일이고 `file://`로 열어도 동작한다. UI는 영어다.

## 화면

- 왼쪽: 입력 이미지 (0 = 흰색, 1 = 검정). 마우스로 그리고, Shift 드래그나 오른쪽 드래그로 지운다. 빨간 사각형이 지금 보는 3 × 3 조각이다.
- 가운데 위: 커널 9개 숫자. 위아래로 드래그해 바꾸고(픽셀당 0.02, Shift는 0.002), 더블클릭하면 직접 입력한다.
- 가운데 아래: 지금 출력 픽셀 하나의 계산. 입력 조각 × 커널 = 곱 9개, 그 합. ReLU를 켜면 ReLU(합)도 보인다.
- 오른쪽: 출력. 빨강은 양수(조각이 커널과 닮았다), 파랑은 음수. 색 범위는 출력 전체의 최대 |값|에 맞춘다.
  출력 픽셀에 마우스를 올리면 그 픽셀의 계산과 입력 조각이 함께 바뀐다.

## 조작

- Image: 7, 3, A, Square, Circle, Cross, Blank. 글자는 브라우저 글꼴로 28 × 28에 그려 쓴다.
- Kernel: Identity, Blur, Sharpen, Vertical edge(Sobel x), Horizontal edge(Sobel y), Diagonal line, Outline(라플라시안), Random(누를 때마다 #1, #2, ...; 같은 번호면 모두 같은 값).
- ▶ Slide: 출력을 비우고 왼쪽 위부터 한 픽셀씩 채운다. Speed는 초당 5, 40, 400픽셀. Step은 한 픽셀, Show all은 끝까지 채운다.
- Output: Sum only / ReLU(sum).
- 키보드: Space는 Slide/Pause, → 는 Step.
- URL 옵션: `?img=square&kernel=horizontal` (키는 위 목록의 영문 키: `seven`, `three`, `letterA`, `square`, `circle`, `cross`, `blank` / `identity`, `blur`, `sharpen`, `vertical`, `horizontal`, `diagonal`, `outline`).

## 계산

패딩 없음, stride 1, bias 없음. `out[r][c] = Σ_{i,j} img[r+i][c+j] · w[i][j]` (딥러닝 관례대로 커널을 뒤집지 않는다).
코어는 `<script id="ck-core">`의 전역 `CK`이고 DOM을 쓰지 않는다.

## 게시

`mas1004-2026/tools/publish_demo.sh cnn-kernel`

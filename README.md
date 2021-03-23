# motion_classification_contest

## Dacon에서 주관한 Exercise Motion Classfication Contest 도전 기록입니다.

- 데이터 설명
  - 운동 중 3축 가속도계(acceleerometer)와 3축 자이로스코프(gyroscope)를 활용해 측정된 센서 데이터
  - 하나의 데이터는 6개의 차원(가속도계, 자이로스코프의 x,y,z) 에 대한 변화량이 600 time step으로 주어진 시계열 데이터
  - train set: 1747개의 데이터
  - test set: 781개의 데이터
  - 총 61개의 label, Shoulder Press, Push up 등의 운동 상태

- EDA
  - 61개의 클래스에 대한 불균형이 굉장히 심한 것을 알 수 있음
  - ![image](https://user-images.githubusercontent.com/50436240/112161117-290c7a80-8c2e-11eb-9bbb-5d1b44dad90a.png)
  - 데이터 증강 기법 활용 필요 -> 아래의 세가지 테크닉을 모두 적용하여, 각각의 클래스를 700개까지 증가시킴
    1. Random Noise 추가
    2. 시계열의 데이터 중 임의의 시작점과 끝점을 잡고 길이를 늘리는 방법
    3. 시계열 데이터 값에 상수배를 곱하는 방법

  - 가속도계 분포
  - ![image](https://user-images.githubusercontent.com/50436240/112161946-f31bc600-8c2e-11eb-848f-0af85ad4c044.png)


  - 자이로스코프 분포
  - ![image](https://user-images.githubusercontent.com/50436240/112162102-147cb200-8c2f-11eb-82c4-e965b6bed7a3.png)


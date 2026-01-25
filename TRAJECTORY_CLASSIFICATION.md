# Trajectory Classification Criteria

경로(Trajectory) 분류 기준 문서

## 📋 개요

VAE-Planner 프로젝트에서 사용하는 궤적 분류 기준을 정의합니다. 궤적은 주행 패턴에 따라 여러 카테고리로 분류됩니다.

## 🎯 기본 분류 카테고리

### 1. Stop (정지)
- **기준**: 시작점과 끝점 사이의 총 이동 거리가 매우 짧음
- **임계값**: 
  - `train.py`: 1.5m 미만
  - `visualization_server/app.py`: 2.0m 미만
- **추가 조건**: 
  - 유효한 이동이 5개 미만인 경우도 정지로 분류
  - 모든 궤적의 시작점은 (0, 0)으로 정규화됨

### 2. Straight (직진)
- **기준**: 방향 변화가 거의 없고, 횡방향 이동이 작음
- **임계값**:
  - 평균 회전 강도: < 0.02
  - 누적 회전: < 0.3
  - 횡방향 변위: < 2.0m
- **각도 기준** (`app.py`): 시작-끝점 각도가 -10° ~ +10° 범위

### 3. Left Turn (좌회전)
- **기준**: 왼쪽으로 회전하는 경로
- **판단 기준**:
  - 누적 회전 > 0.1 (양수) 또는
  - 누적 회전 > 0 이고 y 변위 > 1.0m
- **각도 기준** (`app.py`): 시작-끝점 각도 > 10°

### 4. Right Turn (우회전)
- **기준**: 오른쪽으로 회전하는 경로
- **판단 기준**:
  - 누적 회전 < -0.1 (음수) 또는
  - 누적 회전 < 0 이고 y 변위 < -1.0m
- **각도 기준** (`app.py`): 시작-끝점 각도 < -10°

## 🔍 상세 분류 (visualization_server/app.py)

`visualization_server/app.py`에서는 추가적인 세부 분류를 제공합니다:

### 곡률 기반 분류

- **Sharp Turn (급커브)**: `{direction}_sharp`
  - 평균 곡률 > 0.15 라디안 또는
  - 최대 곡률 > 0.3 라디안

- **Normal Turn (일반 커브)**: `{direction}`
  - 위 조건을 만족하지 않는 경우

### 속도 기반 분류

- **Slow (느린 경로)**: `{direction}_slow`
  - 평균 속도 < 0.5 m/step (10Hz 기준 = 5 m/s)
  - 곡률보다 우선순위가 높음

## 📐 좌표계 정의

- **로컬 좌표계**: 모든 궤적의 시작점은 (0, 0)으로 정규화됨
- **X축**: 차량 정면 방향 (전진 방향)
- **Y축**: 차량 좌측 방향 (왼쪽이 양수)
- **각도**: X축 기준 반시계방향이 양수 (왼쪽 회전)

## 🔧 계산 방법

### 1. 총 이동 거리
```python
total_distance = np.linalg.norm(end - start)
```

### 2. 방향 벡터 (속도 벡터)
```python
deltas = np.diff(trajectory_xy, axis=0)  # (79, 2)
```

### 3. 회전 각도 계산
```python
# 연속된 속도 벡터 간의 외적 (z 성분)
v1_norm = v1 / ||v1||
v2_norm = v2 / ||v2||
cross = v1_norm[0] * v2_norm[1] - v1_norm[1] * v2_norm[0]
# 양수: 좌회전, 음수: 우회전
```

### 4. 누적 회전
```python
cumulative_rotation = sum(cross_products)
```

### 5. 곡률 계산 (app.py)
```python
# 연속된 속도 벡터 간의 각도 변화
dot = np.dot(v1_unit, v2_unit)
angle_change = np.arccos(np.clip(dot, -1.0, 1.0))
```

## 📊 분류 우선순위

1. **Stop**: 거리 < 임계값 → 즉시 정지로 분류
2. **Slow**: 속도 < 임계값 → `{direction}_slow`
3. **Sharp**: 곡률 > 임계값 → `{direction}_sharp`
4. **Normal**: 기본 방향 분류 → `{direction}`

## 🔄 함수별 차이점

### train.py의 classify_trajectory
- **출력**: `'stop'`, `'left'`, `'right'`, `'straight'`
- **용도**: 학습 후 latent space 분석
- **특징**: 간단하고 빠른 분류

### visualization_server/app.py의 classify_trajectory
- **출력**: `'stop'`, `'straight'`, `'left'`, `'right'`, `'{direction}_sharp'`, `'{direction}_slow'`
- **용도**: 시각화 서버에서 상세한 분류
- **특징**: 곡률과 속도 정보 포함

## 📝 사용 예시

```python
# train.py 버전
label = classify_trajectory(trajectory_xy)
# 출력: 'stop', 'left', 'right', 'straight'

# visualization_server/app.py 버전
label = classify_trajectory(trajectory_xy)
# 출력: 'stop', 'straight', 'left', 'right', 
#       'straight_sharp', 'left_sharp', 'right_sharp',
#       'straight_slow', 'left_slow', 'right_slow'
```

## ⚙️ 임계값 조정

분류 성능을 개선하려면 다음 임계값들을 조정할 수 있습니다:

### train.py
- `total_distance < 1.5`: 정지 판단 거리
- `rotation_strength_mean < 0.02`: 직진 판단 회전 강도
- `cumulative_rotation > 0.1`: 좌/우회전 판단 임계값

### visualization_server/app.py
- `total_distance < 2.0`: 정지 판단 거리
- `angle_deg 범위 ±10°`: 직진 판단 각도 범위
- `avg_curvature > 0.15`: 급커브 판단 곡률
- `avg_speed < 0.5`: 느린 경로 판단 속도

## 🎨 시각화

분류된 궤적은 latent space 분석 시 색상으로 구분됩니다:

- **Stop**: 빨간색 (red)
- **Left Turn**: 파란색 (blue)
- **Right Turn**: 녹색 (green)
- **Straight**: 주황색 (orange)

## 📚 참고

- 궤적 데이터는 80 타임스텝 (8초 × 10Hz)으로 구성됨
- 각 타임스텝은 [x, y] 좌표 (2차원)
- 전체 궤적은 160차원 벡터 (80 × 2)

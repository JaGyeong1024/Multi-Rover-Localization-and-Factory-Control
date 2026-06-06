# Multi-Rover Localization & Factory Control — Host PC

멀티 로버(리더 + 팔로워) 스마트팩토리의 호스트 PC 소프트웨어.
**UWB · 천장 카메라(YOLO11n-OBB) · IMU**를 융합해 각 로버의 글로벌 위치를 추정하고(`/{robot}/global/position`), 실시간 관제 대시보드를 제공한다.

## 요구 사항
- Ubuntu 22.04 · ROS 2 Humble · NVIDIA GPU (CUDA 12.4+)
- UWB 모듈(DWM1001) + 천장 카메라(Logitech BRIO) — 둘 다 USB 연결

## 실행

```bash
git clone https://github.com/JaGyeong1024/Multi-Rover-Localization-and-Factory-Control.git ~/capstone
cd ~/capstone
bash setup/setup.sh             # 1회: udev · 의존성 · CUDA venv · 대시보드 deps · 빌드
# (안내가 뜨면 1회 재로그인) → UWB 동글 + 카메라 USB 연결

bash host_ws/run_pipeline.sh    # 위치추정만 → /{robot}/global/position 발행
bash host_ws/run_dashboard.sh   # 전체 + 관제 대시보드 → http://localhost:5173
```

YOLO 모델과 카메라 캘리브레이션이 레포에 포함되어 **clone만으로 바로 실행**된다.

## 구성

| 경로 | 역할 |
|------|------|
| `host_ws/` | ROS 2 워크스페이스 — 센서 융합(EKF) + perception 노드 + 실행 스크립트 |
| `dashboard/` | 웹 관제 UI (Vite + rosbridge / web_video) |
| `setup/` | 1회 머신 세팅 (udev · 의존성 · 캘리브레이션) |
| `data/` | 런타임 모델 가중치 + 카메라 캘리브레이션 |

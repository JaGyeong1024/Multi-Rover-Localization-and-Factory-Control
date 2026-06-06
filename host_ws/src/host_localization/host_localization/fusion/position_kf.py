"""PositionKF — CP(constant-position) 위치 융합 KF (순수 numpy, ROS 비의존).

기존 `experiments/yolo_runtime/kf_fusion.py`의 검증된 로직을 **런타임 패키지로 이관(canonical)**.
상태 [x, y]. YOLO(작은 σ) + UWB(qf 가중, affine 보정 후) 융합, Mahalanobis 게이팅으로 NLOS reject.
- CP 권장: 슬로우 로버 + 고빈도 측정엔 속도상태 불필요(CV는 velocity 튀면 발산 — 2026-06-02 NIS 확인).
- 헤딩↔위치 결합(odom 활용)은 이 클래스 아님 — SE(2) unicycle은 `se2nhc.py`(EKFSE2, 현행 추정기).
"""
import numpy as np


class PositionKF:
    def __init__(self, sigma_rw=1.0, gate=9.21):
        self.sig_rw = float(sigma_rw)     # 위치 random-walk (m/√s)
        self.gate = float(gate)           # Mahalanobis χ²(2 dof, 99%) = 9.21
        self.x = None                     # [x, y]
        self.P = None

    def predict(self, dt):
        if self.x is None or not (0.0 < dt <= 1.0):
            return
        self.P = self.P + np.eye(2) * (self.sig_rw ** 2 * dt)

    def update(self, z, sigma):
        """z=(x,y) 측정, sigma=측정 σ(m). 반환 True=수용 / False=게이팅 reject(이상치)."""
        z = np.asarray(z, float)
        if self.x is None:
            self.x = z.copy()
            self.P = np.eye(2) * 0.1
            return True
        R = np.eye(2) * (sigma * sigma)
        y = z - self.x                    # H = I (위치 직접관측)
        S = self.P + R
        d2 = float(y @ np.linalg.solve(S, y))
        if d2 > self.gate:
            return False
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K) @ self.P
        return True

    @property
    def pos(self):
        return None if self.x is None else (float(self.x[0]), float(self.x[1]))

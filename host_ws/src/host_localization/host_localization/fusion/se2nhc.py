"""SE(2) NHC 융합 — 공통 front-end + EKF backend (docs/fusion_benchmark_plan.md §1).

상태 x = [px, py, vx, vy, h_off] @ uwb_global (2D). heading h = imu_yaw + h_off.
모션 = 등속(CV, predict 선형). 비선형은 NHC·wheel pseudo-measurement에만.

핵심: **NHC(non-holonomic, zero-sideslip)** 가 위치(속도벡터 방향)↔헤딩(h_off)을 엮음.
  v_lat = -vx·sin h + vy·cos h ≈ 0  (이동 중에만; v<v* 시 skip)
  ∂v_lat/∂h_off = -(전진속도) → 빠를수록 h_off 강관측 = 위치궤적이 헤딩 보정(COG 앵커 내장).

새 외부 의존성 없음(numpy + chi2 상수표).
"""
import numpy as np

PX, PY, VX, VY, HOFF = range(5)
DIM = 5
CHI2_95 = {1: 3.841, 2: 5.991}        # scipy 회피용 상수


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class EKFSE2:
    """SE(2)+NHC EKF. per-robot 설정은 생성자 인자로(leader/follower 분기)."""

    def __init__(self,
                 q_pos=1e-6, q_vel=0.5, q_hoff=1e-4,      # 프로세스 잡음/sec (q_hoff: leader 크게)
                 uwb_sigma0=0.15, uwb_qf_ref=70.0,        # R_uwb(qf)=σ0·(ref/qf)
                 r_bev=0.05 ** 2,
                 r_nhc=0.05 ** 2, nhc_vmin=0.10,          # NHC: 측면속도 잡음, 속도게이트
                 r_wheel=0.05 ** 2, use_wheel=False,      # 팔로워만 use_wheel=True
                 huber_uwb=3.0,                           # UWB robust(soft) 임계(Mahalanobis)
                 bev_gate=4.0,                            # BEV hard-reject 임계(Mahalanobis) — ID-switch 방어
                 r_cog0=0.30 ** 2, cog_vmin=0.08,         # COG→h_off 측정(헤딩 통합, INS course-aiding)
                 cog_vref=0.20, cog_yaw_gate=0.30,        #   r_cog0 σ≈17°=skid COG 실측잡음(작으면 χ²가 정상COG 거부)
                 # ⚠ skid 로버는 COG(중심점 진행방향)≠헤딩(슬립) → COG 대부분 garbage, χ²가 깨끗한 직진만 수용.
                 p0=(1.0, 1.0, 1.0, 1.0, (np.pi) ** 2)):
        self.x = np.zeros(DIM)
        self.P = np.diag(p0).astype(float)
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_hoff]).astype(float)
        self.uwb_sigma0, self.uwb_qf_ref = float(uwb_sigma0), float(uwb_qf_ref)
        self.r_bev = float(r_bev)
        self.r_nhc, self.nhc_vmin = float(r_nhc), float(nhc_vmin)
        self.r_wheel, self.use_wheel = float(r_wheel), bool(use_wheel)
        self.huber_uwb = float(huber_uwb)
        self.bev_gate = float(bev_gate)           # BEV Mahalanobis hard-reject 임계(ID-switch 방어)
        self.r_cog0 = float(r_cog0)               # COG 측정 분산 @ cog_vref (rad²)
        self.cog_vmin = float(cog_vmin)           # 이 속도 미만이면 COG 무의미 → skip
        self.cog_vref = float(cog_vref)           # R 적응 기준속도(빠를수록 R 작음)
        self.cog_yaw_gate = float(cog_yaw_gate)   # |yaw_rate| 이상이면 회전 중 → skip
        self._cog_seen = False                    # COG로 h_off 앵커됐나(헤딩 출력 유효성)
        self._cog_boot = []                        # 부트스트랩용 최근 b_meas(cog-imu) — 연속 직진 확인
        self._flip_count = 0                       # 180° 자가복구: 연속 ~180° reject 카운트
        self.imu_yaw = 0.0
        self._init = False
        self.dbg = False          # 검증 §2: True면 NHC innovation 수집(런타임 비활성)
        self.dbg_nhc = []

    # ── 입력/헤딩 ────────────────────────────────────────────────────────
    def set_imu_yaw(self, yaw):
        self.imu_yaw = float(yaw)

    def heading(self):
        return wrap(self.imu_yaw + self.x[HOFF])

    # ── predict (선형 CV) + NHC ──────────────────────────────────────────
    def predict(self, dt):
        if dt <= 0.0:
            return
        F = np.eye(DIM)
        F[PX, VX] = dt
        F[PY, VY] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q * dt
        self._nhc()

    # ── 일반 EKF update ──────────────────────────────────────────────────
    def _update(self, H, innov, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.x[HOFF] = wrap(self.x[HOFF])
        I = np.eye(DIM)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T   # Joseph form(수치안정)
        return float(innov @ np.linalg.solve(S, innov))               # NIS

    # ── 측정 ─────────────────────────────────────────────────────────────
    def update_uwb(self, x, y, qf):
        if not self._init:
            self.x[PX], self.x[PY] = x, y
            self.x[VX] = self.x[VY] = 0.0
            self._init = True
            return None
        sig = self.uwb_sigma0 * (self.uwb_qf_ref / float(np.clip(qf, 1.0, 100.0)))
        R = np.eye(2) * (sig * sig)
        H = np.zeros((2, DIM)); H[0, PX] = 1.0; H[1, PY] = 1.0
        innov = np.array([x, y]) - H @ self.x
        # soft(Huber) 게이팅: md 크면 R 팽창(하드 reject 대신 다운웨이트)
        S = H @ self.P @ H.T + R
        md = np.sqrt(max(float(innov @ np.linalg.solve(S, innov)), 0.0))
        if md > self.huber_uwb:
            R = R * (md / self.huber_uwb) ** 2
        return self._update(H, innov, R)

    def update_bev(self, x, y):
        if not self._init:
            self.x[PX], self.x[PY] = x, y; self._init = True; return None
        H = np.zeros((2, DIM)); H[0, PX] = 1.0; H[1, PY] = 1.0
        R = np.eye(2) * self.r_bev
        innov = np.array([x, y]) - H @ self.x
        # ID-switch 방어: swap된 BEV(다른 로버 위치, 30~50cm 멀음)는 큰 Mahalanobis로 **hard reject**.
        # 공분산 적응: BEV 신뢰 시(P 작음) tight→swap reject / dropout 후(P 큼) loose→복귀 BEV 수용.
        # reject 시 UWB 태그 앵커로 coast(태그 불변 → 영구 ID-switch 불가). update_uwb Huber와 대칭.
        S = H @ self.P @ H.T + R
        md = np.sqrt(max(float(innov @ np.linalg.solve(S, innov)), 0.0))
        if md > self.bev_gate:
            return None
        return self._update(H, innov, R)

    def update_wheel(self, v_fwd):
        """팔로워 전진속도. v_fwd = vx·cos h + vy·sin h."""
        if not self._init or not self.use_wheel:
            return None
        h = self.heading(); c, s = np.cos(h), np.sin(h)
        vx, vy = self.x[VX], self.x[VY]
        hx = vx * c + vy * s
        H = np.zeros((1, DIM)); H[0, VX] = c; H[0, VY] = s
        H[0, HOFF] = -vx * s + vy * c
        return self._update(H, np.array([v_fwd - hx]), np.array([[self.r_wheel]]))

    def update_cog(self, cog, speed, yaw_rate=0.0):
        """COG(진행방향) → h_off 절대 측정 (INS course-aiding). 헤딩을 EKF로 통합.

        정지·회전 중엔 COG가 헤딩과 무관 → 게이트. z=cog, h=imu_yaw+h_off (h_off에 **선형**),
        R=속도적응(빠를수록 COG 신뢰 → R 작음). NHC 단독의 **180° 모호를 해소**(COG=실제
        진행방향이라 innov가 앞/뒤를 구분). 부트스트랩·슬루리밋은 공분산에서 *emergent*:
        초기 P[h_off]=π² 커서 첫 COG를 강하게 수용(자동 부트스트랩) → 수렴 후엔 P 작아
        나쁜 COG가 χ²에 걸리고 게인이 작다(자동 슬루). (손 게이트 4개를 이 한 메서드로 대체.)"""
        if not self._init or speed < self.cog_vmin or abs(yaw_rate) > self.cog_yaw_gate:
            return None
        if not self._cog_seen:
            # 부트스트랩: COG가 h_off 절대기준을 *처음* 확립. NHC(+BEV)가 미리 P[h_off]를 줄여
            # 과신하면(특히 follower IMU ~120° 오프셋) 큰 보정 COG를 χ²가 거부 → 영영 미앵커.
            # ⚠ 단 첫 *한* 샘플로 잡으면 그 순간이 후진/스킵이면 h_off 180° 오류(테일 가리킴).
            # → **연속 직진(최근 b_meas 3개가 원형일관)**일 때만 그 평균으로 부트스트랩(후진/노이즈 순간 회피).
            self._cog_boot.append(wrap(cog - self.imu_yaw))
            self._cog_boot = self._cog_boot[-3:]
            if len(self._cog_boot) >= 3:
                c = float(np.mean(np.cos(self._cog_boot))); s = float(np.mean(np.sin(self._cog_boot)))
                if np.hypot(c, s) > 0.90:                 # 원형일관(스프레드 <~25° = 직진)
                    self.x[HOFF] = float(np.arctan2(s, c))   # 원형평균 = 진행방향 기준
                    self.P[HOFF, :] = 0.0; self.P[:, HOFF] = 0.0
                    self.P[HOFF, HOFF] = (0.5) ** 2
                    self._cog_seen = True
            return None
        sig2 = self.r_cog0 * (self.cog_vref / max(speed, 1e-3)) ** 2     # 속도적응 R
        R = np.array([[sig2]])
        H = np.zeros((1, DIM)); H[0, HOFF] = 1.0
        innov = np.array([wrap(cog - self.imu_yaw - self.x[HOFF])])
        S = H @ self.P @ H.T + R
        if float(innov @ np.linalg.solve(S, innov)) > CHI2_95[1]:        # χ² 게이팅(이상 COG reject)
            # 180° 자가복구: yaw게이트 통과(=직진)인데 COG가 ~180° 어긋난 게 **연속**이면
            # h_off가 뒤집힌 것(잘못된 부트스트랩) → flip. 휠/지자계 없이 COG 일관성만으로 자가교정.
            # 턴의 일시적 큰 innov는 yaw게이트 + 연속요건(4회)으로 걸러짐.
            if abs(float(innov[0])) > 0.75 * np.pi:                      # |innov| > 135°(≈180°)
                self._flip_count += 1
                if self._flip_count >= 4:
                    self.x[HOFF] = wrap(self.x[HOFF] + np.pi)            # 180° 뒤집기
                    self.P[HOFF, :] = 0.0; self.P[:, HOFF] = 0.0
                    self.P[HOFF, HOFF] = (0.5) ** 2                      # 재불확실화(이후 COG 정제)
                    self._flip_count = 0
            else:
                self._flip_count = 0
            return None
        self._flip_count = 0                                            # 정상 수용 → 카운트 리셋
        return self._update(H, innov, R)

    def _nhc(self):
        """non-holonomic: 측면속도=0 (이동 중에만)."""
        if not self._init:
            return
        vx, vy = self.x[VX], self.x[VY]
        if np.hypot(vx, vy) < self.nhc_vmin:
            return
        h = self.heading(); c, s = np.cos(h), np.sin(h)
        v_lat = -vx * s + vy * c
        if self.dbg:
            self.dbg_nhc.append(v_lat)
        H = np.zeros((1, DIM)); H[0, VX] = -s; H[0, VY] = c
        H[0, HOFF] = -(vx * c + vy * s)
        self._update(H, np.array([-v_lat]), np.array([[self.r_nhc]]))

    # ── 출력 ─────────────────────────────────────────────────────────────
    @property
    def initialized(self):
        return self._init

    @property
    def speed(self):
        return float(np.hypot(self.x[VX], self.x[VY]))

    @property
    def heading_valid(self):
        return self._cog_seen          # COG로 h_off 앵커 전엔 헤딩 출력 부적합(identity 발행)

    @property
    def pose(self):
        return float(self.x[PX]), float(self.x[PY]), self.heading()

    @property
    def state(self):
        return self.x.copy()

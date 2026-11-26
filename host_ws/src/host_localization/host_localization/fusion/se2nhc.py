"""SE(2) EKF with non-holonomic (NHC) and course-over-ground (COG) heading aiding.

State x = [px, py, vx, vy, h_off, bx, by]; heading h = imu_yaw + h_off. The ceiling camera
defines the position frame; b is the UWB-to-camera frame offset (random walk in time and distance).
Process: constant velocity; ZUPT (zero velocity) while the wheels hold still.
Measurements: camera fix z = p (Mahalanobis gate); UWB z = p + b (Huber soft gate); NHC
v_lat = -vx sin h + vy cos h = 0 while moving; COG z = h (chi2 gate, wrong-branch hop);
box axis z = h mod obb_period; optional wheel speed.
"""
import math

import numpy as np

PX, PY, VX, VY, HOFF, BX, BY = range(7)
DIM = 7
_POS, _VEL, _UBIAS = [PX, PY], [VX, VY], [BX, BY]
CHI2_95 = {1: 3.841, 2: 5.991}


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def wrap_mod(a, period):
    """Wrap to [-period/2, period/2) — for angles only defined modulo `period`."""
    return (a + period / 2.0) % period - period / 2.0


class EKFSE2:
    """SE(2)+NHC EKF; per-robot settings are constructor arguments."""

    def __init__(self,
                 q_pos=1e-6, q_vel=0.5, q_hoff=1e-4,      # process noise per second
                 uwb_sigma0=0.15, uwb_qf_ref=70.0,        # R_uwb(qf) = sigma0 * (ref / qf)
                 r_bev=0.05 ** 2,
                 r_nhc=0.05 ** 2, nhc_vmin=0.10,          # NHC noise; speed gate [m/s]
                 r_wheel=0.05 ** 2, use_wheel=False,
                 huber_uwb=3.0,                           # UWB soft gate (Mahalanobis)
                 bev_gate=4.0,                            # camera hard gate (Mahalanobis)
                 r_cog0=0.30 ** 2,                        # COG heading noise at cog_vref
                 cog_vmin=0.05,                           # [m/s] slower segments give no course
                 straight_curvature=0.7,                  # [1/m] NHC/COG only on straighter runs
                 cog_vref=0.20, cog_yaw_gate=0.30,        # COG R reference speed [m/s]; yaw-rate gate [rad/s]
                 flip_min=4, flip_max=20,                 # wrong-branch recovery: consecutive rejects
                 r_obb=np.radians(8.0) ** 2,              # box-axis heading noise
                 obb_period=np.pi / 2,                    # box-axis ambiguity: pi/2 or pi
                 obb_prior_div=6.0,                       # box axis only if sigma_hoff < period/div
                 p0=(1.0, 1.0, 1.0, 1.0, (np.pi) ** 2),
                 ubias_sigma0=3.0,                        # [m] prior std of the UWB-camera offset
                 q_ubias_t=0.005 ** 2,                    # [m^2/s] offset random walk in time
                 q_ubias_s=0.02,                          # [m^2/m] ... and per metre travelled
                 bev_reacquire_n=20,                      # refused fixes in one steady track before re-acquiring
                 bev_track_step=0.15,                     # [m] max step within that track
                 uwb_identity_tol=0.35,                   # [m] camera vs corrected UWB; farther = another rover
                 zupt_psd=5e-6):                          # [m^2/s] zero-velocity measurement density
        self.x = np.zeros(DIM)
        self._p0 = tuple(float(v) for v in p0) + (float(ubias_sigma0) ** 2,) * 2
        self.P = np.diag(self._p0)
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_hoff, 0.0, 0.0]).astype(float)
        self.q_ubias_t, self.q_ubias_s = float(q_ubias_t), float(q_ubias_s)
        self.bev_reacquire_n, self.bev_track_step = int(bev_reacquire_n), float(bev_track_step)
        self.uwb_identity_tol = float(uwb_identity_tol)
        self.zupt_psd = float(zupt_psd)
        self.uwb_sigma0, self.uwb_qf_ref = float(uwb_sigma0), float(uwb_qf_ref)
        self.r_bev = float(r_bev)
        self.r_nhc, self.nhc_vmin = float(r_nhc), float(nhc_vmin)
        self.r_wheel, self.use_wheel = float(r_wheel), bool(use_wheel)
        self.huber_uwb = float(huber_uwb)
        self.bev_gate = float(bev_gate)
        self.r_cog0 = float(r_cog0)
        self.cog_vmin = float(cog_vmin)
        self.cog_vref = float(cog_vref)
        self.straight_curvature = float(straight_curvature)
        self.imu_wz = 0.0                          # IMU yaw rate [rad/s]
        self.cog_yaw_gate = float(cog_yaw_gate)
        self._cog_seen = False                    # h_off anchored by COG
        self._cog_boot = []                        # recent (cog - imu_yaw)
        self.flip_min, self.flip_max = int(flip_min), int(flip_max)
        self.r_obb, self.obb_period = float(r_obb), float(obb_period)
        self.obb_sigma_max = float(obb_period) / float(obb_prior_div)
        self._branch = np.pi                       # heading ambiguity step (obb_period once OBB is used)
        self._hop_k, self._hop_n = 0, 0            # wrong-branch candidate, consecutive rejects
        self._cog_ok = 0                           # COG updates accepted since the last (re)anchor
        self.imu_yaw = 0.0
        self._init = False                         # position initialised (by either source)
        self._camera_seen = False                  # position is in the camera frame
        self._ubias_init = False                   # UWB offset initialised
        self._refused, self._refused_at = 0, None  # steady track of gate-refused camera fixes
        self.camera_fresh = True                   # set by RobotTracker: camera sees the motion
        self.stationary = False                    # set by RobotTracker: h_off frozen
        self.zero_velocity = False                 # set by RobotTracker: ZUPT in predict
        self.dbg = False                           # collect NHC innovations (debug)
        self.dbg_nhc = []

    def set_imu_yaw(self, yaw, wz=0.0):
        self.imu_yaw = float(yaw)
        self.imu_wz = float(wz)

    def heading(self):
        return wrap(self.imu_yaw + self.x[HOFF])

    def predict(self, dt):
        if dt <= 0.0:
            return
        F = np.eye(DIM)
        F[PX, VX] = dt
        F[PY, VY] = dt
        self.x = F @ self.x
        Q = self.Q * dt
        if self.stationary:
            # q_hoff models yaw drift while driving; parked it would only inflate sigma(h_off).
            Q[HOFF, HOFF] = 0.0
        self.P = F @ self.P @ F.T + Q
        if self._ubias_init:
            walk = (self.q_ubias_t + self.q_ubias_s * self.speed) * dt
            self.P[BX, BX] += walk
            self.P[BY, BY] += walk
        if self.zero_velocity:
            self._zero_velocity_update(dt)
        self._nhc()

    def inflate_position(self, d):
        """Add d [m] of position sigma (the next camera fix passes the gate)."""
        self.P[PX, PX] += d * d
        self.P[PY, PY] += d * d

    def _zero_velocity_update(self, dt):
        """Velocity 0 with noise zupt_psd / dt (information rate independent of predict rate)."""
        if not self._init:
            return
        H = np.zeros((2, DIM)); H[0, VX] = 1.0; H[1, VY] = 1.0
        self._update(H, -self.x[_VEL], np.eye(2) * (self.zupt_psd / dt))

    def _update(self, H, innov, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.x[HOFF] = wrap(self.x[HOFF])
        I = np.eye(DIM)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T   # Joseph form
        return float(innov @ np.linalg.solve(S, innov))               # NIS

    def update_uwb(self, x, y, qf):
        """UWB fix = position + frame offset (the position itself before any camera fix)."""
        z = np.array([x, y], dtype=float)
        if not self._init:
            self.x[_POS] = z
            self.x[_VEL] = 0.0
            self._init = True
            return None
        sig = self.uwb_sigma0 * (self.uwb_qf_ref / float(np.clip(qf, 1.0, 100.0)))
        R = np.eye(2) * (sig * sig)
        if self._camera_seen and not self._ubias_init:
            self._init_uwb_offset(z, R)
            return None
        H = np.zeros((2, DIM)); H[0, PX] = 1.0; H[1, PY] = 1.0
        if self._ubias_init:
            H[0, BX] = 1.0; H[1, BY] = 1.0
        innov = z - H @ self.x
        # Huber soft gate: large Mahalanobis distance inflates R.
        S = H @ self.P @ H.T + R
        md = np.sqrt(max(float(innov @ np.linalg.solve(S, innov)), 0.0))
        if md > self.huber_uwb:
            R = R * (md / self.huber_uwb) ** 2
        return self._update(H, innov, R)

    def _init_uwb_offset(self, z, R):
        """First UWB fix in the camera frame: b = z - p (fully correlated with p)."""
        self.x[_UBIAS] = z - self.x[_POS]
        self.P[_UBIAS, :] = -self.P[_POS, :]
        self.P[:, _UBIAS] = -self.P[:, _POS]
        self.P[np.ix_(_UBIAS, _UBIAS)] = self.P[np.ix_(_POS, _POS)] + R
        self._ubias_init = True

    def _adopt_camera(self, z):
        """Put the position on a camera fix; velocity and UWB offset restart, h_off is kept."""
        for block in (_POS, _VEL, _UBIAS):
            self.P[block, :] = 0.0
            self.P[:, block] = 0.0
            for i in block:
                self.P[i, i] = self._p0[i]
        self.x[_POS] = z
        self.x[_VEL] = 0.0
        self.x[_UBIAS] = 0.0
        self._init = self._camera_seen = True
        self._ubias_init = False
        self._refused, self._refused_at = 0, None

    def _refused_camera_fix(self, z, uwb_fix):
        """One-off refusal = ID switch (ignore); a steady track UWB confirms is adopted."""
        steady = (self._refused_at is not None
                  and float(np.hypot(*(z - self._refused_at))) < self.bev_track_step)
        self._refused = self._refused + 1 if steady else 1
        self._refused_at = z
        if self._refused < self.bev_reacquire_n or not self._uwb_confirms(z, uwb_fix):
            return False
        self._adopt_camera(z)
        return True

    def _uwb_confirms(self, z, uwb_fix):
        """Camera fix z matches UWB minus offset (True without evidence against)."""
        if uwb_fix is None or not self._ubias_init:
            return True
        expected = np.asarray(uwb_fix, dtype=float) - self.x[_UBIAS]
        return float(np.hypot(*(z - expected))) < self.uwb_identity_tol

    def update_bev(self, x, y, uwb_fix=None):
        """Camera fix; uwb_fix = latest fresh UWB fix or None (identity check). True if used."""
        z = np.array([x, y], dtype=float)
        if not self._camera_seen:
            self._adopt_camera(z)
            return True
        H = np.zeros((2, DIM)); H[0, PX] = 1.0; H[1, PY] = 1.0
        R = np.eye(2) * self.r_bev
        innov = z - H @ self.x
        # ID-switch defence: the gate follows P (tight while trusted, loose after a dropout).
        S = H @ self.P @ H.T + R
        md = np.sqrt(max(float(innov @ np.linalg.solve(S, innov)), 0.0))
        if md > self.bev_gate:
            return self._refused_camera_fix(z, uwb_fix)
        self._refused, self._refused_at = 0, None
        self._update(H, innov, R)
        return True

    def bev_gate_distance(self, x, y):
        """(Mahalanobis distance, metres) of a camera fix; (inf, inf) before the first one."""
        if not self._camera_seen:
            return math.inf, math.inf
        z = np.array([x, y], dtype=float)
        H = np.zeros((2, DIM)); H[0, PX] = 1.0; H[1, PY] = 1.0
        innov = z - H @ self.x
        S = H @ self.P @ H.T + np.eye(2) * self.r_bev
        md = np.sqrt(max(float(innov @ np.linalg.solve(S, innov)), 0.0))
        return float(md), float(np.hypot(*innov))

    def update_bev_candidate(self, x, y, max_dist):
        """Other-class fix (class flip): used only inside the gate and max_dist [m]; never adopts."""
        md, dist = self.bev_gate_distance(x, y)
        if md > self.bev_gate or dist > max_dist:
            return False
        H = np.zeros((2, DIM)); H[0, PX] = 1.0; H[1, PY] = 1.0
        self._update(H, np.array([x, y], dtype=float) - H @ self.x, np.eye(2) * self.r_bev)
        return True

    def uwb_vouches(self, x, y, uwb_fix):
        """True when a fresh UWB fix minus the offset puts this rover at (x, y)."""
        if uwb_fix is None or not self._ubias_init:
            return False
        return self._uwb_confirms(np.array([x, y], dtype=float), uwb_fix)

    def update_wheel(self, v_fwd):
        """Wheel forward speed v_fwd = vx cos h + vy sin h."""
        if not self._init or not self.use_wheel:
            return None
        h = self.heading(); c, s = np.cos(h), np.sin(h)
        vx, vy = self.x[VX], self.x[VY]
        hx = vx * c + vy * s
        H = np.zeros((1, DIM)); H[0, VX] = c; H[0, VY] = s
        H[0, HOFF] = -vx * s + vy * c
        return self._update(H, np.array([v_fwd - hx]), np.array([[self.r_wheel]]))

    def update_cog(self, cog, speed, yaw_rate=0.0):
        """COG -> absolute h_off measurement; skipped when slow or turning. R shrinks with speed.
        Resolves the 180 deg ambiguity NHC leaves."""
        if (not self._init or self.stationary or speed < self.cog_vmin
                or abs(yaw_rate) > self.cog_yaw_gate):
            return None
        if not self._cog_seen:
            # Bootstrap only after three consistent samples (straight run).
            self._cog_boot.append(wrap(cog - self.imu_yaw))
            self._cog_boot = self._cog_boot[-3:]
            if len(self._cog_boot) >= 3:
                c = float(np.mean(np.cos(self._cog_boot))); s = float(np.mean(np.sin(self._cog_boot)))
                if np.hypot(c, s) > 0.90:
                    self._reanchor(float(np.arctan2(s, c)))
                    self._cog_seen = True
            return None
        sig2 = self.r_cog0 * (self.cog_vref / max(speed, 1e-3)) ** 2     # speed-adaptive R
        R = np.array([[sig2]])
        H = np.zeros((1, DIM)); H[0, HOFF] = 1.0
        innov = np.array([wrap(cog - self.imu_yaw - self.x[HOFF])])
        S = H @ self.P @ H.T + R
        if float(innov @ np.linalg.solve(S, innov)) > CHI2_95[1]:        # chi2 gate
            self._cog_rejected(float(innov[0]))
            return None
        self._hop_n = 0
        self._cog_ok += 1
        return self._update(H, innov, R)

    def _reanchor(self, h_off):
        """Set h_off absolutely: decorrelate, re-open P, reset counters."""
        self.x[HOFF] = wrap(h_off)
        self.P[HOFF, :] = 0.0; self.P[:, HOFF] = 0.0
        self.P[HOFF, HOFF] = (0.5) ** 2
        self._hop_n, self._cog_ok = 0, 0

    def _cog_rejected(self, innov):
        """Wrong-branch recovery: repeated rejects on the same other branch (180 deg, or +-90 deg
        with the box axis) hop h_off there; needed rejects grow with accepted COGs (reversing
        looks like a 180 deg error)."""
        n_branch = int(round(2.0 * np.pi / self._branch))
        k = int(np.round(innov / self._branch)) % n_branch
        on_branch = abs(wrap_mod(innov, self._branch)) < self._branch / 4.0
        if k == 0 or not on_branch:
            self._hop_n = 0
            return
        self._hop_n = self._hop_n + 1 if k == self._hop_k else 1
        self._hop_k = k
        if self._hop_n >= min(self.flip_max, self.flip_min + self._cog_ok):
            self._reanchor(self.x[HOFF] + k * self._branch)

    def update_obb_heading(self, axis_yaw):
        """Box axis -> h_off measurement modulo obb_period; only once COG-anchored and sigma is
        small. COG stays the branch authority; this tracks within it and works while stopped."""
        period = self.obb_period
        if not self._init or not self._cog_seen or self.P[HOFF, HOFF] > self.obb_sigma_max ** 2:
            return None
        H = np.zeros((1, DIM)); H[0, HOFF] = 1.0
        R = np.array([[self.r_obb]])
        innov = np.array([wrap_mod(axis_yaw - self.imu_yaw - self.x[HOFF], period)])
        S = H @ self.P @ H.T + R
        if float(innov @ np.linalg.solve(S, innov)) > CHI2_95[1]:        # bad box -> skip
            return None
        self._branch = period
        return self._update(H, innov, R)

    def _nhc(self):
        """Lateral velocity = 0 while moving; needs camera-seen motion and non-stationary wheels."""
        if not self._init or not self.camera_fresh or self.stationary:
            return
        vx, vy = self.x[VX], self.x[VY]
        speed = np.hypot(vx, vy)
        if speed < self.nhc_vmin or abs(self.imu_wz) > self.straight_curvature * speed:
            return                      # turning: the tracked point moves sideways
        h = self.heading(); c, s = np.cos(h), np.sin(h)
        v_lat = -vx * s + vy * c
        if self.dbg:
            self.dbg_nhc.append(v_lat)
        H = np.zeros((1, DIM)); H[0, VX] = -s; H[0, VY] = c
        H[0, HOFF] = -(vx * c + vy * s)
        self._update(H, np.array([-v_lat]), np.array([[self.r_nhc]]))

    @property
    def initialized(self):
        return self._init

    @property
    def in_camera_frame(self):
        """True once a camera fix placed the position (else bare UWB coordinates)."""
        return self._camera_seen

    @property
    def position_sigma(self):
        """[m] 1-sigma along the worst axis."""
        a, b, c = self.P[PX, PX], self.P[PY, PY], self.P[PX, PY]
        return float(math.sqrt(max(0.5 * (a + b) + math.sqrt(0.25 * (a - b) ** 2 + c * c), 0.0)))

    @property
    def uwb_offset(self):
        """(bx, by) UWB offset from the camera frame; None until learned."""
        return (float(self.x[BX]), float(self.x[BY])) if self._ubias_init else None

    @property
    def refused_track(self):
        """Length of the current steady track of refused camera fixes."""
        return self._refused

    @property
    def speed(self):
        return float(np.hypot(self.x[VX], self.x[VY]))

    @property
    def heading_valid(self):
        return self._cog_seen

    @property
    def pose(self):
        return float(self.x[PX]), float(self.x[PY]), self.heading()

    @property
    def state(self):
        return self.x.copy()

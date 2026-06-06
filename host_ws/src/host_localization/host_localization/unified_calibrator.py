#!/usr/bin/env python3
"""
unified_calibrator.py
Phase A: Intrinsic 캘리브레이션 (체커보드)
Phase B: BEV 통합 캘리브레이션 (ArUco + combined remap)

실행:
  python3 -m host_localization.unified_calibrator --phase intrinsic
  python3 -m host_localization.unified_calibrator --phase bev
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import yaml
from pupil_apriltags import Detector

from host_localization.camera_io import (
    DEFAULT_CONTROLS, open_camera as open_frozen_camera)

# ── 공유 경로 ──────────────────────────────────────────────────────────────
DEFAULT_CALIBRATION_DIR = os.path.join(str(Path.home()), '.factory_calibration')
DEFAULT_INTRINSIC_PATH  = os.path.join(DEFAULT_CALIBRATION_DIR, 'intrinsic.yaml')
DEFAULT_CALIBRATION_PATH = os.path.join(DEFAULT_CALIBRATION_DIR, 'calibration.yaml')
DEFAULT_REMAP_PATH       = os.path.join(DEFAULT_CALIBRATION_DIR, 'remap.npz')

# 캘리브레이션 포인트 파일 (패키지 기준 상대 경로)
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CAL_POINTS_PATH = os.path.join(_PKG_DIR, 'config', 'calibration_points.yaml')


def open_camera(index: int, width=1920, height=1080, fps=30):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 {index} 열기 실패")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_ZOOM, 100)
    return cap


# ═══════════════════════════════════════════════════════════════════════════
# Phase A: Intrinsic
# ═══════════════════════════════════════════════════════════════════════════

class IntrinsicCalibrator:
    def __init__(self, camera_index=2, board_cols=9, board_rows=6, square_mm=25.0):
        self.camera_index = camera_index
        self.board_cols = board_cols
        self.board_rows = board_rows
        self.square_m = square_mm / 1000.0

        # 3D 체커보드 좌표 (Z=0 평면)
        self.obj_p = np.zeros((board_cols * board_rows, 3), np.float32)
        self.obj_p[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
        self.obj_p *= self.square_m

        self.obj_pts = []
        self.img_pts = []
        self.img_size = None

    def run(self, output_path=DEFAULT_INTRINSIC_PATH, num_target=25):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cap = open_camera(self.camera_index)

        print(f"체커보드 {self.board_cols}×{self.board_rows} | 목표: {num_target}장")
        print("키: s=캡처  c=캘리브레이션  r=초기화  q=종료")

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.img_size is None:
                self.img_size = (gray.shape[1], gray.shape[0])

            found, corners = cv2.findChessboardCorners(
                gray, (self.board_cols, self.board_rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

            display = frame.copy()
            if found:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(display, (self.board_cols, self.board_rows),
                                          corners2, found)
                cv2.putText(display, "DETECTED - press 's' to capture",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Searching for chessboard...",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            count = len(self.obj_pts)
            cv2.putText(display, f"Captured: {count}/{num_target}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow('Intrinsic Calibration', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and found:
                self.obj_pts.append(self.obj_p)
                self.img_pts.append(corners2)
                print(f"  캡처 {len(self.obj_pts)}/{num_target}")
                if len(self.obj_pts) >= num_target:
                    print("목표 도달! 'c' 키로 캘리브레이션 실행")
            elif key == ord('c'):
                if len(self.obj_pts) < 4:
                    print("[경고] 최소 4장 필요")
                    continue
                self._calibrate(output_path)
            elif key == ord('r'):
                self.obj_pts.clear()
                self.img_pts.clear()
                print("초기화됨")

        cap.release()
        cv2.destroyAllWindows()

    def _calibrate(self, output_path):
        print(f"캘리브레이션 실행... ({len(self.obj_pts)}장)")
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_pts, self.img_pts, self.img_size, None, None)

        print(f"Reprojection Error: {ret:.4f} px  ({'OK' if ret <= 0.5 else 'NG'})")
        print(f"K:\n{K}")
        print(f"D: {D.flatten()}")

        data = {
            'camera_matrix': K.tolist(),
            'dist_coeffs': D.flatten().tolist(),
            'image_width': self.img_size[0],
            'image_height': self.img_size[1],
            'reprojection_error': float(ret),
            'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"저장: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Phase B: BEV 통합
# ═══════════════════════════════════════════════════════════════════════════

class BEVCalibrator:
    def __init__(self, camera_index=2,
                 intrinsic_path=DEFAULT_INTRINSIC_PATH,
                 cal_points_path=DEFAULT_CAL_POINTS_PATH,
                 pixels_per_meter=200):
        self.camera_index = camera_index
        self.intrinsic_path = intrinsic_path
        self.cal_points_path = cal_points_path
        self.ppm = pixels_per_meter

        self.K = None
        self.D = None
        self.H = None
        self.map1 = None
        self.map2 = None
        self.bev_w = 0
        self.bev_h = 0
        self.real_origin_x = 0.0
        self.real_origin_y = 0.0

        # AprilTag ID → world (x_m, y_m). ID로 직접 매칭(고유 ID 가정).
        self.world_by_id = {}
        self.world_corners = []   # BEV 캔버스 크기 계산용 (x,y) 목록
        self.cal_marker_ids = []  # 검출 대상 ID 목록

        # quad_decimate=1.0 (no downsampling) matters for the small ceiling tags.
        self._detector = Detector(families='tag36h11', nthreads=4,
                                  quad_decimate=1.0, decode_sharpening=0.25)

    def _load_intrinsic(self):
        """Load K, D from a charuco .npz (keys K, D) or an intrinsic.yaml
        (camera_matrix, dist_coeffs). The npz is the same intrinsic SoT that
        global_cam_node publishes as camera_info — no intermediate file."""
        if not os.path.isfile(self.intrinsic_path):
            raise FileNotFoundError(
                f"intrinsic 파일 없음: {self.intrinsic_path}\n"
                f"charuco npz(K,D) 또는 calibrate_intrinsic.py의 intrinsic.yaml 필요.")
        if self.intrinsic_path.endswith('.npz'):
            d = np.load(self.intrinsic_path)
            self.K = np.asarray(d['K'], dtype=np.float64).reshape(3, 3)
            self.D = np.asarray(d['D'], dtype=np.float64).reshape(1, -1)
            rms = float(d['rms']) if 'rms' in d.files else None
        else:
            with open(self.intrinsic_path) as f:
                data = yaml.safe_load(f)
            self.K = np.array(data['camera_matrix'], dtype=np.float64)
            self.D = np.array(data['dist_coeffs'], dtype=np.float64)
            rms = data.get('reprojection_error')
        rms_str = f'{rms:.4f}' if isinstance(rms, (int, float)) else '?'
        print(f"Intrinsic 로드 ({os.path.basename(self.intrinsic_path)}): "
              f"K ok, D={self.D.size}계수, reprojection_error={rms_str}")

    def _load_cal_points(self):
        with open(self.cal_points_path) as f:
            data = yaml.safe_load(f)
        markers = data['calibration_markers']
        # ID로 매칭하므로 ID는 고유해야 한다(각 모서리 다른 ID).
        self.world_by_id = {}
        self.cal_marker_ids = []
        for m in markers:
            mid = int(m['id'])
            if mid in self.world_by_id:
                raise ValueError(
                    f"calibration_points.yaml: ID {mid} 중복 — "
                    f"ID 매칭은 고유 ID를 요구한다.")
            self.world_by_id[mid] = (float(m['x']), float(m['y']))
            self.cal_marker_ids.append(mid)
        self.world_corners = list(self.world_by_id.values())
        print(f"캘리브레이션 마커 {len(self.world_corners)}개 로드 "
              f"(ID: {self.cal_marker_ids})")

    def _compute_bev_size(self):
        xs = [v[0] for v in self.world_corners]
        ys = [v[1] for v in self.world_corners]
        margin = 0.3
        self.real_origin_x = min(xs) - margin
        self.real_origin_y = min(ys) - margin
        real_w = (max(xs) - min(xs)) + 2 * margin
        real_h = (max(ys) - min(ys)) + 2 * margin
        self.bev_w = int(real_w * self.ppm)
        self.bev_h = int(real_h * self.ppm)
        print(f"BEV 크기: {self.bev_w}×{self.bev_h} px  ({real_w:.2f}×{real_h:.2f} m @ {self.ppm} px/m)")

    def _world_to_bev(self, x_m, y_m):
        px = (x_m - self.real_origin_x) * self.ppm
        py = self.bev_h - (y_m - self.real_origin_y) * self.ppm  # Y 반전
        return px, py

    def _bev_to_world(self, px, py):
        x_m = px / self.ppm + self.real_origin_x
        y_m = (self.bev_h - py) / self.ppm + self.real_origin_y
        return x_m, y_m

    def build_combined_remap(self):
        H_inv = np.linalg.inv(self.H)
        bev_xx, bev_yy = np.meshgrid(
            np.arange(self.bev_w, dtype=np.float32),
            np.arange(self.bev_h, dtype=np.float32))
        bev_pts = np.stack([bev_xx, bev_yy], axis=-1).reshape(-1, 1, 2)

        undist_pts = cv2.perspectiveTransform(bev_pts, H_inv).reshape(-1, 2)

        # Undistorted px -> normalized rays -> reapply the FULL distortion model
        # to get raw px. cv2.projectPoints handles ANY D length (the charuco
        # intrinsic here is 14-coeff rational+thin-prism+tilted). A hand-coded
        # plumb_bob (first 5 coeffs) silently over-distorts a rational D — its
        # large numerator k1..k3 belong over the k4..k6 denominator — and the BEV
        # blows up into a fisheye.
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        rays = np.column_stack([
            (undist_pts[:, 0] - cx) / fx,
            (undist_pts[:, 1] - cy) / fy,
            np.ones(len(undist_pts))]).astype(np.float64)
        raw, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), self.K, self.D)
        raw = raw.reshape(-1, 2)

        self.map1 = raw[:, 0].reshape(self.bev_h, self.bev_w).astype(np.float32)
        self.map2 = raw[:, 1].reshape(self.bev_h, self.bev_w).astype(np.float32)
        print("Combined remap 테이블 생성 완료")

    def _detect_markers(self, frame):
        """AprilTag(36h11) detection via pupil_apriltags — the official library,
        which (unlike cv2.aruco's apriltag dict) reliably decodes real-world tags.
        Returns marker centers keyed by ID: {id: [(cx, cy), ...]}. The center
        (not corners) is matched downstream; world coords are tag centers too.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = {}
        for det in self._detector.detect(gray):
            result.setdefault(int(det.tag_id), []).append(
                (float(det.center[0]), float(det.center[1])))
        return result

    def _match_by_id(self, detections):
        """검출 마커 중심을 AprilTag ID로 world 좌표와 짝짓는다.
        Returns (ids, pixel_pts, world_pts) — 같은 순서, 미검출 ID는 건너뜀.
        ID 매칭이라 이미지 회전/반전과 무관하게 축 방향이 보장된다."""
        ids, pixel_pts, world_pts = [], [], []
        for mid in self.cal_marker_ids:
            dets = detections.get(mid, [])
            if not dets:
                continue
            ids.append(mid)
            pixel_pts.append(dets[0])  # 고유 ID라 마커당 1개
            world_pts.append(self.world_by_id[mid])
        return ids, pixel_pts, world_pts

    def _compute_homography(self, detections):
        ids, pixel_pts, world_pts = self._match_by_id(detections)
        n = len(self.world_by_id)
        if len(pixel_pts) < n:
            print(f"[오류] {n}개 필요, {len(pixel_pts)}개 검출됨 (검출 ID: {ids})")
            return False

        for mid, (px, py), (wx, wy) in zip(ids, pixel_pts, world_pts):
            print(f"  매칭 ID {mid}: pixel=({px:.0f},{py:.0f}) ↔ world=({wx},{wy})")

        raw_pts = np.array(pixel_pts, dtype=np.float32)
        undist_pts = cv2.undistortPoints(
            raw_pts.reshape(-1, 1, 2), self.K, self.D, P=self.K).reshape(-1, 2)

        bev_pts = np.array([self._world_to_bev(x, y) for x, y in world_pts],
                           dtype=np.float32)

        self.H, mask = cv2.findHomography(undist_pts, bev_pts, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0
        print(f"H 계산 완료 (inliers: {inliers}/{n})")
        return True

    def _validate(self, detections):
        if self.H is None or self.map1 is None:
            print("[경고] 먼저 's' 키로 H와 remap 계산하세요")
            return

        ids, pixel_pts, world_pts = self._match_by_id(detections)
        n = len(self.world_by_id)
        if len(pixel_pts) < n:
            print(f"[경고] 검증 마커 부족 ({len(pixel_pts)}/{n})")
            return

        errors = []
        for mid, (px_raw, py_raw), (x_gt, y_gt) in zip(ids, pixel_pts, world_pts):
            pt = np.array([[[px_raw, py_raw]]], dtype=np.float32)
            undist = cv2.undistortPoints(pt, self.K, self.D, P=self.K)
            bev_px = cv2.perspectiveTransform(undist, self.H)[0][0]
            x_est, y_est = self._bev_to_world(bev_px[0], bev_px[1])
            err = np.sqrt((x_est - x_gt) ** 2 + (y_est - y_gt) ** 2)
            errors.append(err)
            print(f"  ID {mid}: gt=({x_gt:.3f},{y_gt:.3f}) "
                  f"est=({x_est:.3f},{y_est:.3f}) err={err*100:.1f}cm")
        if errors:
            print(f"평균 오차: {np.mean(errors)*100:.1f}cm  최대: {max(errors)*100:.1f}cm")

    def _save(self, output_path, remap_output_path):
        scale = 1.0 / self.ppm
        data = {
            'homography_matrix': self.H.flatten().tolist(),
            'bev_width': self.bev_w,
            'bev_height': self.bev_h,
            'scale_x': scale,
            'scale_y': scale,
            'pixels_per_meter': self.ppm,
            'real_origin_x': self.real_origin_x,
            'real_origin_y': self.real_origin_y,
            'y_axis_inverted': True,
            'remap_file': os.path.basename(remap_output_path),
            'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        np.savez_compressed(remap_output_path, map1=self.map1, map2=self.map2)
        print(f"저장: {output_path}")
        print(f"저장: {remap_output_path}")

    def run(self, output_path=DEFAULT_CALIBRATION_PATH,
            remap_output_path=DEFAULT_REMAP_PATH,
            timeout=20.0, min_samples=15):
        """헤드리스 자동 외부캘리 — 화면/키 입력 없음.
          - 프레임마다 AprilTag(36h11) 검출 → ID별 running-mean 누적
          - 4개 ID가 각각 min_samples 이상 모이면 → H+remap 계산·검증·저장 (return True)
          - timeout 안에 4개 ID 못 모으면 → 경고 + 미저장 종료 (return False)
        캘리 중 카메라·태그가 고정이라 프레임 간 누적이 유효(픽셀 위치 불변, 평균은 노이즈만 감소)."""
        self._load_intrinsic()
        self._load_cal_points()
        self._compute_bev_size()
        os.makedirs(DEFAULT_CALIBRATION_DIR, exist_ok=True)

        # FULL freeze (geometric + tuned photometric) so calib 픽셀 == runtime 픽셀.
        cap = open_frozen_camera(self.camera_index, controls=DEFAULT_CONTROLS,
                                 verbose=False)
        need = len(self.world_by_id)
        print(f"[auto] 카메라 열림 — 태그 {sorted(self.cal_marker_ids)} 누적 시작 "
              f"(timeout {timeout:.0f}s, 태그당 최소 {min_samples}샘플)")

        accum = {}          # id -> [mean_x, mean_y, n] (running mean)
        start = time.time()
        last_log = 0.0
        try:
            while time.time() - start < timeout:
                ret, frame = cap.read()
                if not ret:
                    continue
                for mid, pts in self._detect_markers(frame).items():
                    if mid in self.cal_marker_ids and pts:
                        cx, cy = pts[0]
                        if mid not in accum:
                            accum[mid] = [cx, cy, 1]
                        else:
                            mx, my, n = accum[mid]
                            n += 1
                            accum[mid] = [mx + (cx - mx) / n, my + (cy - my) / n, n]
                now = time.time()
                if now - last_log >= 1.0:
                    last_log = now
                    counts = {m: accum[m][2] for m in sorted(accum)}
                    print(f"[auto] 누적 {counts} ({len(accum)}/{need})")
                # 성공: 4개 ID 전부 min_samples 이상 → 바로 종료 후 계산
                if len(accum) == need and all(v[2] >= min_samples for v in accum.values()):
                    break
        finally:
            cap.release()

        if len(accum) < need:
            missing = [i for i in self.cal_marker_ids if i not in accum]
            print(f"[실패] {timeout:.0f}s 내 4개 ID 매칭 실패 — 누락 {missing} "
                  f"(검출 {sorted(accum)}). 조명/태그 배치/시야 확인 후 재시도. 저장 안 함.")
            return False

        dets = {mid: [(v[0], v[1])] for mid, v in accum.items()}
        if not self._compute_homography(dets):
            print("[실패] H 계산 실패. 저장 안 함.")
            return False
        self.build_combined_remap()
        self._validate(dets)
        self._save(output_path, remap_output_path)
        print("[완료] 자동 캘리 저장 완료.")
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Factory Calibration Tool')
    parser.add_argument('--phase', choices=['intrinsic', 'bev'], required=True)
    parser.add_argument('--camera', type=int, default=2)
    parser.add_argument('--board-cols', type=int, default=9)
    parser.add_argument('--board-rows', type=int, default=6)
    parser.add_argument('--square-mm', type=float, default=25.0)
    parser.add_argument('--ppm', type=int, default=200, help='pixels per meter')
    parser.add_argument('--intrinsic', default=DEFAULT_INTRINSIC_PATH)
    parser.add_argument('--cal-points', default=DEFAULT_CAL_POINTS_PATH)
    args = parser.parse_args()

    if args.phase == 'intrinsic':
        cal = IntrinsicCalibrator(
            camera_index=args.camera,
            board_cols=args.board_cols,
            board_rows=args.board_rows,
            square_mm=args.square_mm,
        )
        cal.run()
    else:
        cal = BEVCalibrator(
            camera_index=args.camera,
            intrinsic_path=args.intrinsic,
            cal_points_path=args.cal_points,
            pixels_per_meter=args.ppm,
        )
        cal.run()


if __name__ == '__main__':
    main()

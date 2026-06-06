import React, { useState, useRef, useLayoutEffect } from 'react';

// 천장 카메라 — BEV만(참고용). MJPEG = web_video_server(:8080), rosbridge 아님.
// type=ros_compressed: CompressedImage(JPEG) 토픽 직접 스트림(검증됨). 패딩 없이 채움.
const VIDEO_SERVER = 'http://localhost:8080';
const TOPIC = '/global_cam/processed'; // base 토픽명(+ros_compressed로 /compressed 해소)
const STREAM_URL = `${VIDEO_SERVER}/stream?topic=${TOPIC}&type=ros_compressed`;

// 측정 전 임시 종횡비(스트림 로드되면 실제 값으로 교체).
// 1.6 ≈ 실측 BEV(820×508=1.614)에 근접 → 첫 프레임 전 레이아웃 점프 최소화.
const DEFAULT_AR = 1.6;
const BORDER_X = 2; // .panel 좌우 보더(1px×2) — 패널 폭에 가산해 이미지영역이 정확히 AR

export default function CameraPanel() {
  const [failed, setFailed] = useState(false);
  const [ar, setAr] = useState(DEFAULT_AR);
  const [width, setWidth] = useState(null); // 패널에 직접 박는 폭(px)
  const viewRef = useRef(null);

  // 패널 '폭'을 (이미지영역 높이 × AR)로 결정적으로 지정한다.
  // 높이는 행이 정하고(폭과 독립) → 폭만 계산해 박으면 패널이 이미지에 딱 맞아
  // 패딩 0, 트윈이 남는 폭을 흡수. flex/grid intrinsic-sizing(원본 픽셀 누수)을 회피.
  useLayoutEffect(() => {
    const view = viewRef.current;
    if (!view) return undefined;
    const compute = () => {
      const h = view.clientHeight; // = 상단 행 높이 − 헤더 (이미지 영역 높이)
      if (h > 0) setWidth(h * ar + BORDER_X);
    };
    compute();
    const ro = new ResizeObserver(compute); // 창/행 높이 변화에 동적 추종
    ro.observe(view);
    return () => ro.disconnect();
  }, [ar]);

  // 실제 스트림 종횡비를 읽어 폭 계산에 반영 → 크롭·왜곡·레터박스 0(패널==이미지 비율).
  const onLoad = (e) => {
    const { naturalWidth: w, naturalHeight: h } = e.target;
    if (w > 0 && h > 0) setAr(w / h);
    setFailed(false);
  };

  return (
    <div
      className="panel camera-panel"
      style={width ? { width: `${width}px` } : undefined}
    >
      <div className="panel-head">
        <span className="panel-title">Global CAM (BEV)</span>
      </div>
      <div className="camera-view" ref={viewRef}>
        {failed ? (
          <div className="camera-fallback">
            <span className="camera-fallback-icon" aria-hidden="true">⌧</span>
            <span>No camera signal</span>
            <span className="camera-fallback-sub">check web_video_server (:8080)</span>
          </div>
        ) : (
          <img
            className="camera-img"
            src={STREAM_URL}
            alt="천장 카메라 BEV"
            onError={() => setFailed(true)}
            onLoad={onLoad}
          />
        )}
      </div>
    </div>
  );
}

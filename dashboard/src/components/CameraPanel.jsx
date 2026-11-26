import React, { useState, useRef, useLayoutEffect } from 'react';

// Ceiling camera (BEV) panel. Streamed as MJPEG from web_video_server (:8080), not rosbridge;
// type=ros_compressed streams the CompressedImage (JPEG) topic directly.
const VIDEO_SERVER = 'http://localhost:8080';
const TOPIC = '/global_cam/processed'; // base topic name (ros_compressed resolves /compressed)
const STREAM_URL = `${VIDEO_SERVER}/stream?topic=${TOPIC}&type=ros_compressed`;

// Aspect ratio used until the stream reports its real one (BEV is ~820x508 = 1.614).
const DEFAULT_AR = 1.6;
const BORDER_X = 2; // .panel left+right border (1px x 2), added so the image area has exactly AR

export default function CameraPanel() {
  const [failed, setFailed] = useState(false);
  const [ar, setAr] = useState(DEFAULT_AR);
  const [width, setWidth] = useState(null); // panel width in px
  const viewRef = useRef(null);

  // Set the panel width to (image-area height x AR): the row fixes the height, so the panel
  // fits the image exactly and the twin absorbs the remaining width (avoids flex/grid
  // intrinsic sizing from the source pixels).
  useLayoutEffect(() => {
    const view = viewRef.current;
    if (!view) return undefined;
    const compute = () => {
      const h = view.clientHeight; // image-area height (top row minus header)
      if (h > 0) setWidth(h * ar + BORDER_X);
    };
    compute();
    const ro = new ResizeObserver(compute); // follow window / row height changes
    ro.observe(view);
    return () => ro.disconnect();
  }, [ar]);

  // Use the stream's real aspect ratio: no crop, distortion or letterbox.
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

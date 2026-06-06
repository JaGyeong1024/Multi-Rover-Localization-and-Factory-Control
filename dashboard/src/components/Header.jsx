import React, { useLayoutEffect, useRef, useState } from 'react';

// 상단바: CBNU 로고 + 제목(한/영) 좌측 정렬 + E-STOP 우측.
// rosbridge/pose 지연·로버 신뢰는 시스템 상태/로버 상태 패널로 이동했다(레퍼런스).
const SUB_BASE = 12; // 영문 기준 폰트크기(px) — 측정 후 한글 줄 폭에 맞춰 스케일
const FIT_RATIO = 1.0; // 영문 줄 폭 = 한글 폭 × 이 비율 (1=꽉맞춤, <1=한 단계 작게)

export default function Header({ onEstop, estopActive }) {
  const mainRef = useRef(null); // 한글 줄(기준 폭)
  const subRef = useRef(null); // 영문 줄(폰트크기로 한글 폭에 맞춤)
  const [subSize, setSubSize] = useState(SUB_BASE);

  // 영문 줄 폭 = 한글 줄 폭이 되도록 폰트크기를 비례 스케일(text-align justify 자간왜곡 회피).
  useLayoutEffect(() => {
    const measure = () => {
      const main = mainRef.current;
      const sub = subRef.current;
      if (!main || !sub) return;
      const mainW = main.scrollWidth; // 한글 자연 폭(@18px)
      const subW = sub.scrollWidth; // 영문 자연 폭(@현재 크기)
      if (mainW > 0 && subW > 0) setSubSize((s) => s * (mainW / subW) * FIT_RATIO);
    };
    measure();
    // 웹폰트가 늦게 로드되면 폭이 달라지므로 로드 완료 후 한 번 더 맞춘다.
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(measure);
  }, []);

  return (
    <header className="header">
      <div className="header-brand">
        <img className="header-logo" src="/cbnu.png" alt="CBNU" />
        <div className="header-title">
          <span className="header-title-main" ref={mainRef}>공장 다중 로버 관제 콘솔</span>
          <span
            className="header-title-sub"
            ref={subRef}
            style={{ fontSize: `${subSize}px` }}
          >
            Factory Multi-Rover Control Console
          </span>
        </div>
      </div>

      <div className="header-spacer" />

      <button
        type="button"
        className={`estop-btn ${estopActive ? 'estop-btn-active' : ''}`}
        onClick={onEstop}
        title="명령레벨 abort. cmd_vel은 로버 소유 — 하드웨어 킬이 아님."
      >
        비상정지
        <span className="estop-sub">E-STOP</span>
      </button>
    </header>
  );
}

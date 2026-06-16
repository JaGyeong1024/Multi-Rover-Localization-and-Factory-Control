import React, { useLayoutEffect, useRef, useState } from 'react';

// 상단바: CBNU 로고 + 제목(한/영) 좌측 정렬 + E-STOP 우측.
// rosbridge/pose 지연·로버 신뢰는 시스템 상태/로버 상태 패널로 이동했다(레퍼런스).
const SUB_BASE = 12; // 영문 기준 폰트크기(px) — 측정 후 한글 줄 폭에 맞춰 스케일
const FIT_RATIO = 1.0; // 영문 줄 폭 = 한글 폭 × 이 비율 (1=꽉맞춤, <1=한 단계 작게)

export default function Header({ onEstop, estopActive }) {
  const mainRef = useRef(null); // 한글 줄(기준 폭)
  const subRef = useRef(null); // 영문 줄(폰트크기로 한글 폭에 맞춤)
  const [subSize, setSubSize] = useState(SUB_BASE);

  // 영문 줄 폭 = 한글 줄 폭이 되도록 폰트크기를 비례 스케일(자간왜곡 회피).
  // 결정적으로: 기준 px는 React state가 아니라 '실제 렌더된' 폰트크기(getComputedStyle)에서
  //   읽어 DOM과 어긋나지 않게 하고(=상태/DOM desync 제거), 폭은 fractional rect로 측정.
  //   계산이 멱등이라 폰트 로드 전 잘못 잡혀도 fonts.ready 재측정에서 정확히 수렴한다.
  useLayoutEffect(() => {
    const measure = () => {
      const main = mainRef.current;
      const sub = subRef.current;
      if (!main || !sub) return;
      const mainW = main.getBoundingClientRect().width; // 한글 자연 폭
      const subW = sub.getBoundingClientRect().width; // 영문 현재 폭
      const curPx = parseFloat(getComputedStyle(sub).fontSize); // 실제 렌더된 크기
      if (!(mainW > 0) || !(subW > 0) || !(curPx > 0)) return;
      const target = curPx * (mainW / subW) * FIT_RATIO;
      setSubSize((prev) => (Math.abs(prev - target) > 0.1 ? target : prev)); // 수렴 시 멈춤
    };
    const raf = requestAnimationFrame(measure); // 초기 레이아웃 후
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(measure); // 폰트 로드 후
    return () => cancelAnimationFrame(raf);
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

import React, { useLayoutEffect, useRef, useState } from 'react';
import { useMission, useEstop } from '../ros/useMission';

// Top bar: logo + Korean/English title on the left, E-STOP on the right.
const SUB_BASE = 12; // px, base size of the English line before it is scaled to the Korean width
const FIT_RATIO = 1.0; // English width = Korean width x this (1 = exact fit)

export default function Header() {
  const { engageEstop } = useMission();
  const { active: estopActive } = useEstop();
  const mainRef = useRef(null); // Korean line (reference width)
  const subRef = useRef(null); // English line (font size fitted to the Korean width)
  const [subSize, setSubSize] = useState(SUB_BASE);

  // Scale the English font so its line is as wide as the Korean one (no letter-spacing hacks).
  // The base size is read from the rendered font (getComputedStyle), not React state, so it
  // cannot drift from the DOM; the computation is idempotent and re-measured on fonts.ready.
  useLayoutEffect(() => {
    const measure = () => {
      const main = mainRef.current;
      const sub = subRef.current;
      if (!main || !sub) return;
      const mainW = main.getBoundingClientRect().width; // natural Korean width
      const subW = sub.getBoundingClientRect().width; // current English width
      const curPx = parseFloat(getComputedStyle(sub).fontSize); // rendered size
      if (!(mainW > 0) || !(subW > 0) || !(curPx > 0)) return;
      const target = curPx * (mainW / subW) * FIT_RATIO;
      setSubSize((prev) => (Math.abs(prev - target) > 0.1 ? target : prev)); // stop once converged
    };
    const raf = requestAnimationFrame(measure); // after first layout
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(measure); // after fonts load
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
        onClick={engageEstop}
        title="명령레벨 abort. cmd_vel은 로버 소유 — 하드웨어 킬이 아님."
      >
        비상정지
        <span className="estop-sub">E-STOP</span>
      </button>
    </header>
  );
}

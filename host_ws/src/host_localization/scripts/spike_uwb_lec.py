#!/usr/bin/env python3
"""Capture `lec` output and compare firmware POS with z-fixed range LSE.

This is a pre-council spike tool, not a runtime node. It sends `lec` to the
GrowSpace/DWM listener, captures DIST lines, parses per-anchor ranges, solves
a 2D least-squares position with tag z fixed, and writes a markdown summary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import statistics
import time

import numpy as np
import serial


DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / 'analysis' / 'lec_spike_log.md'


@dataclass
class AnchorRange:
    label: str
    anchor_id: str
    x: float
    y: float
    z: float
    distance: float


@dataclass
class LecSample:
    raw: str
    anchors: list[AnchorRange]
    pos_x: float
    pos_y: float
    pos_z: float
    qf: float
    lse_x: float | None = None
    lse_y: float | None = None
    diff_m: float | None = None


def parse_lec_line(line: str) -> LecSample | None:
    parts = [p.strip() for p in line.strip().split(',')]
    if len(parts) < 3 or parts[0] != 'DIST':
        return None
    try:
        n = int(parts[1])
    except ValueError:
        return None

    anchors: list[AnchorRange] = []
    i = 2
    for _ in range(n):
        if i + 5 >= len(parts) or not parts[i].startswith('AN'):
            return None
        try:
            anchors.append(AnchorRange(
                label=parts[i],
                anchor_id=parts[i + 1].upper(),
                x=float(parts[i + 2]),
                y=float(parts[i + 3]),
                z=float(parts[i + 4]),
                distance=float(parts[i + 5]),
            ))
        except ValueError:
            return None
        i += 6

    try:
        pos_i = parts.index('POS', i)
        pos_x = float(parts[pos_i + 1])
        pos_y = float(parts[pos_i + 2])
        pos_z = float(parts[pos_i + 3])
        qf = float(parts[pos_i + 4])
    except (ValueError, IndexError):
        return None

    return LecSample(
        raw=line.strip(),
        anchors=anchors,
        pos_x=pos_x,
        pos_y=pos_y,
        pos_z=pos_z,
        qf=qf,
    )


def solve_xy_z_fixed(
    anchors: list[AnchorRange],
    tag_z: float,
    initial_xy: tuple[float, float],
    max_iter: int = 20,
) -> tuple[float, float] | None:
    if len(anchors) < 3:
        return None

    xy = np.array(initial_xy, dtype=float)
    for _ in range(max_iter):
        residuals = []
        jacobian = []
        for a in anchors:
            dx = xy[0] - a.x
            dy = xy[1] - a.y
            dz = tag_z - a.z
            pred = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            if pred < 1e-6:
                continue
            residuals.append(pred - a.distance)
            jacobian.append([dx / pred, dy / pred])
        if len(residuals) < 2:
            return None
        r = np.array(residuals, dtype=float)
        j = np.array(jacobian, dtype=float)
        try:
            step, *_ = np.linalg.lstsq(j, -r, rcond=None)
        except np.linalg.LinAlgError:
            return None
        xy += step
        if float(np.linalg.norm(step)) < 1e-6:
            break
    return float(xy[0]), float(xy[1])


def classify_diff(diffs_m: list[float]) -> str:
    if not diffs_m:
        return 'lec failed or raw range unavailable'
    arr = np.array(diffs_m, dtype=float)
    median_cm = float(np.median(arr)) * 100.0
    p90_cm = float(np.percentile(arr, 90)) * 100.0
    std_cm = float(np.std(arr)) * 100.0
    if median_cm < 1.0 and p90_cm < 2.0:
        return 'raw range extractable; firmware POS is close to z-fixed LSE (plain LSE box)'
    if std_cm > 2.0 and median_cm < 5.0:
        return ('raw range extractable; POS-vs-LSE diff is time-varying '
                '(firmware time filter / EMA suspected)')
    if median_cm > 5.0:
        return 'raw range extractable; firmware POS differs materially from z-fixed LSE (firmware KF suspected)'
    return 'raw range extractable; POS-vs-LSE difference is moderate or mixed'


def write_markdown(
    path: Path,
    port: str,
    duration: float,
    tag_z: float,
    raw_lines: list[str],
    samples: list[LecSample],
) -> None:
    """Append-only log writer.

    First invocation creates the file with a top-level header. Every subsequent
    invocation appends a new `## Run @ ...` block, preserving prior runs so the
    council can compare repeated spikes (anchor moves, firmware reflash, etc.).
    """
    diffs = [s.diff_m for s in samples if s.diff_m is not None]
    parsed = [s for s in samples if s.diff_m is not None]
    summary = classify_diff([d for d in diffs if d is not None])
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    new_file = not path.exists()
    out: list[str] = []

    if new_file:
        out.extend([
            '# Spike-001 — lec Raw Range Feasibility Log',
            '',
            'Append-only log. Each `## Run @ ...` block is one spike invocation.',
            '',
        ])

    out.extend([
        f'## Run @ {ts}',
        '',
        '### Run Metadata',
        f'- Device: `{port}`',
        '- Command: `lec`',
        f'- Duration: {duration:.1f}s',
        f'- Tag physical z: {tag_z:.3f} m',
        '- Operator: ',
        '- Anchor setup changed since baseline bag: yes/no',
        '',
        '### Result Branch',
        f'- Interpretation: {summary}',
        '',
        '### Raw Output Samples (first 10)',
        '',
        '```text',
    ])
    out.extend(raw_lines[:10])
    out.extend([
        '```',
        '',
        '### Parsed Samples (first 30)',
        '',
        '| idx | anchors | pos_x | pos_y | pos_z | qf | lse_x | lse_y | diff_cm |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for idx, sample in enumerate(parsed[:30], start=1):
        diff_cm = sample.diff_m * 100.0 if sample.diff_m is not None else float('nan')
        out.append(
            f'| {idx} | {len(sample.anchors)} | {sample.pos_x:.3f} | '
            f'{sample.pos_y:.3f} | {sample.pos_z:.3f} | {sample.qf:.0f} | '
            f'{sample.lse_x:.3f} | {sample.lse_y:.3f} | {diff_cm:.2f} |'
        )

    out.extend(['', '### Summary'])
    out.append(f'- Parsed DIST lines: {len(parsed)}')
    if diffs:
        arr = np.array(diffs, dtype=float)
        out.append(f'- Median POS-vs-LSE diff: {np.median(arr) * 100.0:.2f} cm')
        out.append(f'- p90 POS-vs-LSE diff: {np.percentile(arr, 90) * 100.0:.2f} cm')
        out.append(f'- Std POS-vs-LSE diff: {np.std(arr) * 100.0:.2f} cm')
    else:
        out.append('- Median/p90/std POS-vs-LSE diff: n/a')
    out.extend([
        f'- Interpretation: {summary}',
        '',
        '### Council #001 Impact',
        '- [ ] Keep Council #001 as raw range root-cause vs correction map.',
        '- [ ] Narrow Council #001 to correction model selection.',
        '- Notes:',
        '',
        '---',
        '',
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'w' if new_file else 'a'
    with open(path, mode, encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')


def capture(args: argparse.Namespace) -> tuple[list[str], list[LecSample]]:
    raw_lines: list[str] = []
    samples: list[LecSample] = []
    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    try:
        time.sleep(0.3)
        ser.write(b'\r\r')
        time.sleep(0.2)
        ser.reset_input_buffer()
        ser.write(b'lec\r')
        time.sleep(0.5)  # let firmware persist NVM toggle and start streaming

        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            raw_lines.append(line)
            sample = parse_lec_line(line)
            if sample is None:
                continue
            xy = solve_xy_z_fixed(
                sample.anchors,
                args.tag_z,
                (sample.pos_x, sample.pos_y),
            )
            if xy is not None:
                sample.lse_x, sample.lse_y = xy
                sample.diff_m = float(np.hypot(
                    sample.lse_x - sample.pos_x,
                    sample.lse_y - sample.pos_y,
                ))
            samples.append(sample)
    finally:
        try:
            # Toggle lec OFF to restore listener stream state. les NVM toggle
            # is left as-is; uwb_reader_node._ensure_location_stream handles
            # its own startup probe.
            ser.write(b'lec\r')
            time.sleep(0.2)
        finally:
            ser.close()
    return raw_lines, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='/dev/ttyUWB')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--tag-z', type=float, default=0.146)
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    raw_lines, samples = capture(args)
    write_markdown(args.output, args.port, args.duration, args.tag_z, raw_lines, samples)

    diffs = [s.diff_m for s in samples if s.diff_m is not None]
    print(f'raw lines: {len(raw_lines)}')
    print(f'parsed DIST lines: {len(diffs)}')
    if diffs:
        arr = np.array(diffs, dtype=float)
        print(f'median diff: {np.median(arr) * 100.0:.2f} cm')
        print(f'p90 diff: {np.percentile(arr, 90) * 100.0:.2f} cm')
    print(f'wrote: {args.output}')


if __name__ == '__main__':
    main()

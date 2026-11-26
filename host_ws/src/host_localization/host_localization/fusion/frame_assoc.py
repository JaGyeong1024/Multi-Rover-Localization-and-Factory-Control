"""Per-frame camera association: which box of one frame is which rover (no rclpy).

Boxes are assigned by position (nearest first, metres), one box per robot and one robot per box:
  own       box of the robot's own class inside its BEV gate -> on_bev
  cross     box of another class inside the gate and within max_dist -> on_bev_candidate
  fallback  a robot with neither gets its best untaken own-class box outside other gates -> on_bev
A box wanted by two merged estimates (closer than 2 * max_dist) goes to the robot whose fresh
UWB fix vouches for it. No smoothing over time.
"""
from typing import Dict, Iterable, List, NamedTuple, Optional

OWN, CROSS, FALLBACK = 'own', 'cross', 'fallback'


class Box(NamedTuple):
    label: str              # YOLO class, mapped to a robot name
    conf: float
    x: float
    y: float
    yaw: Optional[float]    # box long axis (mod pi) or None


def best_own(boxes: List[Box], name: str) -> Optional[int]:
    """Index of the highest-confidence box labelled name (as in bev_pose), or None."""
    idx = [i for i, b in enumerate(boxes) if b.label == name]
    return max(idx, key=lambda i: boxes[i].conf) if idx else None


def _estimates_merge(fusions, a, b, max_dist):
    pa, pb = fusions[a].est.pose, fusions[b].est.pose
    return ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5 < 2.0 * max_dist


def associate(fusions: Dict[str, object], boxes: List[Box], max_dist: float,
              taken: Iterable[int] = ()) -> Dict[str, tuple]:
    """{robot: (box index, OWN | CROSS | FALLBACK)} for the robots in ``fusions``.

    ``taken``: box indices already consumed outside (robots still on the best-per-class topic).
    The fusions must already be advanced to the frame time (process_frame does it)."""
    taken = set(taken)
    gates = {}                                   # (robot, box) -> md, only inside the gate
    pairs = []
    for name, f in fusions.items():
        for i, b in enumerate(boxes):
            if i in taken:
                continue
            md, dist = f.bev_gate_distance(b.x, b.y)
            if md > f.est.bev_gate:
                continue
            gates[(name, i)] = md
            own = b.label == name
            if own or dist <= max_dist:
                pairs.append((dist, 0 if own else 1, name, i))
    # Box wanted by two merged estimates: a one-sided UWB vouch decides.
    by_box = {}
    for p in pairs:
        by_box.setdefault(p[3], []).append(p)
    keep = []
    for i, ps in by_box.items():
        names = [p[2] for p in ps]
        if len(ps) >= 2 and all(_estimates_merge(fusions, names[0], n, max_dist) for n in names[1:]):
            vouched = [p for p in ps if fusions[p[2]].uwb_vouches(boxes[i].x, boxes[i].y)]
            if len(vouched) == 1:
                ps = vouched
        keep.extend(ps)
    keep.sort()
    out, used = {}, set(taken)
    for _, rank, name, i in keep:
        if name in out or i in used:
            continue
        out[name] = (i, OWN if rank == 0 else CROSS)
        used.add(i)
    for name in fusions:
        if name in out:
            continue
        own = [i for i, b in enumerate(boxes) if b.label == name and i not in used
               and not any(gates.get((o, i)) is not None for o in fusions if o != name)]
        if own:
            i = max(own, key=lambda k: boxes[k].conf)
            out[name] = (i, FALLBACK)
            used.add(i)
    return out


def process_frame(fusions: Dict[str, object], participants: Iterable[str], boxes: List[Box],
                  t: float, rx: float, max_dist: float) -> Dict[str, tuple]:
    """Feed one camera frame to the participating robots. Robots of ``fusions`` that do not
    take part keep their best own-class box (they get it from the best-per-class topic), so
    that box is not offered to anyone else. Returns {robot: (box index, kind, used)}."""
    participants = [n for n in participants if n in fusions]
    legacy = [n for n in fusions if n not in participants]
    taken = [i for i in (best_own(boxes, n) for n in legacy) if i is not None]
    part = {n: fusions[n] for n in participants}
    for f in part.values():
        f.advance(t)
    result = {}
    for name, (i, kind) in associate(part, boxes, max_dist, taken).items():
        b = boxes[i]
        f = part[name]
        if kind == CROSS:
            used = f.on_bev_candidate(t, rx, b.x, b.y, max_dist)
        else:
            used = f.on_bev(t, rx, b.x, b.y, b.yaw)
        result[name] = (i, kind, used)
    return result

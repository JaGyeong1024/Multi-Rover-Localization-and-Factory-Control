"""Mission queue data model (ROS-free, unit-testable). Multi-part selection builds
the waiting list; the sequencer pulls pending items into capacity-N batches. The
node renders this into a MissionQueue msg (dashboard feed)."""
from dataclasses import dataclass, field


@dataclass
class Item:
    """One mission item = one part fetch with its derived lane stop node(s).

    Parallel pick: leader and follower stop at two DIFFERENT nodes per shelf
    (leader at shelf-center, follower 0.25 m to the leader's right). `lane_node`
    stays as the leader's node (the ordering/render key); `leader_node` /
    `follower_node` default to it when a single-node (legacy platooning) item is
    constructed, so old call sites and tests keep working unchanged.
    """
    id: str                 # "fetch_gear_01"
    part_id: str            # "gear_01"
    x: float                # part coord (uwb_global)
    y: float
    lane_node: int          # leader's stop node (== nearest node to the part; sort/render key)
    leader_node: int = -1   # leader pick node  (defaults to lane_node)
    follower_node: int = -1  # follower pick node (defaults to lane_node)
    status: str = 'pending'  # pending | active | done | failed
    part_state: str = 'QUEUED'  # QUEUED | ASSIGNED | LOADED | DELIVERED

    def __post_init__(self):
        if self.leader_node < 0:
            self.leader_node = self.lane_node
        if self.follower_node < 0:
            self.follower_node = self.lane_node


@dataclass
class MissionQueueModel:
    items: list = field(default_factory=list)

    def enqueue(self, item: Item):
        if any(it.id == item.id for it in self.items):
            return False
        self.items.append(item)
        return True

    def pending(self):
        return [it for it in self.items if it.status == 'pending']

    def by_id(self, item_id):
        for it in self.items:
            if it.id == item_id:
                return it
        return None

    def active_id(self):
        for it in self.items:
            if it.status == 'active':
                return it.id
        return ''

    def cancel(self, item_id):
        """Remove ONE item by id unless it is currently active (in-flight). Returns True
        if removed — lets the operator drop a single queued/finished part without clearing
        the whole queue. The active (being-fetched) item is protected."""
        it = self.by_id(item_id)
        if it is None or it.status == 'active':
            return False
        self.items = [x for x in self.items if x.id != item_id]
        return True

    def remove(self, item_id):
        """Unconditional removal by id (caller owns the cancel policy). True if removed."""
        n = len(self.items)
        self.items = [x for x in self.items if x.id != item_id]
        return len(self.items) < n

    def purge_finished(self):
        """Drop all done/failed items — used to auto-reset the queue when a mission completes,
        so the operator gets a clean, dispatchable queue without per-item cancelling."""
        self.items = [it for it in self.items if it.status not in ('done', 'failed')]

    def clear(self):
        self.items = []

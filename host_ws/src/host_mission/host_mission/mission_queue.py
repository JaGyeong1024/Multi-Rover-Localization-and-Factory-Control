"""Mission queue data model (ROS-free, unit-testable). Multi-part selection builds
the waiting list; the sequencer pulls pending items into capacity-N batches. The
node renders this into a MissionQueue msg (dashboard feed)."""
from dataclasses import dataclass, field


@dataclass
class Item:
    """One mission item = one part fetch with its derived lane stop node."""
    id: str                 # "fetch_gear_01"
    part_id: str            # "gear_01"
    x: float                # part coord (uwb_global)
    y: float
    lane_node: int          # nearest graph node to the part (the stop)
    status: str = 'pending'  # pending | active | done | failed
    part_state: str = 'QUEUED'  # QUEUED | ASSIGNED | LOADED | DELIVERED


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

    def clear(self):
        self.items = []

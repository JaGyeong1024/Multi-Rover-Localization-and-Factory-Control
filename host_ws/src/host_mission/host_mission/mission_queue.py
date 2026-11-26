"""Mission queue data model (ROS-free): the waiting list the sequencer pulls batches from."""
from dataclasses import dataclass, field


def _unset(target):
    """True for the 'not given' markers (None or a negative node id)."""
    return target is None or (isinstance(target, int) and target < 0)


@dataclass
class Item:
    """One part fetch with its leader and follower stop targets (opaque goal keys); ``lane_node``
    is the leader's target, which ``leader_node`` / ``follower_node`` default to."""
    id: str                 # "fetch_gear_01"
    part_id: str            # "gear_01"
    x: float                # part coord (uwb_global)
    y: float
    # Stop targets are opaque hashable keys: lane node ids (int) or goal names (str).
    lane_node: object          # leader's stop target (sort/render key)
    leader_node: object = -1   # leader pick target  (defaults to lane_node)
    follower_node: object = -1  # follower load target (defaults to lane_node)
    status: str = 'pending'  # pending | active (done: transient, removed on delivery)
    part_state: str = 'QUEUED'  # QUEUED | ASSIGNED | LOADED | DELIVERED

    def __post_init__(self):
        if _unset(self.leader_node):
            self.leader_node = self.lane_node
        if _unset(self.follower_node):
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
        """Remove ONE waiting item by id; an active item is protected. True if removed."""
        it = self.by_id(item_id)
        if it is None or it.status == 'active':
            return False
        self.items = [x for x in self.items if x.id != item_id]
        return True

    def remove(self, item_id):
        n = len(self.items)
        self.items = [x for x in self.items if x.id != item_id]
        return len(self.items) < n

    def clear_waiting(self):
        """Operator 'clear': drop the waiting parts only; the running trip is untouched."""
        self.items = [it for it in self.items if it.status == 'active']

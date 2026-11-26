"""ReconnectingCapture policy tests (pure Python: no camera, no ROS graph)."""
from host_localization.camera_io import ReconnectingCapture


class FakeCap:
    def __init__(self, fail_reads=0):
        self.fail_reads = fail_reads
        self.released = False

    def isOpened(self):
        return not self.released

    def release(self):
        self.released = True

    def read(self):
        if self.fail_reads > 0:
            self.fail_reads -= 1
            return False, None
        return True, 'frame'


class Rig:
    def __init__(self, opens):
        self.opens = list(opens)   # per open: FakeCap or Exception
        self.calls = 0
        self.t = 0.0
        self.events = []

    def open(self):
        self.calls += 1
        item = self.opens.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def make(self, threshold=3, retry=2.0):
        return ReconnectingCapture(self.open, lambda: self.t, threshold, retry,
                                   lambda k, m: self.events.append(k))


def test_no_reopen_below_threshold():
    first = FakeCap(fail_reads=2)
    rig = Rig([first])
    cap = rig.make(threshold=3)
    assert [cap.read()[0] for _ in range(3)] == [False, False, True]
    assert rig.calls == 1 and not first.released and rig.events == []


def test_reopen_after_threshold_recovers():
    first = FakeCap(fail_reads=99)
    second = FakeCap()
    rig = Rig([first, second])
    cap = rig.make(threshold=3)
    assert [cap.read()[0] for _ in range(3)] == [False] * 3
    assert rig.calls == 2 and first.released          # reopened through the same open_fn
    assert cap.read() == (True, 'frame')
    assert rig.events == ['reopen', 'recovered']


def test_failed_reopen_retries_bounded():
    first = FakeCap(fail_reads=99)
    rig = Rig([first, OSError('gone'), OSError('gone'), FakeCap()])
    cap = rig.make(threshold=2, retry=2.0)
    for _ in range(10):                               # t=0: one attempt, then throttled
        cap.read()
    assert rig.calls == 2
    rig.t = 2.1
    cap.read()
    assert rig.calls == 3
    rig.t = 4.3
    cap.read()
    assert rig.calls == 4
    assert cap.read() == (True, 'frame')
    assert rig.events.count('reopen_failed') == 2 and rig.events[-1] == 'recovered'

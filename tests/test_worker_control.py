import json
import pytest
import server

class FakeProcess:
    killed = False
    def poll(self):
        return -9 if self.killed else None
    def kill(self):
        self.killed = True
    def wait(self, timeout=None):
        return -9 if self.killed else 0

def test_cancellation_stops_worker(monkeypatch, tmp_path):
    process = FakeProcess()
    monkeypatch.setattr(server.subprocess, 'Popen', lambda *a, **k: process)
    server.cancel_requested.set()
    try:
        with pytest.raises(RuntimeError, match='canceled'):
            server.run_worker(tmp_path/'document', {}, {})
        assert process.killed
    finally:
        server.cancel_requested.clear()

def test_hard_timeout_stops_worker(monkeypatch, tmp_path):
    process = FakeProcess()
    monkeypatch.setattr(server.subprocess, 'Popen', lambda *a, **k: process)
    monkeypatch.setenv('ADP_REQUEST_TIMEOUT', '1')
    times = iter([0, 2])
    monkeypatch.setattr(server.time, 'monotonic', lambda: next(times))
    with pytest.raises(RuntimeError, match='timeout'):
        server.run_worker(tmp_path/'document', {}, {})
    assert process.killed

def test_result_is_returned_only_after_worker_completion(monkeypatch, tmp_path):
    process = FakeProcess()
    process.poll = lambda: 0
    (tmp_path/'result.json').write_text(json.dumps({'result': {'data': {'id':'001'}}}))
    monkeypatch.setattr(server.subprocess, 'Popen', lambda *a, **k: process)
    assert server.run_worker(tmp_path/'document', {}, {}) == {'data': {'id':'001'}}

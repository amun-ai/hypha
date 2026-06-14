"""F6 Phase 1: autoscaling must only act on the leader replica (P2).

Every replica runs the autoscaling monitor loop, but only the leader may
actually scale — otherwise N replicas independently (and redundantly) scale the
same app. These tests assert _check_and_scale short-circuits on a non-leader
before doing any work, and proceeds on the leader.

No mocks of the system under test: a minimal real app_controller stub provides
store.is_leader(); the AutoscalingManager is the real class.
"""
import pytest

from hypha.apps import AutoscalingManager

pytestmark = pytest.mark.asyncio


class _FakeStore:
    def __init__(self, leader):
        self._leader = leader

    def is_leader(self):
        return self._leader


class _FakeController:
    def __init__(self, leader):
        self.store = _FakeStore(leader)


async def test_check_and_scale_short_circuits_when_not_leader():
    mgr = AutoscalingManager(_FakeController(leader=False))
    queried = []

    async def _fake_get_app_instances(app_id):
        queried.append(app_id)
        return []

    mgr._get_app_instances = _fake_get_app_instances
    await mgr._check_and_scale("app-1", None, {})
    assert queried == [], "non-leader must not query/scale at all"


async def test_check_and_scale_proceeds_when_leader():
    mgr = AutoscalingManager(_FakeController(leader=True))
    queried = []

    async def _fake_get_app_instances(app_id):
        queried.append(app_id)
        return []  # 0 instances → method returns after the leader gate

    mgr._get_app_instances = _fake_get_app_instances
    await mgr._check_and_scale("app-1", None, {})
    assert queried == ["app-1"], "leader must proceed past the gate and query instances"

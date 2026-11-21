"""滞回 / 时间稳定性 / 并发 / 性能 / 错误分类等验收测试（新增，不改既有测试）。"""

import math
import threading
import time

import pytest

import lod
from lod import (
    InvalidDistanceError,
    InvalidThresholdError,
    LodSelector,
    UnknownObjectError,
)


# --------------------------------------------------------------------------- #
# 错误可区分
# --------------------------------------------------------------------------- #
def test_lod_error_types_are_distinct():
    with pytest.raises(InvalidThresholdError):
        LodSelector(4, 0.0)
    with pytest.raises(InvalidThresholdError):
        LodSelector(0, 1.0)
    with pytest.raises(InvalidThresholdError):
        LodSelector(4, -1.0)
    with pytest.raises(InvalidThresholdError):
        LodSelector(4, math.nan)
    with pytest.raises(InvalidThresholdError):
        LodSelector(4, math.inf)

    s = LodSelector(4, 1.0)
    with pytest.raises(InvalidDistanceError):
        s.select(-0.001, 1.0)
    with pytest.raises(InvalidDistanceError):
        s.select(math.nan, 1.0)
    with pytest.raises(InvalidDistanceError):
        s.select(math.inf, 1.0)
    with pytest.raises(InvalidDistanceError):
        s.select(1.0, math.nan)
    with pytest.raises(InvalidDistanceError):
        s.select(1.0, -1.0)

    # 三类错误彼此可区分，且分别兼容 ValueError / KeyError 的既有约定。
    assert issubclass(InvalidThresholdError, ValueError)
    assert issubclass(InvalidDistanceError, ValueError)
    assert issubclass(UnknownObjectError, KeyError)
    with pytest.raises(UnknownObjectError):
        s.select(1.0, 1.0, obj="ghost")
    with pytest.raises(KeyError):
        s.select(1.0, 1.0, obj="ghost")
    with pytest.raises(UnknownObjectError):
        s.unregister("ghost")


# --------------------------------------------------------------------------- #
# 返回值范围与类型
# --------------------------------------------------------------------------- #
def test_lod_level_always_in_range():
    s = LodSelector(4, 1.0)
    for d in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 1e3, 1e9]:
        level = s.select(d, 1.0)
        assert isinstance(level, int)
        assert 0 <= level < 4


def test_lod_single_level():
    s = LodSelector(1, 1.0)
    for d in [0.0, 1.0, 1e6]:
        assert s.select(d, 1.0) == 0


def test_lod_threshold_boundary_is_strict():
    # SSE == threshold 时不允许降级（严格小于才升档）。
    s = LodSelector(4, 1.0)
    assert s.select(1.0, 1.0) == 0       # SSE == t
    assert s.select(1.0 + 1e-9, 1.0) >= 1  # SSE < t
    # 每一档进入边界同样是严格小于：SSE == t/2 停在 1。
    s2 = LodSelector(4, 1.0)
    assert s2.select(2.0, 1.0) == 1


def test_lod_monotone_full_sweep():
    s = LodSelector(6, 1.0)
    prev = -1
    for i in range(0, 200):
        level = s.select(2.0 ** (i / 10.0), 1.0)
        assert level >= prev
        assert 0 <= level < 6
        prev = level


# --------------------------------------------------------------------------- #
# 滞回 / 时间稳定性
# --------------------------------------------------------------------------- #
def test_lod_hysteresis_no_flapping_plus_minus_5pct():
    s = LodSelector(4, 1.0)
    # 先把对象稳定送到等级 2（进入边界 distance=2，SSE=t/2）。
    assert s.select(4.0, 1.0) == 2
    anchor = 4.0
    switches = 0
    last = 2
    # 在锚点附近 ±5% 反复抖动，等级不得来回跳变。
    for i in range(200):
        factor = 1.0 + (0.05 if i % 2 == 0 else -0.05)
        level = s.select(anchor * factor, 1.0)
        if level != last:
            switches += 1
            last = level
    assert switches <= 1  # 有上界：整个抖动序列至多一次切换


def test_lod_hysteresis_switch_count_bounded_per_object():
    s = LodSelector(4, 1.0)
    s.select(4.0, 1.0)  # 进入等级 2
    near = 4.0
    level_seq = [s.select(near * (1.0 + (0.05 if i % 2 else -0.05)), 1.0)
                 for i in range(500)]
    # 相邻调用的切换总次数有上界（滞回带内不允许反复横跳）。
    switches = sum(1 for a, b in zip(level_seq, level_seq[1:]) if a != b)
    assert switches <= 1


def test_lod_hysteresis_eventually_drops_when_far_enough():
    s = LodSelector(4, 1.0)
    assert s.select(4.0, 1.0) == 2
    # 靠近但未越过滞回带（降档需 SSE >= 1.1 * t/2，即 distance <= 2/1.1 ≈ 1.818）。
    assert s.select(1.9, 1.0) == 2
    # 足够近时仍能降回精细档，滞回不是死锁。
    assert s.select(1.0, 1.0) <= 1


def test_lod_hysteresis_independent_per_object():
    s = LodSelector(4, 1.0)
    s.register("a")
    s.register("b")
    assert s.select(4.0, 1.0, obj="a") == 2
    assert s.select(0.5, 1.0, obj="b") == 0
    # 两对象状态互不影响，且抖动中各自保持等级。
    assert s.select(4.0 * 0.96, 1.0, obj="a") == 2
    assert s.select(0.5 * 1.04, 1.0, obj="b") == 0


# --------------------------------------------------------------------------- #
# 注册 / 无 LOD
# --------------------------------------------------------------------------- #
def test_lod_registered_object_forms_equivalent():
    s = LodSelector(4, 1.0)
    s.register("o")
    assert s.select(4.0, 1.0, "o") == 2
    assert s.select("o", 4.0, 1.0) == 2
    assert s.select(4.0, 1.0, obj="o") == 2
    assert s.select(4.0, 1.0, object_id="o") == 2


def test_lod_unregister_raises_and_reuse():
    s = LodSelector(4, 1.0)
    s.register("o")
    s.select(4.0, 1.0, "o")
    s.unregister("o")
    with pytest.raises(UnknownObjectError):
        s.select(4.0, 1.0, "o")
    s.register("o")
    assert s.select(4.0, 1.0, "o") == 2  # 重新注册后可用，状态已重置


def test_lod_object_without_lod_returns_original():
    s = LodSelector(4, 1.0)
    s.register("static", levels=0, original=0)
    for d in [0.0, 1.0, 100.0, 1e9]:
        assert s.select(d, 1.0, "static") == 0
    s.register("static2", levels=0, original=2)
    assert s.select(1e9, 1.0, "static2") == 2


# --------------------------------------------------------------------------- #
# 并发安全
# --------------------------------------------------------------------------- #
def test_lod_concurrent_calls_safe_and_bounded():
    s = LodSelector(8, 1.0)
    n_threads = 8
    per_thread = 2000
    errors = []

    def worker(tid):
        try:
            obj = "obj-%d" % tid
            s.register(obj)
            rng_state = tid + 1
            for i in range(per_thread):
                # 简单确定性 LCG，避免依赖 random 的线程行为。
                rng_state = (1103515245 * rng_state + 12345) & 0x7FFFFFFF
                d = 0.1 + (rng_state % 10000) / 10.0
                level = s.select(d, 1.0, obj)
                if not (0 <= level < 8):
                    raise AssertionError("level out of range: %r" % (level,))
        except BaseException as exc:  # noqa: BLE001 - 测试中回传任意异常
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


# --------------------------------------------------------------------------- #
# 性能：10 万次选择 < 200 ms
# --------------------------------------------------------------------------- #
def test_lod_100k_selections_under_200ms():
    s = LodSelector(4, 1.0)
    distances = [0.5 + (i % 500) * 0.1 for i in range(100_000)]
    start = time.perf_counter()
    for d in distances:
        level = s.select(d, 1.0)
        assert 0 <= level < 4
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 200.0, "100k selections took %.1f ms" % elapsed_ms


def test_lod_100k_registered_object_under_200ms():
    s = LodSelector(4, 1.0)
    s.register("hero")
    distances = [0.5 + (i % 500) * 0.1 for i in range(100_000)]
    start = time.perf_counter()
    for d in distances:
        s.select(d, 1.0, "hero")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 200.0, "100k registered selections took %.1f ms" % elapsed_ms

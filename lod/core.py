"""按屏幕空间误差选择 LOD 的选择器。

规则：
- SSE = radius / distance；SSE 严格小于阈值才允许使用更粗（索引更大）的等级，
  边界相等时不降级，等级随距离单调不减。
- 每个对象维护自己的当前等级，带滞回：升档即时生效，降档必须越过进入边界
  10%（覆盖距离 ±5% 抖动），因此阈值附近抖动不会来回跳变。
- 选择器内部状态加锁，支持多线程并发调用。
"""

import math
import threading
from numbers import Real

__all__ = [
    "LodSelector",
    "LodError",
    "InvalidThresholdError",
    "InvalidDistanceError",
    "UnknownObjectError",
]


class LodError(Exception):
    """LOD 选择器相关错误的基类。"""


class InvalidThresholdError(LodError, ValueError):
    """非法的 levels 或 threshold（同时是 ValueError）。"""


class InvalidDistanceError(LodError, ValueError):
    """非法的 distance / radius：负数或非有限值（同时是 ValueError）。"""


class UnknownObjectError(LodError, KeyError):
    """访问未注册（或已注销）的对象（同时是 KeyError）。"""


# 降档（回到更精细等级）需要越过进入边界的比例；10% 余量覆盖 ±5% 距离抖动。
_HYSTERESIS = 1.1


class _State:
    __slots__ = ("level", "levels")

    def __init__(self, levels, level=0):
        # levels == 0 表示该对象没有 LOD，select 始终返回其原等级。
        self.levels = levels
        self.level = level


def _is_real(value):
    return isinstance(value, Real) and not isinstance(value, bool)


class LodSelector:
    """带每对象状态、滞回与线程安全的 LOD 选择器。

    参数:
        levels: LOD 等级总数，必须为 >= 1 的整数；返回值恒在 [0, levels)。
        threshold: 屏幕空间误差阈值，必须为有限正数。
    """

    def __init__(self, levels, threshold):
        if not isinstance(levels, int) or isinstance(levels, bool) or levels < 1:
            raise InvalidThresholdError(
                "levels must be a positive integer, got %r" % (levels,)
            )
        if not _is_real(threshold) or not math.isfinite(threshold) or threshold <= 0:
            raise InvalidThresholdError(
                "threshold must be a finite positive number, got %r" % (threshold,)
            )
        self.levels = levels
        self.threshold = float(threshold)
        self._lock = threading.RLock()
        self._states = {}
        # 未指定对象时使用的默认对象状态，保持 select(distance, radius) 旧用法。
        self._default = _State(levels)
        self._states[None] = self._default

    # ------------------------------------------------------------------ #
    # 对象注册
    # ------------------------------------------------------------------ #
    def register(self, obj, levels=None, original=0):
        """注册对象并初始化其每对象状态。

        levels 缺省使用选择器的等级数；levels=0 表示该对象没有 LOD，
        之后 select 始终返回 original（原等级）。
        """
        if obj is None:
            raise InvalidThresholdError("object id must not be None")
        if levels is None:
            levels = self.levels
        if not isinstance(levels, int) or isinstance(levels, bool) or levels < 0:
            raise InvalidThresholdError(
                "per-object levels must be a non-negative int, got %r" % (levels,)
            )
        if not isinstance(original, int) or isinstance(original, bool) or original < 0:
            raise InvalidThresholdError(
                "original level must be a non-negative int, got %r" % (original,)
            )
        if levels > 0 and original >= levels:
            raise InvalidThresholdError(
                "original level %d out of range [0, %d)" % (original, levels)
            )
        with self._lock:
            self._states[obj] = _State(levels, original)
        return obj

    add_object = register

    def unregister(self, obj):
        """注销对象；对未注册对象抛 UnknownObjectError。"""
        if obj is None:
            raise UnknownObjectError("cannot unregister the default object")
        with self._lock:
            if obj not in self._states:
                raise UnknownObjectError(repr(obj))
            del self._states[obj]
        return obj

    remove_object = unregister

    # ------------------------------------------------------------------ #
    # 选择
    # ------------------------------------------------------------------ #
    def select(self, distance, radius=None, obj=None, **kwargs):
        """返回对象当前应使用的 LOD 等级，整数且落在 [0, levels)。

        支持三种调用形式：
            select(distance, radius)
            select(distance, radius, obj) / select(distance, radius, obj=...)
            select(obj, distance, radius)
        """
        if obj is None:
            obj = kwargs.get("object_id", kwargs.get("id"))

        # select(obj, distance, radius) 对象在前的形式：首参不是数值。
        if not _is_real(distance):
            if radius is None or not _is_real(radius):
                raise InvalidDistanceError(
                    "distance must be a finite non-negative number"
                )
            obj, distance, radius = distance, radius, obj
        elif radius is None:
            raise InvalidDistanceError("radius is required")

        if not _is_real(distance) or not math.isfinite(distance) or distance < 0:
            raise InvalidDistanceError(
                "distance must be a finite non-negative number, got %r" % (distance,)
            )
        if not _is_real(radius) or not math.isfinite(radius) or radius < 0:
            raise InvalidDistanceError(
                "radius must be a finite non-negative number, got %r" % (radius,)
            )

        with self._lock:
            state = self._states.get(obj)
            if state is None:
                raise UnknownObjectError(repr(obj))

            # 无 LOD 的对象：返回原等级，不做任何切换。
            if state.levels == 0:
                return int(state.level)

            levels = state.levels
            # 计算无滞回的原始等级（严格小于阈值才升档）。
            if distance == 0.0:
                sse = math.inf
                raw = 0
            else:
                sse = float(radius) / float(distance)
                raw = 0
                bound = self.threshold
                top = levels - 1
                while raw < top and sse < bound:
                    raw += 1
                    bound *= 0.5  # 下一档边界：t, t/2, t/4, ...

            current = state.level
            if raw > current:
                # 距离变远：立即升档。
                current = raw
            elif raw < current:
                # 距离变近：必须比当前档的进入边界再精细 10% 才降档，
                # 进入当前档的 SSE 边界为 threshold / 2^(current-1)。
                entry_bound = self.threshold
                for _ in range(current - 1):
                    entry_bound *= 0.5
                if sse >= entry_bound * _HYSTERESIS:
                    current = raw
            state.level = current
            return int(current)

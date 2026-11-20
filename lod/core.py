"""lod 核心 API（初始未实现）。"""


class LodSelector:
    def __init__(self, levels, threshold):
        """构造"""
        raise NotImplementedError

    def select(self, distance, radius):
        """选择等级"""
        raise NotImplementedError

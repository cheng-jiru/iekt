"""
@Author: jiru.cheng
@Time: 2025/7/24 11:40
@File: ChainTracker.py
"""
class ChainTracker:
    def __init__(self, chains):
        self.chains = chains  # list of list
        self.pointers = [0 for _ in chains]  # 每个链一个指针

    def get_available_actions(self):
        """
        返回当前所有可以推荐的知识点（每个链的指针位置）
        """
        actions = []
        for i, p in enumerate(self.pointers):
            if p < len(self.chains[i]):
                actions.append(self.chains[i][p])
        return actions

    def update(self, recommended_concept):
        """
        推荐成功后，把所有链中当前指向 recommended_concept 的指针都往后移
        """
        for i, p in enumerate(self.pointers):
            if p < len(self.chains[i]) and self.chains[i][p] == recommended_concept:
                self.pointers[i] += 1

    def is_finished(self):
        return all(p >= len(c) for p, c in zip(self.pointers, self.chains))
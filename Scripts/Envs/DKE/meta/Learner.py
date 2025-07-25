"""
@Author: jiru.cheng
@Time: 2025/7/18 10:11
@File: Learner.py
"""
import json
import copy
import os
import numpy as np
from Scripts.Envs.meta import MetaLearner, MetaInfinityLearnerGroup
from utils.graph import build_learning_chains_best_cover
from Scripts.Envs.DKE.meta.ChainTracker import ChainTracker
__all__ = ["Learner", "LearnerGroup"]


class Learner(MetaLearner):
    def __init__(self,
                 initial_log,
                 learning_target: set,
                 concept_graph,  # <- 加入图结构
                 _id=None,
                 seed=None):
        super(Learner, self).__init__(user_id=_id)

        # info of learner：state/target/knowledge_structure/logs
        # 学习目标
        self._target = learning_target
        self._logs = initial_log
        self._state = []  # 掌握状态向量，在外部强化学习中维护更新
        self.random_state = np.random.RandomState(seed)

        # ===== 加入链结构推荐模块 =====
        self._chains = build_learning_chains_best_cover(concept_graph, learning_target)
        self._chain_tracker = ChainTracker(self._chains)
    def update_logs(self, logs):
        self._logs = logs

    @property
    def profile(self):
        return {
            "id": self.id,
            "logs": self._logs,
            "target": self.target
        }

    def learn(self, learning_item, score):
        """
        记录学习行为，并更新链指针
        """
        self._logs.append([learning_item, score])
        # 可选：更新 mastery 状态向量中的对应值（由外部控制）
        self._chain_tracker.update(learning_item)  # 更新链状态（指针往后移）

    @property
    def state(self):
        return self._state

    def response(self, test_item) -> ...:
        return self._state[test_item]

    @property
    def target(self):
        return self._target


class LearnerGroup(MetaInfinityLearnerGroup):
    def __init__(self, dataRec_path, seed=None):
        super(LearnerGroup, self).__init__()
        self.data_path = dataRec_path
        self.random_state = np.random.RandomState(seed)
        if not os.path.isdir(self.data_path) and not 'npz' in self.data_path:
            with open(self.data_path, 'r', encoding="utf-8") as f:
                self.datatxt = f.readlines()

    def __next__(self):
        session = [[0, 0]]
        learning_targets = set()
        while len({log[0] for log in session}) < 20:
            index = self.random_state.randint(len(self.datatxt))
            session = json.loads(self.datatxt[index])
            learning_targets = {step[0] for i, step in enumerate(session) if i >= 0.8 * len(session)}

        initial_log = copy.deepcopy(session[:int(len(session) * 0.6)])


        return Learner(
            initial_log=initial_log,
            learning_target=learning_targets,
        )

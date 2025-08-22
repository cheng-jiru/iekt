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
from utils.graph import build_learning_chains_best_cover, build_graph
from Scripts.Envs.DKE.meta.ChainTracker import ChainTracker

__all__ = ["Learner", "LearnerGroup"]
# 当前文件绝对路径
file_path = os.path.abspath(__file__)

# 当前文件所在目录
dir_path = os.path.dirname(file_path)


class Learner(MetaLearner):
    def __init__(self,
                 initial_log,
                 learning_target: set,
                 _id=None,
                 seed=None):
        super(Learner, self).__init__(user_id=_id)

        # 学习目标
        self._target = learning_target
        self._logs = initial_log
        self._state = []  # 掌握状态向量，在外部强化学习中维护更新
        self.random_state = np.random.RandomState(seed)
        self.init_len = len(initial_log)
        # ===== 加入链结构推荐模块 =====
        concept_graph = build_graph()
        self._chains = build_learning_chains_best_cover(concept_graph, learning_target)
        self._chain_tracker = ChainTracker(self._chains)

    def update_logs(self, logs):
        self._logs = logs

    @property
    def profile(self):
        return {
            "id": self.id,
            "logs": self._logs,
            "target": self.target,
            "chains": self._chains,
            "init_len": self.init_len
        }

    def learn(self, learning_item, score=None, related_knowledge=[]):
        self._logs.append([learning_item, score, related_knowledge])

    def update_concept(self, concept_id):
        self._chain_tracker.update(concept_id)  # 更新链状态（指针往后移）

    @property
    def state(self):
        return self._state

    def response(self, test_item) -> ...:
        """
        这里的test_item应该是知识点的id
        """
        return self._state[test_item]

    @property
    def target(self):
        return self._target


class LearnerGroup(MetaInfinityLearnerGroup):
    def __init__(self, dataRec_path, seed=None):
        super(LearnerGroup, self).__init__()
        self.data_path = os.path.join(dir_path, "../meta", dataRec_path)
        self.random_state = np.random.RandomState(seed)
        with open(self.data_path, 'r', encoding="utf-8") as f:
            self.datatxt = f.readlines()

    def __next__(self):
        session = [[0, 0, [0]]]
        learning_targets = set()
        while len({log[0] for log in session}) < 20:
            index = self.random_state.randint(len(self.datatxt))
            session = json.loads(self.datatxt[index])
            learning_targets = {kc
                                for i, step in enumerate(session)
                                if i >= int(0.7 * len(session))
                                for kc in step[2]
                                }

        initial_log = copy.deepcopy(session[:int(len(session) * 0.6)])

        return Learner(
            initial_log=initial_log,
            learning_target=learning_targets,
        )


if __name__ == '__main__':
    learner_group = LearnerGroup("converted_sequences_with_kc.jsonl", seed=42)
    for _ in range(1):
        learner = next(learner_group)
        print("Learning Target:", learner._target)
        # 模拟学习一个知识点
        if learner._chains:
            first_chain = learner._chains
            actions = learner._chain_tracker.get_available_actions()
            print("Available Actions:", actions)
            action = actions[0]
            learner._chain_tracker.update(action)
            actions = learner._chain_tracker.get_available_actions()
            print("Available Actions after learn:", actions)

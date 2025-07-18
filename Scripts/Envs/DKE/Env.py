"""
@Author: jiru.cheng
@Time: 2025/7/11 17:01
@File: Env.py
"""
from Scripts.Envs.DKE.meta.Scorer import KESScorer
from Scripts.Envs.meta import *
from Scripts.Envs.spaces import ListSpace
import numpy as np


class DKEEnv(Env):
    def __init__(self, seed=None):
        super(DKEEnv, self).__init__()
        self.item_list = [i for i in range(1, 11)]
        self.action_space = ListSpace(self.item_list, seed=seed)
        self.scorer = KESScorer()
        self._learner = None

    @property
    def parameters(self) -> dict:
        return {
            "action_space": self.action_space
        }

    def learn_and_test(self):
        # 学习以及更新学生状态
        pass

    def _exam(self, learner: Learner, detailed=False, reduce="sum") -> (dict, int, float):
        """
        理解成有没有掌握该知识点
        """
        state = learner.state
        knowledge_response = {}  # dict
        for test_item in learner.target:
            knowledge_response[test_item] = [test_item, self.scorer.response_function(state, test_item)]
        if detailed:
            return_thing = knowledge_response
        elif reduce == "sum":
            return_thing = np.sum([v for _, v in knowledge_response.values()])  # np.sum   []:list   knowledge_response
        elif reduce in {"mean", "ave"}:
            return_thing = np.average([v for _, v in knowledge_response.values()])
        else:
            raise TypeError("unknown reduce type %s" % reduce)  # unknown reduce type
        return return_thing

    def update_learner_state(self):
        pass

    def begin_episode(self, *args, **kwargs):
        pass

    def end_episode(self, *args, **kwargs):
        pass

    def reset(self):
        self._learner = None

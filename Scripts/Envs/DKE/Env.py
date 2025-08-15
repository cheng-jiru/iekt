"""
@Author: jiru.cheng
@Time: 2025/7/11 17:01
@File: Env.py
"""
from Scripts.Envs.DKE.meta.Learner import LearnerGroup,Learner
from Scripts.Envs.DKE.meta.Scorer import KESScorer
from Scripts.Envs.meta import *
from Scripts.Envs.spaces import ListSpace
import numpy as np
import json
from utils.agent_utils import get_graph_embeddings, get_kt_model
import time

class DKEEnv(Env):
    def __init__(self, seed=None):
        super(DKEEnv, self).__init__()

        self.scorer = KESScorer()
        self._learner = None
        self.random_state = np.random.RandomState(seed)
        self.concept_graph_embeddings, self.exercise_graph_embeddings = get_graph_embeddings('dbe')
        self.num_concept = 98
        self.num_exercise = 212
        self.max_concept = 4
        with open("new_knowledge_component.json", "r") as f:
            kc_map = json.load(f)  # 题目ID(str) -> [知识点ID(int)]
        # 题目编号从 1 到 212（int 类型）
        self.item_list = list(range(1, 213))
        self.concept_list = list(range(1, 99))  # 知识点编号从 1 到 98（int 类型）
        self.exercise_item_base = [
            Item(item_id=i, knowledge=kc_map[str(i)]) for i in self.item_list if str(i) in kc_map
        ]
        self.exercise_action_space = ListSpace(self.item_list, seed=seed)

        self.action_space = ListSpace(self.concept_list, seed=seed)
        # KTnet
        self.Ktagent = get_kt_model()

        # learners
        dataRec_path="converted_sequences_with_kc.jsonl",
        self.learners = LearnerGroup(dataRec_path, seed=seed)
        self._learner = None
        self._initial_score = None
        self.episode_start_time = time.time()
        self.episode_end_time = time.time()

    @property
    def parameters(self) -> dict:
        return {
            "action_space": self.action_space
        }

    def learn_and_test(self):
        # 学习以及更新学生状态
        logs = self._learner.profile['logs']
        self._learner._state=[]

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

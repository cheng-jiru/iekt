"""
@Author: jiru.cheng
@Time: 2025/7/11 17:01
@File: Env.py
"""
from Scripts.Envs.DKE.meta.Learner import LearnerGroup, Learner
from Scripts.Envs.DKE.meta.Scorer import KESScorer
from Scripts.Envs.meta import *
from Scripts.Envs.spaces import ListSpace
import numpy as np
import json

from utils.Reward import episode_reward
from utils.agent_utils import get_graph_embeddings, get_kt_model
import time
import torch


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
        self.kc_map=kc_map
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
        dataRec_path = "converted_sequences_with_kc.jsonl",
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

    def learn_and_test(self, learner: Learner, practice_id):
        # 这一步时学习知识点
        logs = self._learner.profile['logs']
        concept_related=self.kc_map[str(practice_id)] #相关知识点有哪些
        practice_score=self.update_learner_state


    def _exam(self, learner: Learner, detailed=False, reduce="sum") -> (dict, int, float):
        """
        理解成有没有掌握该知识点
        """
        state = learner.state
        knowledge_response = {}  # dict
        for test_item in learner.target:
            knowledge_response[test_item] = [test_item, self.scorer.response_concept_function(state, test_item)]
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
        logs = self._learner.profile['logs']
        init_len = self._learner.profile['init_len']
        input_data = self.generate_sequence_demo(logs, max_skills=self.max_concept)
        _, update_state, practice_sore = self.Ktagent.test_single_student(input_data, h=self._learner._state, init_len=init_len)
        self._learner._state = update_state.squeeze(0).tolist()
        return practice_sore

    def init_learner_state(self):
        logs = self._learner.profile['logs']
        input_data = self.generate_sequence_demo(logs, max_skills=self.max_concept)
        _, init_state, _ = self.Ktagent.test_single_student(input_data)
        self._learner._state = init_state.squeeze(0).tolist()  # tensor转化为 list
        return self._learner._state

    def begin_episode(self, *args, **kwargs):
        self._learner = next(self.learners)
        init_state=self.init_learner_state()  # 初始化一下状态
        self._initial_score = self._exam(self._learner)  # learner initial_score
        while self._initial_score >= len(self._learner.target):
            self._learner = next(self.learners)  # learner（learning target、state、knowledge_structure）
            self.update_learner_state()
            self._initial_score = self._exam(self._learner)  # learner initial_score
        return self._learner.profile, self._exam(self._learner, detailed=True),init_state

    def end_episode(self, *args, **kwargs):
        observation = self._exam(self._learner, detailed=True)
        initial_score, self._initial_score = self._initial_score, None
        final_score = self._exam(self._learner)
        reward = episode_reward(initial_score, final_score, len(self._learner.target))
        info = {"initial_score": initial_score, "final_score": final_score}
        done = final_score == len(self._learner.target)
        return observation, reward, done, info

    def reset(self):
        self._learner = None

    def step_p(self,practice_item_id, *args, **kwargs):
        #学习单个题目的时候需要的东西
        pass

    def generate_sequence_demo(self,
                               logs: list,
                               max_skills: int = 4,
                               ):
        """
        生成一整个学生答题序列的输入特征 X。

        参数：
        - records: [[problem_id, answer, [skills]], ...] 学生答题序列
        - max_skills: 每题最多关联的知识点数量

        返回：
        - X: 序列化输入数据 (T, 8个字段)
        """

        X = []

        for problem_id, answer, skills in logs:
            # 填充或截断知识点ID到 max_skills 长度
            skills = skills[:max_skills] + [0] * (max_skills - len(skills))

            # 掩码：非零为有效知识点
            mask = [1 if s != 0 else 0 for s in skills]

            # 只保留需要的字段，其余字段填 0（忽略）
            x0 = torch.tensor(0., dtype=torch.float32)  # 知识点最后出现时间编码（忽略）
            x1 = torch.tensor(0., dtype=torch.float32)  # 知识点one-hot编码（忽略）
            x2 = torch.tensor([skills])  # 知识点 ID，(1, 4)
            x3 = torch.tensor(0., dtype=torch.float32)  # 知识点出现次数编码（忽略）
            x4 = torch.tensor([[answer]], dtype=torch.float32)  # 答题结果，(1, 1)
            x5 = torch.tensor([mask], dtype=torch.float32)  # 有效知识点掩码，(1, 4)
            x6 = torch.tensor([problem_id])  # 题目ID，(1,)
            x7 = torch.tensor(0., dtype=torch.float32)  # 知识点关联矩阵（忽略）

            X.append([x0, x1, x2, x3, x4, x5, x6, x7])

        return X

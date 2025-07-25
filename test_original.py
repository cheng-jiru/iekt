import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import time
import os
from sklearn import metrics
from sklearn.metrics import accuracy_score
import logging as log
import numpy
import tqdm
import pickle
from utils_tools import batch_data_to_device
from torch.optim.lr_scheduler import ReduceLROnPlateau
import models

"""
测试单个学生的答题预测 一个时间步包含8个特征，全是tensor，
x[0] 32*100：知识点最后出现的时间编码，无用可作为空,可忽略，
x[1] 32*100：知识点one_hot编码，可为空，32*99，99为题目数量,可忽略
x[2] 32*4 题目关联知识点id，最多的是4个，
x[3] 32*100 知识点出现次数编码，可忽略
x[4] 32*1 题目答题结果，0/1
x[5] 32*4 知识点是否过滤的掩码
x[6] 32   题目id
x[7] 32*4*99 知识点关联矩阵 可忽略
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import models


class KtAgent():
    def __init__(self, model_path, device, dim):
        super(KtAgent, self).__init__()
        self.model = torch.load(model_path, map_location=torch.device(device))
        self.dim=dim
        self.device = device


    def test_single_student(self, single_seq_data):
        """
        single_seq_data: 单个学生的单条或序列数据，格式要跟训练时输入格式对应，比如：
                         [(question_id, other_feats..., ground_truth), ...]
        返回：
            - 每题的答题预测概率列表
            - 每题对应的学生状态列表（hidden states）
            - 每题预测的答题结果（0/1）
        """

        device = self.device
        self.model.eval()
        eval_sigmoid = torch.nn.Sigmoid()

        # 初始化状态
        h = torch.zeros(1, self.dim).to(device)

        prob_list = []
        state_list = []
        pred_label_list = []

        seq_len = len(single_seq_data)
        for seqi in range(seq_len):
            # 取出当前时间步的8个特征并迁移到目标 device
            timestep = [x.to(device) for x in single_seq_data[seqi]]
            ques_id = timestep[6]
            related_concept_index = timestep[2]  # 题目关联的知识点索引
            filtered_concept_index = timestep[5]
            v = self.model.get_ques_representation_ave(
                ques_id,
                related_concept_index,  # 举例
                filtered_concept_index,  # 举例
                1
            )
            ques_h = torch.cat([v, h], dim=1)

            flip_prob_emb = self.model.pi_cog_func(ques_h)
            m = Categorical(flip_prob_emb)
            emb_ap = m.sample()
            emb_p = self.model.cog_matrix[emb_ap, :]
            rt_x = torch.zeros(1, 1, self.dim* 2).to(device)
            h_v, v, logits, rt_x = self.model.obtain_v(timestep, h, rt_x, emb_p)
            prob = eval_sigmoid(logits)

            out_operate_logits = (prob > 0.5).float()
            out_x_logits = torch.cat([
                h_v.mul(out_operate_logits.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1 - out_operate_logits).repeat(1, h_v.size()[-1]).float())
            ], dim=1)

            out_x = torch.cat([out_x_logits, out_x_logits], dim=1)  # 强化学习不能用真实的结果，必须以预测的当成环境的结果

            flip_prob_emb = self.model.pi_sens_func(out_x)
            m = Categorical(flip_prob_emb)
            emb_a = m.sample()
            emb = self.model.acq_matrix[emb_a, :]

            h = self.model.update_state(h, v, emb, out_operate_logits)

            # 记录预测结果、概率和状态
            prob_list.append(prob.item())
            state_list.append(h.detach().cpu().clone())
            pred_label_list.append(int(out_operate_logits.item()))

        return prob_list, state_list, pred_label_list


def generate_single_step_demo(
        problem_id: int,
        skills: list,
        answer: int = 1,
        max_skills: int = 4,
        problem_number: int = 99,
        concept_num: int = 100,
):
    """
    生成单个学生在某一时间步的输入特征 x。

    参数：
    - problem_id: 当前题目的 ID（例如 15）
    - skills: 与该题关联的知识点 ID 列表（最多4个），如 [3, 10]
    - answer: 答题结果（0/1），仿真时用预测值
    - max_skills: 每题最多关联的知识点数量，默认为 4
    - problem_number: 总题目数量，用于构造 x[7] 时确定维度
    - concept_num: 总知识点数量，用于构造可忽略字段时使用

    返回：
    - x: 包含8个字段的列表，可直接用于模型
    """

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

    return [[x0, x1, x2, x3, x4, x5, x6, x7]]


if __name__ == '__main__':
    model_path = 'run/h_number_graph/best_model.pt'
    text_data = generate_single_step_demo(12, [1, 4], )
    agent = KtAgent(model_path, 4, dim=99)
    prob_list, state_list, pred_label_list = agent.test_single_student(text_data)
    print("预测概率列表:", prob_list)
    print("预测状态列表:", state_list)
    print("预测标签列表:", pred_label_list)

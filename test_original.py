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
from utils import batch_data_to_device
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


def test_single_student(model, single_seq_data, args, device=None):
    """
    single_seq_data: 单个学生的单条或序列数据，格式要跟训练时输入格式对应，比如：
                     [(question_id, other_feats..., ground_truth), ...]
    返回：
        - 每题的答题预测概率列表
        - 每题对应的学生状态列表（hidden states）
        - 每题预测的答题结果（0/1）
    """
    if device is None:
        device = args.device
    model.eval()
    eval_sigmoid = torch.nn.Sigmoid()

    # 初始化状态
    h = torch.zeros(1, args.dim).to(device)

    prob_list = []
    state_list = []
    pred_label_list = []

    seq_len = len(single_seq_data)
    for seqi in range(seq_len):
        # 取出当前时间步的8个特征并迁移到目标 device
        timestep = [x.to(device) for x in single_seq_data[seqi]]
        ques_id = timestep[6]
        related_concept_index=timestep[2] # 题目关联的知识点索引
        filtered_concept_index=timestep[5]
        v = model.get_ques_representation_ave(
            ques_id,
            related_concept_index,  # 举例
            filtered_concept_index,  # 举例
            1
        )
        ques_h = torch.cat([v, h], dim=1)

        flip_prob_emb = model.pi_cog_func(ques_h)
        m = Categorical(flip_prob_emb)
        emb_ap = m.sample()
        emb_p = model.cog_matrix[emb_ap, :]
        rt_x = torch.zeros(1, 1, args.dim * 2).to(args.device)
        h_v, v, logits, rt_x = model.obtain_v(timestep, h, rt_x, emb_p)
        prob = eval_sigmoid(logits)

        out_operate_logits = (prob > 0.5).float()
        out_x_logits = torch.cat([
            h_v.mul(out_operate_logits.repeat(1, h_v.size()[-1]).float()),
            h_v.mul((1 - out_operate_logits).repeat(1, h_v.size()[-1]).float())
        ], dim=1)

        out_x = torch.cat([out_x_logits, out_x_logits], dim=1)  # 强化学习不能用真实的结果，必须以预测的当成环境的结果

        flip_prob_emb = model.pi_sens_func(out_x)
        m = Categorical(flip_prob_emb)
        emb_a = m.sample()
        emb = model.acq_matrix[emb_a, :]

        h = model.update_state(h, v, emb, out_operate_logits)

        # 记录预测结果、概率和状态
        prob_list.append(prob.item())
        state_list.append(h.detach().cpu().clone())
        pred_label_list.append(int(out_operate_logits.item()))

    return prob_list, state_list, pred_label_list

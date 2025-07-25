#!/usr/bin/env python
# encoding: utf-8

import torch
from torch.utils.data import DataLoader

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import pickle
import argparse
import logging as log

import models
import importlib
import train_original as train
import test_original as test
from dataset import Dataset
import numpy as np
import random
import math
from datetime import datetime

parser = argparse.ArgumentParser(description='IEKT')
parser.add_argument('--debug',          action='store_true',        help='log debug messages or not')
parser.add_argument('--run_exist',      action='store_true',        help='run dir exists ok or not')
parser.add_argument('--run_dir',        type=str,   default='run/h_number_graph/', help='dir to save log and models')
parser.add_argument('--data_dir',       type=str,   default='data/new_mini_09/') #assistment2009-2010
parser.add_argument('--checkpoint_path',type=str,  default= 'none',   help='the path of checkpoint')
parser.add_argument('--log_every',      type=int,   default=0,      help='number of steps to log loss, do not log if 0')
parser.add_argument('--eval_every',     type=int,   default=0,      help='number of steps to evaluate, only evaluate after each epoch if 0')
parser.add_argument('--save_every',     type=int,   default=20,      help='number of steps to save model')
parser.add_argument('--device',         type=int,   default=-1,      help='gpu device id, cpu if -1')
parser.add_argument('--model',          type=str,   default='iekt',   help='run model')
parser.add_argument('--n_layer',type=int,   default=2,      help='number of mlp hidden layers in decoder')
parser.add_argument('--dim',type=int,   default=99,     help='hidden size for nodes')
parser.add_argument('--n_epochs',       type=int,   default=300,   help='number of epochs to train')
parser.add_argument('--batch_size',     type=int,   default=32,      help='number of instances in a batch')
parser.add_argument('--lr',             type=float, default=1e-3,   help='learning rate')
parser.add_argument('--dropout',        type=float, default=0,   help='dropout')
parser.add_argument('--seq_len',       type=int, default=200,   help='the length of the sequence') 
parser.add_argument('--gamma',        type=float, default=0.93,   help='graph_type') 
parser.add_argument('--cog_levels',        type=int, default=10,   help='the response action space for cognition estimation')
parser.add_argument('--acq_levels',        type=int, default=10,   help='the response action space for  sensitivity estimation')
parser.add_argument('--lamb',        type=float, default=40,   help='hyper parameter for loss')
parser.add_argument('--decay',type=float, default=1e-6,   help='hyper parameter for decay')
args = parser.parse_args() 

if args.debug:
    args.run_exist = True
    args.run_dir = 'debug'
os.makedirs(args.run_dir, exist_ok=args.run_exist)


log.basicConfig(
    format='%(asctime)s: %(message)s',
    datefmt='%m/%d %I:%M:%S %p', level=log.DEBUG if args.debug else log.INFO)
current_time = datetime.now().strftime("%H:%M:%S")
log.getLogger().addHandler(log.FileHandler(os.path.join(args.run_dir, f'{current_time}_log.txt'), mode='w'))
log.info('args: %s' % str(args))
args.device = 'cpu' if args.device < 0 else 'cuda:%i' % args.device
args.device = torch.device(args.device)

def preprocess():
    datasets = {}
    with open(args.data_dir + 'problem_skill_maxSkillOfProblem_number.pkl', 'rb') as fp:
            problem_number, concept_number, max_concept_of_problem = pickle.load(fp)

    
    setattr(args, 'max_concepts', max_concept_of_problem)
    setattr(args, 'concept_num', concept_number)
    setattr(args, 'problem_number', problem_number)
    setattr(args, 'prob_dim', int(math.log(problem_number,2)) + 1)
    
    for split in ['train', 'valid', 'test']:
        file_name = os.path.join(args.data_dir, 'dataset_%s.pkl' % split)
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                datasets[split] = pickle.load(f)
            log.info('Dataset split %s loaded' % split)
        else:
            datasets[split] = Dataset(args.problem_number, args.concept_num, root_dir=args.data_dir, split=split)
            with open(file_name, 'wb') as f:
                pickle.dump(datasets[split], f)
            log.info('Dataset split %s created and dumpped' % split)

    loaders = {}
    for split in ['train', 'valid', 'test']:
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            collate_fn=datasets[split].collate,
            shuffle=True if split == 'train' else False
        )

    return loaders

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
    x0 = torch.tensor(0., dtype=torch.float32)   # 知识点最后出现时间编码（忽略）
    x1 = torch.tensor(0., dtype=torch.float32)  # 知识点one-hot编码（忽略）
    x2 = torch.tensor([skills])              # 知识点 ID，(1, 4)
    x3 = torch.tensor(0., dtype=torch.float32)   # 知识点出现次数编码（忽略）
    x4 = torch.tensor([[answer]], dtype=torch.float32)           # 答题结果，(1, 1)
    x5 = torch.tensor([mask], dtype=torch.float32)                # 有效知识点掩码，(1, 4)
    x6 = torch.tensor([problem_id])          # 题目ID，(1,)
    x7 = torch.tensor(0., dtype=torch.float32)   # 知识点关联矩阵（忽略）

    return [[x0, x1, x2, x3, x4, x5, x6, x7]]


if __name__ == '__main__':
    loaders = preprocess()
    Model = getattr(models, args.model)
    # 加载静态文本嵌入（来自 BERT）
    exercise_bert_emb = np.load("exercise_embedding_graph.npy")  # shape: [problem_num, 1024]
    concept_bert_emb = np.load("concept_embedding_graph.npy")  # shape: [concept_num, 1024]
    exercise_bert_emb[0] = 0  # 第一个问题的嵌入向量置为0
    concept_bert_emb[0] = 0  # 第一个知识点的嵌入向量置为0
    # 转换为 tensor
    exercise_bert_emb = torch.tensor(exercise_bert_emb, dtype=torch.float32)
    concept_bert_emb = torch.tensor(concept_bert_emb, dtype=torch.float32)
    if args.checkpoint_path != 'none':
        model = torch.load(args.checkpoint_path, map_location = torch.device(args.device))
    else:
        model = Model(args,exercise_bert_emb,concept_bert_emb).to(args.device)
        # model = Model(args).to(args.device)

    # log.info(str(vars(args)))
    # text_data=generate_single_step_demo(12,[1,4],)
    # # 把text_data放到gpu上
    #
    train.train(model, loaders, args)
    # print(test.test_single_student(model, text_data, args))

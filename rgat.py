"""
@Author: jiru.cheng
@Time: 2025/7/22 10:30
@File: rgat.py
"""
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import heterograph
import numpy as np
import json
import pandas as pd
from dgl.nn.pytorch import HeteroGraphConv, GATConv
import random


# =======================
# 1. 数据加载
# =======================

def load_embeddings():
    exercise_emb = np.load('exercise_embedding_matrix.npy')  # [N_ex, 1024]
    concept_emb = np.load('concept_embedding_matrix.npy')  # [N_concept, 1024]
    return torch.tensor(exercise_emb, dtype=torch.float32), torch.tensor(concept_emb, dtype=torch.float32)


def load_relations():
    with open('new_knowledge_component.json') as f:
        ex2con = json.load(f)

    df = pd.read_csv('KC_Relationships_mapped.csv')
    con2con = list(zip(df['from_knowledgecomponent_id'], df['to_knowledgecomponent_id']))

    return ex2con, con2con


# ========== 构建异构图 ==========
def build_graph(ex2con, con2con, num_ex, num_con):
    data_dict = {
        ('exercise', 'ex2con', 'concept'): [],
        ('concept', 'con2ex', 'exercise'): [],
        ('concept', 'con2con', 'concept'): []
    }

    for ex_str, con_list in ex2con.items():
        ex = int(ex_str)
        for con in con_list:
            data_dict[('exercise', 'ex2con', 'concept')].append((ex, con))
            data_dict[('concept', 'con2ex', 'exercise')].append((con, ex))

    for src, dst in con2con:
        data_dict[('concept', 'con2con', 'concept')].append((src, dst))

    graph = heterograph({
        rel: (torch.tensor([s for s, _ in edges]), torch.tensor([d for _, d in edges]))
        for rel, edges in data_dict.items()
    }, num_nodes_dict={
        'exercise': num_ex,
        'concept': num_con
    })

    return graph


# ========== 模型 ==========
class RGATEmbedder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, rel_names):
        super().__init__()
        self.layer1 = HeteroGraphConv({
            rel: GATConv(in_dim, hid_dim, num_heads=4)
            for rel in rel_names
        }, aggregate='mean')
        self.layer2 = HeteroGraphConv({
            rel: GATConv(hid_dim * 4, out_dim, num_heads=1)
            for rel in rel_names
        }, aggregate='mean')

    def forward(self, graph, inputs):
        h = self.layer1(graph, inputs)
        h = {k: F.elu(v.flatten(1)) for k, v in h.items()}
        h = self.layer2(graph, h)
        h = {k: v.mean(1) if v.ndim == 3 else v for k, v in h.items()}
        return h


# ----------------------
# 计算题目对Jaccard相似度权重
# ----------------------
def compute_jaccard_similarity(ex2con, ex1, ex2):
    set1 = set(ex2con[str(ex1)])
    set2 = set(ex2con[str(ex2)])
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union == 0:
        return 0.0
    return intersection / union


# ---------------------------
# 正样本对构建与采样
# ---------------------------
def build_positive_pairs(ex2con, con2con,
                         max_expos=1000, max_excon=200, max_concon=200):
    # 反向映射: 知识点 -> 题目列表
    c2e = {}
    for ex_str, cons in ex2con.items():
        ex = int(ex_str)
        for c in cons:
            c2e.setdefault(c, []).append(ex)

    # 题目-题目正样本对 (同一知识点下两两配对)
    ex_pos_pairs = []
    ex_pos_weights = []
    for con, ex_list in c2e.items():
        if len(ex_list) < 2:
            continue
        for i in range(len(ex_list)):
            for j in range(i + 1, len(ex_list)):
                ex1, ex2 = ex_list[i], ex_list[j]
                weight = compute_jaccard_similarity(ex2con, ex1, ex2)
                ex_pos_pairs.append((ex1, ex2))
                ex_pos_weights.append(weight)
    # if len(ex_pos_pairs) > max_expos:
    #     idxs = random.sample(range(len(ex_pos_pairs)), max_expos)
    #     ex_pos_pairs = [ex_pos_pairs[i] for i in idxs]
    #     ex_pos_weights = [ex_pos_weights[i] for i in idxs]

    # 题目-知识点正样本对 (题目与其包含知识点)
    ex_con_pairs = []
    for ex_str, cons in ex2con.items():
        ex = int(ex_str)
        for c in cons:
            ex_con_pairs.append((ex, c))
    # if len(ex_con_pairs) > max_excon:
    #     ex_con_pairs = random.sample(ex_con_pairs, max_excon)

    # 知识点-知识点正样本对 (前后继关系)
    con_pos_pairs = []
    for src, dst in con2con:
        con_pos_pairs.append((src, dst))
    # if len(con_pos_pairs) > max_concon:
    #     con_pos_pairs = random.sample(con_pos_pairs, max_concon)

    print(f"Positive pairs: {len(ex_pos_pairs)} ex-ex, {len(ex_con_pairs)} ex-con, {len(con_pos_pairs)} con-con")

    return ex_pos_pairs, ex_pos_weights, ex_con_pairs, con_pos_pairs


# ---------------------------
# 负样本对构建与采样
# ---------------------------

def build_negative_pairs(ex2con, con2con, num_ex, num_con,
                         max_ex_neg=1000, max_excon_neg=200, max_concon_neg=200):
    # 题目知识点映射转为集合方便查找
    ex2con_sets = {int(k): set(v) for k, v in ex2con.items()}

    # 题目-题目负样本对：知识点集合无交集
    ex_neg_pairs = []
    for i in range(num_ex):
        for j in range(i + 1, num_ex):
            if ex2con_sets.get(i, set()).isdisjoint(ex2con_sets.get(j, set())):  # 判断两个题目对应的知识点集合是否无交集
                ex_neg_pairs.append((i, j))
    # if len(ex_neg_pairs) > max_ex_neg:
    #     ex_neg_pairs = random.sample(ex_neg_pairs, max_ex_neg)

    # 题目-知识点负样本对：知识点不在题目对应集合中
    ex_con_neg_pairs = []
    for ex in range(num_ex):
        neg_cons = set(range(num_con)) - ex2con_sets.get(ex, set())
        for con in neg_cons:
            ex_con_neg_pairs.append((ex, con))
    # if len(ex_con_neg_pairs) > max_excon_neg:
    #     ex_con_neg_pairs = random.sample(ex_con_neg_pairs, max_excon_neg)

    # 知识点-知识点负样本对：图中无边连接（排除已有边）
    # 建立边集合
    con_edge_set = set(con2con)
    con_neg_pairs = []
    for i in range(num_con):
        for j in range(i + 1, num_con):
            if (i, j) not in con_edge_set and (j, i) not in con_edge_set:
                con_neg_pairs.append((i, j))
    # if len(con_neg_pairs) > max_concon_neg:
    #     con_neg_pairs = random.sample(con_neg_pairs, max_concon_neg)

    print(f"Negative pairs: {len(ex_neg_pairs)} ex-ex, {len(ex_con_neg_pairs)} ex-con, {len(con_neg_pairs)} con-con")

    return ex_neg_pairs, ex_con_neg_pairs, con_neg_pairs


# ----------------------
# 对比学习损失，带权重的题目-题目部分
# ----------------------
def contrastive_loss(ex_embed, con_embed,
                     ex_pos_pairs, ex_pos_weights,
                     ex_con_pairs, con_pos_pairs,
                     temperature=0.5,
                     weight_ex_con=0.3,
                     weight_con_con=0.3,
                     device='cpu'):
    loss = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)

    # 题目-题目加权InfoNCE
    if len(ex_pos_pairs) > 0:
        ex_i, ex_j = zip(*ex_pos_pairs)
        ex_i = torch.tensor(ex_i, device=device)
        ex_j = torch.tensor(ex_j, device=device)
        weights = torch.tensor(ex_pos_weights, device=device)

        emb_i = ex_embed[ex_i]  # [M, d]
        emb_j = ex_embed[ex_j]  # [M, d]

        pos_sim = F.cosine_similarity(emb_i, emb_j) / temperature
        sim_i_all = torch.mm(emb_i, ex_embed.t()) / temperature

        numerator = torch.exp(pos_sim)
        denominator = torch.exp(sim_i_all).sum(dim=1)

        loss_ex_ex = -torch.log(numerator / denominator)
        weighted_loss = (loss_ex_ex * weights).sum() / (weights.sum() + 1e-8)

        loss += weighted_loss
        count += 1

    # 题目-知识点对比
    if len(ex_con_pairs) > 0:
        ex_i, con_j = zip(*ex_con_pairs)
        ex_i = torch.tensor(ex_i, device=device)
        con_j = torch.tensor(con_j, device=device)
        emb_ex = ex_embed[ex_i]
        emb_con = con_embed[con_j]

        pos_sim = F.cosine_similarity(emb_ex, emb_con) / temperature
        sim_ex_all_con = torch.mm(emb_ex, con_embed.t()) / temperature

        numerator = torch.exp(pos_sim)
        denominator = torch.exp(sim_ex_all_con).sum(dim=1)

        loss_ex_con = -torch.log(numerator / denominator).mean()
        loss += weight_ex_con * loss_ex_con
        count += weight_ex_con

    # 知识点-知识点对比
    if len(con_pos_pairs) > 0:
        con_i, con_j = zip(*con_pos_pairs)
        con_i = torch.tensor(con_i, device=device)
        con_j = torch.tensor(con_j, device=device)
        emb_i = con_embed[con_i]
        emb_j = con_embed[con_j]

        pos_sim = F.cosine_similarity(emb_i, emb_j) / temperature
        sim_i_all = torch.mm(emb_i, con_embed.t()) / temperature

        numerator = torch.exp(pos_sim)
        denominator = torch.exp(sim_i_all).sum(dim=1)

        loss_con_con = -torch.log(numerator / denominator).mean()
        loss += weight_con_con * loss_con_con
        count += weight_con_con

    if count > 0:
        loss /= count

    return loss


def contrastive_loss_triplet_weighted(
        ex_embed, con_embed,
        ex_pos_pairs, ex_pos_weights,
        ex_neg_pairs,
        ex_con_pos_pairs,
        ex_con_neg_pairs,
        con_pos_pairs,
        con_neg_pairs,
        margin=0.1,
        weight_ex_con=0.3,
        weight_con_con=0.3,
        num_neg_samples=3,
        device='cpu'
):
    loss = torch.tensor(0., device=device)

    # 题目-题目部分带权重Triplet Loss
    if len(ex_pos_pairs) > 0 and len(ex_neg_pairs) > 0:
        ex_pos_i, ex_pos_j = zip(*ex_pos_pairs)
        weights = torch.tensor(ex_pos_weights, device=device)
        ex_pos_i = torch.tensor(ex_pos_i, device=device)
        ex_pos_j = torch.tensor(ex_pos_j, device=device)

        ex_neg_dict = {}
        for a, b in ex_neg_pairs:
            ex_neg_dict.setdefault(a, []).append(b)
            ex_neg_dict.setdefault(b, []).append(a)

        triplet_losses = []
        triplet_weights = []
        for idx, (a, p) in enumerate(zip(ex_pos_i.tolist(), ex_pos_j.tolist())):
            if a not in ex_neg_dict:
                continue
            neg_candidates = ex_neg_dict[a]
            neg_samples = random.sample(neg_candidates, min(num_neg_samples, len(neg_candidates)))
            anchor = ex_embed[a]
            positive = ex_embed[p]
            neg_triplet_losses = []
            for neg in neg_samples:
                negative = ex_embed[neg]

                pos_dist = cosine_distance(anchor.unsqueeze(0), positive.unsqueeze(0))
                neg_dist = cosine_distance(anchor.unsqueeze(0), negative.unsqueeze(0))

                triplet_loss = F.relu(pos_dist - neg_dist + margin)
                neg_triplet_losses.append(triplet_loss)
            if neg_triplet_losses:
                mean_triplet_loss = torch.stack(neg_triplet_losses).mean()  # 先对负样本均值
                triplet_losses.append(mean_triplet_loss)
                triplet_weights.append(weights[idx])


        if triplet_losses:
            triplet_losses = torch.stack(triplet_losses)
            triplet_weights = torch.stack(triplet_weights)
            weighted_loss = (triplet_losses * triplet_weights).sum() / (triplet_weights.sum() + 1e-8)
            loss += weighted_loss

    # 题目-知识点部分
    if len(ex_con_pos_pairs) > 0 and len(ex_con_neg_pairs) > 0:
        ex_pos_i, con_pos_j = zip(*ex_con_pos_pairs)
        ex_pos_i = torch.tensor(ex_pos_i, device=device)
        con_pos_j = torch.tensor(con_pos_j, device=device)

        ex_con_neg_dict = {}
        for ex, con in ex_con_neg_pairs:
            ex_con_neg_dict.setdefault(ex, []).append(con)

        triplet_losses = []
        for ex, pos_con in zip(ex_pos_i.tolist(), con_pos_j.tolist()):
            if ex not in ex_con_neg_dict:
                continue
            neg_cons = ex_con_neg_dict[ex]
            neg_samples = random.sample(neg_cons, min(num_neg_samples, len(neg_cons)))
            anchor = ex_embed[ex]
            positive = con_embed[pos_con]

            for neg_con in neg_samples:
                negative = con_embed[neg_con]
                pos_dist = cosine_distance(anchor.unsqueeze(0), positive.unsqueeze(0))
                neg_dist = cosine_distance(anchor.unsqueeze(0), negative.unsqueeze(0))
                triplet_loss = F.relu(pos_dist - neg_dist + margin)
                triplet_losses.append(triplet_loss)
        if triplet_losses:
            loss += weight_ex_con * torch.stack(triplet_losses).mean()
    # print(f"Triplet Loss for exercise-concept: {weight_ex_con * torch.stack(triplet_losses).mean().item():.4f}")

    # 知识点-知识点部分
    if len(con_pos_pairs) > 0 and len(con_neg_pairs) > 0:
        con_pos_i, con_pos_j = zip(*con_pos_pairs)
        con_pos_i = torch.tensor(con_pos_i, device=device)
        con_pos_j = torch.tensor(con_pos_j, device=device)

        con_neg_dict = {}
        for a, b in con_neg_pairs:
            con_neg_dict.setdefault(a, []).append(b)
            con_neg_dict.setdefault(b, []).append(a)

        triplet_losses = []
        for a, p in zip(con_pos_i.tolist(), con_pos_j.tolist()):
            if a not in con_neg_dict:
                continue
            neg_candidates = con_neg_dict[a]
            neg_samples = random.sample(neg_candidates, min(num_neg_samples, len(neg_candidates)))
            anchor = con_embed[a]
            positive = con_embed[p]

            for neg in neg_samples:
                negative = con_embed[neg]
                pos_dist = cosine_distance(anchor.unsqueeze(0), positive.unsqueeze(0))
                neg_dist = cosine_distance(anchor.unsqueeze(0), negative.unsqueeze(0))
                triplet_loss = F.relu(pos_dist - neg_dist + margin)
                triplet_losses.append(triplet_loss)
        if triplet_losses:
            loss += weight_con_con * torch.stack(triplet_losses).mean()
        # print(f"Triplet Loss for concept-concept: {weight_con_con * torch.stack(triplet_losses).mean().item():.4f}")

    return loss

def cosine_distance(x1, x2):
    # 输入已经归一化
    return 1 - (x1 * x2).sum(dim=-1)

# ========== 训练循环 ==========
def train():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    # 加载数据
    ex_emb, con_emb = load_embeddings()
    ex2con, con2con = load_relations()
    num_ex, num_con = ex_emb.size(0), con_emb.size(0)

    g = build_graph(ex2con, con2con, num_ex, num_con)
    g = g.to(device)  # 👈 图放到 GPU
    model = RGATEmbedder(in_dim=1024, hid_dim=256, out_dim=99, rel_names=g.etypes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    feat = {
        'exercise': ex_emb.to(device),
        'concept': con_emb.to(device)
    }
    # 构建正样本对（只构建一次）
    ex_pos_pairs, ex_pos_weights, ex_con_pos_pairs, con_pos_pairs = build_positive_pairs(ex2con, con2con)
    # 构建负样本对
    ex_neg_pairs, ex_con_neg_pairs, con_neg_pairs = build_negative_pairs(ex2con, con2con, num_ex, num_con)
    best_loss = float('inf')
    save_path = 'best_rgat_model.pt'
    print(f"Training on {device_str}...")
    for epoch in range(100):
        model.train()
        new_emb = model(g, feat)
        new_emb['exercise'] = F.normalize(new_emb['exercise'], dim=1)
        new_emb['concept'] = F.normalize(new_emb['concept'], dim=1)
        ex_embed = new_emb['exercise']
        con_embed = new_emb['concept']

        loss = contrastive_loss_triplet_weighted(
            ex_embed, con_embed,
            ex_pos_pairs, ex_pos_weights,
            ex_neg_pairs,
            ex_con_pos_pairs,
            ex_con_neg_pairs,
            con_pos_pairs,
            con_neg_pairs,
            margin=0.5,
            weight_ex_con=0.5,
            weight_con_con=0.5,
            device=device_str
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/100 - Loss: {loss.item():.4f}")

        # 保存最优模型
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), save_path)

    # 训练结束加载最优模型
    model.load_state_dict(torch.load(save_path))
    model.eval()

    # 计算最终embedding
    with torch.no_grad():
        final_emb = model(g, feat)

    return model, final_emb


def load_and_visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据（你自己实现的加载函数）
    ex_emb, con_emb = load_embeddings()  # tensor, [num_ex, feat_dim]
    ex2con, con2con = load_relations()
    num_ex, num_con = ex_emb.size(0), con_emb.size(0)

    # 构建图
    g = build_graph(ex2con, con2con, num_ex, num_con)
    g = g.to(device)

    # 初始化模型
    model = RGATEmbedder(in_dim=1024, hid_dim=256, out_dim=99, rel_names=g.etypes)
    model.load_state_dict(torch.load('best_rgat_model.pt'))
    model.to(device)
    model.eval()

    feat = {
        'exercise': ex_emb.to(device),
        'concept': con_emb.to(device)
    }

    # 计算embedding
    with torch.no_grad():
        final_emb = model(g, feat)
        final_emb['exercise'] = F.normalize(final_emb['exercise'], dim=1)
        final_emb['concept'] = F.normalize(final_emb['concept'], dim=1)

    # 保存为npy
    np.save('exercise_embedding_graph.npy', final_emb['exercise'].cpu().numpy())
    np.save('concept_embedding_graph.npy', final_emb['concept'].cpu().numpy())
    print("Saved embeddings to exercise_embedding.npy and concept_embedding.npy")

    # 可视化
    # visualize_embeddings(final_emb['exercise'], final_emb['concept'])
def visualize_embeddings(ex_embed, con_embed):
    # 从 tensor 转 numpy
    ex_np = ex_embed.cpu().detach().numpy()
    con_np = con_embed.cpu().detach().numpy()

    # 合并两个embedding
    combined = np.vstack([ex_np, con_np])

    # t-SNE降到二维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    combined_2d = tsne.fit_transform(combined)

    ex_2d = combined_2d[:ex_np.shape[0]]
    con_2d = combined_2d[ex_np.shape[0]:]

    plt.figure(figsize=(10, 7))
    plt.scatter(ex_2d[:, 0], ex_2d[:, 1], c='blue', label='Exercise', alpha=0.6, s=10)
    plt.scatter(con_2d[:, 0], con_2d[:, 1], c='red', label='Concept', alpha=0.6, s=10)
    plt.legend()
    plt.title("t-SNE Visualization of Exercise and Concept Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()
# ========== 运行 ==========
if __name__ == '__main__':
    # load_and_visualize()
    # train()
    exercise_embedding_graph = np.load('exercise_embedding_graph.npy')
    concept_embedding_graph = np.load('concept_embedding_graph.npy')
    print(exercise_embedding_graph.shape, concept_embedding_graph.shape)


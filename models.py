import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import modules
from torch.distributions import Categorical


class iekt(nn.Module):
    def __init__(self, args, exercise_bert_emb, concept_bert_emb):
        super().__init__()
        self.node_dim = args.dim  # 64
        self.concept_num = args.concept_num
        self.max_concept = args.max_concepts
        self.device = args.device
        self.predictor = modules.funcs(args.n_layer, args.dim * 5, 1, args.dropout)
        self.cog_matrix = nn.Parameter(torch.randn(args.cog_levels, args.dim * 2).to(args.device), requires_grad=True)
        self.acq_matrix = nn.Parameter(torch.randn(args.acq_levels, args.dim * 2).to(args.device), requires_grad=True)
        self.select_preemb = modules.funcs(args.n_layer, args.dim * 3, args.cog_levels, args.dropout)
        self.checker_emb = modules.funcs(args.n_layer, args.dim * 12, args.acq_levels, args.dropout)
        self.prob_emb = nn.Parameter(torch.randn(args.problem_number - 1, args.dim).to(args.device), requires_grad=True)
        self.gru_h = modules.mygru(0, args.dim * 4, args.dim)

        showi0 = []
        for i in range(0, args.n_epochs):
            showi0.append(i)
        # --- 1. 注册原始 BERT 向量为 buffer（不训练，但自动迁移设备、可保存） ---
        self.register_buffer("exercise_text_emb", exercise_bert_emb)
        self.register_buffer("concept_text_emb", concept_bert_emb)
        self.text_proj = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
        )  # 降维后的题目文本向量
        self.text_conj = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
        )  # 降维后的知识点向量
        self.show_index = torch.tensor(showi0).to(args.device)
        self.concept_emb = nn.Parameter(torch.randn(self.concept_num - 1, args.dim).to(args.device), requires_grad=True)
        self.sigmoid = torch.nn.Sigmoid()
        # self.attention_layer = nn.Sequential(
        #     nn.Linear(args.dim * 2, 64),  # h与知识点嵌入拼接后维度
        #     nn.Tanh(),
        #     nn.Dropout(0.3),
        #     nn.Linear(64, 1)
        # )
        # self.gate_net = nn.Sequential(
        #     nn.Linear(args.dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1),
        #     nn.Sigmoid()  # 输出范围 [0, 1]
        # )

    def get_ques_representation(self, prob_ids, related_concept_index, filter0, data_len, h=None):
        """
        :prob_ids: [batch]
        :related_concept_index: [batch, max_concept]
        :filter0: [batch, max_concept]
        :data_len: batch size
        :h: [batch, dim]  # 当前状态的隐藏表示
        """

        # concepts_cat = torch.cat(
        #     [torch.zeros(1, self.node_dim).to(self.device),
        #      self.concept_emb],
        #     dim=0).unsqueeze(0).repeat(data_len, 1, 1)  # concepts_cat:[batch, concept_num, dim]
        reduced_prob_emb = self.text_proj(self.exercise_text_emb)
        reduced_concept_emb = self.text_proj(self.concept_text_emb)
        concepts_cat = reduced_concept_emb.unsqueeze(0).repeat(data_len, 1, 1)  # concepts_cat:[batch, concept_num, dim]
        r_index = self.show_index[0: data_len].unsqueeze(1).repeat(1, self.max_concept)  # [batch, max_concept]
        related_concepts = concepts_cat[r_index, related_concept_index, :]
        # filter_sum = torch.sum(filter0, dim=1)
        #
        # div = torch.where(filter_sum == 0,
        #                   torch.tensor(1.0).to(self.device),
        #                   filter_sum
        #                   ).unsqueeze(1).repeat(1, self.node_dim)
        #
        # concept_level_rep = torch.sum(related_concepts, dim=1) / div

        # 计算状态和知识点的注意力权重
        h_expanded = h.unsqueeze(1).expand(-1, self.max_concept, -1)  # 扩展h的维度
        combined = torch.cat([h_expanded, related_concepts], dim=2)  # [batch, max_concept, dim*2]

        # 计算注意力分数并应用掩码
        attention_scores = self.attention_layer(combined).squeeze(-1)  # [batch, max_concept]
        # attention_scores = torch.sum(h_expanded * related_concepts, dim=-1) / math.sqrt(self.node_dim)
        attention_scores = attention_scores.masked_fill(filter0 == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, max_concept]

        question_embs = self.exercise_text_emb[prob_ids]  # [batch, emb_dim]
        concept_embs = self.concept_text_emb  # [concept_num, emb_dim]
        text_weights = self.compute_text_weights(question_embs, concept_embs,
                                                 related_concept_index)  # [batch, max_concept]
        text_weights = text_weights.masked_fill(filter0 == 0, -1e9)

        text_weights = F.softmax(text_weights, dim=1)
        gate = self.gate_net(h)  # [batch, 1]
        final_weights = gate * text_weights + (1 - gate) * attention_weights  # [batch, max_concept]

        # 加权求和
        concept_level_rep = torch.sum(
            related_concepts * final_weights.unsqueeze(-1),
            dim=1
        )  # [batch, dim]

        # prob_cat = torch.cat([
        #     torch.zeros(1, self.node_dim).to(self.device),
        #     self.prob_emb], dim=0)

        item_emb = reduced_prob_emb[prob_ids]

        v = torch.cat(
            [concept_level_rep,
             item_emb],
            dim=1)  # V: [batch, dim*2]
        return v, related_concepts, attention_weights

    def get_ques_representation_ave(self, prob_ids, related_concept_index, filter0, data_len, ):
        # concepts_cat = torch.cat(
        #     [torch.zeros(1, self.node_dim).to(self.device),
        #      self.concept_emb],
        #     dim=0).unsqueeze(0).repeat(data_len, 1, 1)  # concepts_cat:[batch, concept_num, dim]
        reduced_prob_emb = self.text_proj(self.exercise_text_emb)
        reduced_concept_emb = self.text_proj(self.concept_text_emb)
        concepts_cat = reduced_concept_emb.unsqueeze(0).repeat(data_len, 1, 1)  # concepts_cat:[batch, concept_num, dim]
        r_index = self.show_index[0: data_len].unsqueeze(1).repeat(1, self.max_concept)  # [batch, max_concept]
        related_concepts = concepts_cat[r_index, related_concept_index, :]
        # filter_sum = torch.sum(filter0, dim=1)
        #
        # div = torch.where(filter_sum == 0,
        #                   torch.tensor(1.0).to(self.device),
        #                   filter_sum
        #                   ).unsqueeze(1).repeat(1, self.node_dim)
        #
        # concept_level_rep = torch.sum(related_concepts, dim=1) / div

        question_embs = self.exercise_text_emb[prob_ids]  # [batch, emb_dim]
        concept_embs = self.concept_text_emb  # [concept_num, emb_dim]
        text_weights = self.compute_text_weights(question_embs, concept_embs,
                                                 related_concept_index)  # [batch, max_concept]
        text_weights = text_weights.masked_fill(filter0 == 0, -1e9)

        text_weights = F.softmax(text_weights, dim=1)
        # 加权求和
        concept_level_rep = torch.sum(
            related_concepts * text_weights.unsqueeze(-1),
            dim=1
        )  # [batch, dim]
        item_emb = reduced_prob_emb[prob_ids]
        # prob_cat = torch.cat([
        #     torch.zeros(1, self.node_dim).to(self.device),
        #     self.prob_emb], dim=0)
        #
        # item_emb = prob_cat[prob_ids]

        v = torch.cat(
            [item_emb,concept_level_rep],
            dim=1)  # V: [batch, dim*2]
        return v

    def pi_cog_func(self, x, softmax_dim=1):
        return F.softmax(self.select_preemb(x), dim=softmax_dim)

    def obtain_v(self, this_input, h, x, emb):
        last_show, problem, related_concept_index, show_count, operate, filter0, prob_ids, related_concept_matrix = this_input

        data_len = operate.size()[0]

        v = self.get_ques_representation_ave(prob_ids, related_concept_index, filter0, data_len)
        # v, related_concepts, attention_weights = self.get_ques_representation(prob_ids, related_concept_index, filter0,
        #                                                                       data_len, h=h)
        predict_x = torch.cat([h, v], dim=1)
        h_v = torch.cat([h, v], dim=1)
        prob = self.predictor(torch.cat([
            predict_x, emb
        ], dim=1))  # [batch, 1]
        return h_v, v, prob, x

    def update_state(self, h, v, emb, operate):
        """
        :operate : [batch, 1]  # 0 or 1真实情况
        """
        v_cat = torch.cat([
            v.mul(operate.repeat(1, self.node_dim * 2)),
            v.mul((1 - operate).repeat(1, self.node_dim * 2))], dim=1)
        e_cat = torch.cat([
            emb.mul((1 - operate).repeat(1, self.node_dim * 2)),
            emb.mul((operate).repeat(1, self.node_dim * 2))], dim=1)
        inputs = v_cat + e_cat
        next_p_state = self.gru_h(inputs, h)
        return next_p_state

    def pi_sens_func(self, x, softmax_dim=1):  #
        return F.softmax(self.checker_emb(x), dim=softmax_dim)

    # def predict_multi_sensitivity(self, state, concept_embeddings, mask):
    #     """
    #     预测每个知识点的敏感度向量,但是不是是针对每个关联知识点点才取动作的，是batch*max_concepts个动作
    #     :param state: 当前状态 [batch, dim*12]
    #     :param concept_embeddings: 知识点嵌入 [batch, max_concepts, dim]
    #     :param mask: 知识点掩码 [batch, max_concepts]
    #     :return:
    #         sensitivity_vectors: 敏感度向量 [batch, max_concepts, dim*2]
    #         actions: 动作索引 [batch, max_concepts]
    #         log_probs: 对数概率 [batch, max_concepts]
    #     """
    #     batch_size, max_concepts, dim = concept_embeddings.shape
    #
    #     # 准备输入：将状态与每个知识点嵌入拼接
    #     # state_expanded = state.unsqueeze(1).repeat(1, max_concepts, 1)  # [batch, max_concepts, dim*12]
    #     # combined = torch.cat([state_expanded, concept_embeddings], dim=2)  # [batch, max_concepts, dim*12+dim]
    #
    #     # 展平处理以便批量计算
    #     flat_combined = state.view(-1, dim * 12)  # [batch*max_concepts, dim*12]
    #     flat_probs = self.pi_sens_func(flat_combined)  # [batch*max_concepts, acq_levels]
    #
    #     # 创建分布并采样
    #     m = Categorical(flat_probs)
    #     flat_actions = m.sample()  # [batch*max_concepts]
    #     flat_log_probs = m.log_prob(flat_actions)  # [batch*max_concepts]
    #     flat_vectors = self.acq_matrix[flat_actions]  # [batch*max_concepts, dim*2]
    #
    #     # 恢复原始形状
    #     sensitivity_vectors = flat_vectors.view(batch_size, max_concepts, -1)  # [batch, max_concepts, dim*2]
    #     actions = flat_actions.view(batch_size, max_concepts)  # [batch, max_concepts]
    #     log_probs = flat_log_probs.view(batch_size, max_concepts)  # [batch, max_concepts]
    #
    #     # 应用掩码：无效知识点的向量置零
    #     mask_expanded = mask.unsqueeze(-1)  # [batch, max_concepts, 1]
    #     sensitivity_vectors = sensitivity_vectors * mask_expanded.float()  # [batch, max_concepts, dim*2]
    #
    #     return sensitivity_vectors, actions, log_probs
    def predict_multi_sensitivity(self, state, concept_embeddings, mask):
        """
        预测每个知识点的敏感度向量（过滤无效知识点采样）
        :param state: 当前状态 [batch,max_concepts, dim*12]
        :param concept_embeddings: 知识点嵌入 [batch, max_concepts, dim]
        :param mask: 知识点掩码 [batch, max_concepts]
        :return:
            sensitivity_vectors: 敏感度向量 [batch, max_concepts, dim*2]
            actions: 动作索引 [batch, max_concepts]
            log_probs: 对数概率 [batch, max_concepts]
        """
        batch_size, max_concepts, dim = concept_embeddings.shape
        device = state.device

        # 展平成一维，方便索引
        flat_mask = mask.view(-1)  # [batch*max_concepts]

        # 找有效知识点索引
        valid_indices = flat_mask.nonzero(as_tuple=False).squeeze(1)  # [valid_num]

        # 状态为[batch,max_concepts, dim*12]
        expanded_state = state.view(-1,
                                    dim * 12)  # [batch*max_concepts, dim*12]

        # 只取有效点对应状态
        valid_states = expanded_state[valid_indices]  # [valid_num, dim*12]

        # 计算有效点动作概率
        valid_probs = self.pi_sens_func(valid_states)  # [valid_num, acq_levels]

        # 采样
        m = Categorical(valid_probs)
        valid_actions = m.sample()  # [valid_num]
        valid_log_probs = m.log_prob(valid_actions)  # [valid_num]
        valid_vectors = self.acq_matrix[valid_actions]  # [valid_num, dim*2]

        # 准备全零张量，用于存放完整结果
        flat_actions = torch.zeros(batch_size * max_concepts, dtype=torch.long, device=device)
        flat_log_probs = torch.zeros(batch_size * max_concepts, device=device)
        flat_vectors = torch.zeros(batch_size * max_concepts, dim * 2, device=device)

        # 把有效点结果写回
        flat_actions[valid_indices] = valid_actions
        flat_log_probs[valid_indices] = valid_log_probs
        flat_vectors[valid_indices] = valid_vectors

        # 恢复原始形状
        actions = flat_actions.view(batch_size, max_concepts)
        log_probs = flat_log_probs.view(batch_size, max_concepts)
        # 应用掩码：无效知识点的向量置零
        mask_expanded = mask.unsqueeze(-1)  # [batch, max_concepts, 1]
        sensitivity_vectors = flat_vectors.view(batch_size, max_concepts, dim * 2)
        sensitivity_vectors = sensitivity_vectors * mask_expanded.float()  # [batch, max_concepts, dim*2]

        return sensitivity_vectors, actions, log_probs

    def get_concept_representation(self, prob_ids, related_concept_index, filter0, data_len, h=None):
        """
        基于知识点颗粒度构造每个题目的多个表示: [h, question_emb, concept_i_emb]
    输出形状：[batch, max_concept, dim * 3]"""
        # 获取知识点嵌入矩阵
        # concepts_cat = torch.cat(
        #     [torch.zeros(1, self.node_dim).to(self.device), self.concept_emb],
        #     dim=0
        # ).unsqueeze(0).repeat(data_len, 1, 1)  # [batch, concept_num, dim]
        reduced_prob_emb = self.text_proj(self.exercise_text_emb)
        reduced_concept_emb = self.text_proj(self.concept_text_emb)
        concepts_cat = reduced_concept_emb.unsqueeze(0).repeat(data_len, 1, 1)
        r_index = self.show_index[0: data_len].unsqueeze(1).repeat(1, self.max_concept)  # [batch, max_concept]
        related_concepts = concepts_cat[r_index, related_concept_index, :]  # [batch, max_concept, dim]
        # 获取题目嵌入
        # prob_cat = torch.cat(
        #     [torch.zeros(1, self.node_dim).to(self.device), self.prob_emb],
        #     dim=0
        # )  # [num_prob + 1, dim]
        # question_emb = prob_cat[prob_ids]  # [batch, dim]
        question_emb = reduced_prob_emb[prob_ids]
        # 扩展题目和状态向量为每个知识点一份
        question_expand = question_emb.unsqueeze(1).expand(-1, self.max_concept, -1)  # [batch, max_concept, dim]
        h_expand = h.unsqueeze(1).expand(-1, self.max_concept, -1)  # [batch, max_concept, dim]
        # 拼接为三元组向量
        triple_rep = torch.cat([h_expand, question_expand, related_concepts], dim=-1)  # [batch, max_concept, dim*3]

        # 掩码处理：保留有效知识点
        mask = filter0.unsqueeze(-1).expand(-1, -1, triple_rep.size(-1))  # [batch, max_concept, dim*3]
        triple_rep = triple_rep * mask  # 将无效知识点位置归零

        return triple_rep  # [batch, max_concept, dim*3]

    def compute_text_weights(self, question_embs, concept_embs, related_concept_index):
        # question_embs: [batch, emb_dim]
        # concept_embs: [concept_num, emb_dim]
        # related_concept_index: [batch, max_concept]

        batch, max_concept = related_concept_index.shape
        concept_related = concept_embs[related_concept_index]  # [batch, max_concept, emb_dim]
        question_expanded = question_embs.unsqueeze(1).expand(-1, max_concept, -1)

        text_weights = F.cosine_similarity(question_expanded, concept_related, dim=-1)  # [batch, max_concept]
        # text_weights = F.softmax(cos_sim, dim=-1)  # 归一化为权重
        return text_weights  # [batch, max_concept]

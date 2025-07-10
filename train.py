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


def train(model, loaders, args):
    log.info("training...")
    best_valid_auc = 0
    patience = 30
    no_improve = 0

    BCELoss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    train_sigmoid = torch.nn.Sigmoid()
    show_loss = 100
    for epoch in range(args.n_epochs):
        loss_all = 0
        for step, data in enumerate(loaders['train']):

            with torch.no_grad():
                x, y = batch_data_to_device(data, args.device)
            model.train()
            data_len = len(x[0])
            h = torch.zeros(data_len, args.dim).to(args.device)
            p_action_list, pre_state_list, emb_action_list, op_action_list, actual_label_list, states_list, reward_list, predict_list, ground_truth_list = [], [], [], [], [], [], [], [], []
            log_probs_list, mask_list, attn_weights_list = [], [], []
            rt_x = torch.zeros(data_len, 1, args.dim * 2).to(args.device)
            for seqi in range(0, args.seq_len):  # 每个时间步
                # 数据结构中x[1][seqi] = [ques_id, prob_id, related_concept_index, filter0, out_operate_groundtruth, prob_id, ques_representation]
                # v, concept_embeddings, attention_weights = model.get_ques_representation_ave(
                #     x[1][seqi][6], x[1][seqi][2], x[1][seqi][5], x[1][seqi][5].size()[0], h=h
                # )  # x[1][seqi][6] prob_id,题目id，x[1][seqi][2]题目关联矩阵,x[1][seqi][5]这个是用来表示，知识点的掩码表示，有就是1，没有就是0
                v = model.get_ques_representation_ave(
                    x[1][seqi][6], x[1][seqi][2], x[1][seqi][5], x[1][seqi][5].size()[0])

                ques_h = torch.cat([v, h],
                                   dim=1)  # ques_h:[batch, dim * 2+dim * 1]，ques_representation是一个向量，表示题目和知识点的嵌入加上当时的状态h,cat[v,h]
                flip_prob_emb = model.pi_cog_func(ques_h)                 # flip_prob_emb:[batch, cog_levels]，cog_levels是认知估计的响应动作空间

                m = Categorical(flip_prob_emb)
                emb_ap = m.sample()
                emb_p = model.cog_matrix[emb_ap, :]  # 这是认知向量 [batch, dim * 2]，
                h_v, v, logits, rt_x = model.obtain_v(x[1][seqi], h, rt_x, emb_p)  # h_v是cat[h,v]
                # logits为预测的概率值
                prob = train_sigmoid(logits)  # 存在负数，所以需要sigmoid，保证输出的是0-1之间的概率
                # [batch, 1]，表示做题的概率
                out_operate_groundtruth = x[1][seqi][4]  # [batch, 1]
                out_x_groundtruth = torch.cat([
                    h_v.mul(out_operate_groundtruth.repeat(1, h_v.size()[-1]).float()),
                    h_v.mul((1 - out_operate_groundtruth).repeat(1, h_v.size()[-1]).float())],
                    dim=1)  # 实际的做题结果[batch, dim * 6]

                out_operate_logits = torch.where(prob > 0.5, torch.tensor(1).to(args.device),
                                                 torch.tensor(0).to(args.device))

                out_x_logits = torch.cat([
                    h_v.mul(out_operate_logits.repeat(1, h_v.size()[-1]).float()),
                    h_v.mul((1 - out_operate_logits).repeat(1, h_v.size()[-1]).float())],
                    dim=1)  # 预测出的操作[batch, dim * 6]
                out_x = torch.cat([out_x_groundtruth, out_x_logits], dim=1)

                ground_truth = x[1][seqi][4].squeeze(-1)  # [batch]，真实的做题结果

                # 把每个状态都单独进行控制，状态和题目表示公用一个
                h_v_by_concept = model.get_concept_representation(x[1][seqi][6], x[1][seqi][2], x[1][seqi][5],
                                                                  x[1][seqi][5].size()[0], h=h)

                # Step 1: 扩展为 [batch, max_concept, 1]
                groundtruth_expanded = out_operate_groundtruth.unsqueeze(2).repeat(1, h_v_by_concept.size(1), h_v_by_concept.size(2))  # [batch, 1, 1]

                logits_expanded = out_operate_logits.unsqueeze(2).repeat(1, h_v_by_concept.size(1), h_v_by_concept.size(2))

                # Step 2: 执行与原来一样的拼接操作
                groundtruth_by_concept = torch.cat([
                    h_v_by_concept * groundtruth_expanded.float(),
                    h_v_by_concept * (1 - groundtruth_expanded).float()
                ], dim=-1)  # [batch, max_concept, dim*6]

                logits_by_concept = torch.cat([
                    h_v_by_concept * logits_expanded.float(),
                    h_v_by_concept * (1 - logits_expanded).float()
                ], dim=-1)  # [batch, max_concept, dim*6]

                # Step 3: 拼接 groundtruth 和 logits 特征
                out_x_by_concept = torch.cat([groundtruth_by_concept, logits_by_concept],
                                             dim=-1)  # [batch, max_concept, dim*12]
                # 修改：获取多知识点敏感度向量
                sensitivity_vectors, emb_actions, log_probs = model.predict_multi_sensitivity(
                    out_x_by_concept,  # 当前状态 [batch, dim*12]
                    concept_embeddings,  # 知识点嵌入 [batch, max_concepts, dim]
                    x[1][seqi][5]  # 知识点掩码 [batch, max_concepts]
                )
                # 使用注意力权重加权融合敏感度向量
                # attention_weights: [batch, max_concepts]
                # sensitivity_vectors: [batch, max_concepts, dim*2]
                # 扩展注意力权重维度以进行加权
                attn_weights_expanded = attention_weights.unsqueeze(-1)  # [batch, max_concepts, 1]
                fused_vector = torch.sum(sensitivity_vectors * attn_weights_expanded, dim=1)  # [batch, dim*2]

                # 使用融合的敏感度向量更新状态
                h = model.update_state(h, v, fused_vector, ground_truth.unsqueeze(1))

                # out_x_expanded= out_x.unsqueeze(1).expand(-1, args.max_concepts, -1) # [batch, max_concepts, dim * 12]
                # flip_prob_emb = model.pi_sens_func(out_x_expanded)# [batch, acq_levels]，acq_levels是敏感度估计的响应动作空间
                #
                # m = Categorical(flip_prob_emb)
                # emb_a = m.sample()
                # emb = model.acq_matrix[emb_a,:]
                # h = model.update_state(h, v, emb, ground_truth.unsqueeze(1))

                # 保存多知识点动作和概率
                emb_action_list.append(emb_actions)  # [batch, max_concepts] 最终（step,batch, max_concepts）
                log_probs_list.append(log_probs)  # [batch, max_concepts]
                mask_list.append(x[1][seqi][5])  # [batch, max_concepts] 注意：这里保存的是掩码，用于后续损失计算
                attn_weights_list.append(attention_weights)  # 保存注意力权重用于损失计算

                p_action_list.append(emb_ap)  # 认知向量索引[batch]
                states_list.append(out_x)  # 原论文中的cat[vp，vg]
                pre_state_list.append(ques_h)  # 表示题目和知识点的嵌入加上当时的状态h,cat[v,h]

                ground_truth_list.append(ground_truth)
                predict_list.append(logits.squeeze(1))
                this_reward = torch.where(out_operate_logits.squeeze(1).float() == ground_truth,
                                          torch.tensor(1).to(args.device),
                                          torch.tensor(0).to(args.device))  # (32,)
                reward_list.append(this_reward)

            seq_num = x[0]
            # emb_action_tensor = torch.stack(emb_action_list, dim = 1)#tensor(32,200)
            p_action_tensor = torch.stack(p_action_list, dim=1)  # tensor(32,200)
            state_tensor = torch.stack(states_list, dim=1)  # tensor(32,768,200)
            pre_state_tensor = torch.stack(pre_state_list, dim=1)  # tensor(32,200,192)
            reward_tensor = torch.stack(reward_list, dim=1).float() / (
                seq_num.unsqueeze(-1).repeat(1, args.seq_len)).float()  # tensor(32,200)
            logits_tensor = torch.stack(predict_list, dim=1)  # tensor(32,200)
            ground_truth_tensor = torch.stack(ground_truth_list, dim=1)  # tensor(32,200)
            loss = []
            tracat_logits = []
            tracat_ground_truth = []

            for i in range(0, data_len):  # data_len 样本数量32
                this_seq_len = seq_num[i]  # 时间步长度
                this_reward_list = reward_tensor[i]  #

                this_cog_state = torch.cat([pre_state_tensor[i][0: this_seq_len],
                                            torch.zeros(1, pre_state_tensor[i][0].size()[0]).to(args.device)
                                            ], dim=0)  # 样本有效长度+1，向量长度（192）
                this_sens_state = torch.cat([state_tensor[i][0: this_seq_len],
                                             torch.zeros(1, state_tensor[i][0].size()[0]).to(args.device)
                                             ], dim=0)  # 样本有效长度+1，向量长度（768）

                td_target_cog = this_reward_list[0: this_seq_len].unsqueeze(1)  # 样本有效长度+1，1
                delta_cog = td_target_cog
                delta_cog = delta_cog.detach().cpu().numpy()

                td_target_sens = this_reward_list[0: this_seq_len].unsqueeze(1)
                delta_sens = td_target_sens
                delta_sens = delta_sens.detach().cpu().numpy()

                advantage_lst_cog = []
                advantage = 0.0
                for delta_t in delta_cog[::-1]:
                    advantage = args.gamma * advantage + delta_t[0]
                    advantage_lst_cog.append([advantage])
                advantage_lst_cog.reverse()  # 优势值根据时间步反转一下
                advantage_cog = torch.tensor(advantage_lst_cog, dtype=torch.float).to(args.device)  # 样本序列长度，1

                pi_cog = model.pi_cog_func(this_cog_state[:-1])  # 除了最后一个，因为最后一个是全0的,样本长度*10

                pi_a_cog = pi_cog.gather(1, p_action_tensor[i][0: this_seq_len].unsqueeze(1))  # 取出该动作的概率值，#样本序列长度，1

                loss_cog = -torch.log(pi_a_cog) * advantage_cog  # 样本序列长度，1

                loss.append(torch.sum(loss_cog))

                advantage_lst_sens = []
                advantage = 0.0
                for delta_t in delta_sens[::-1]:
                    # advantage = args.gamma * args.beta * advantage + delta_t[0]
                    advantage = args.gamma * advantage + delta_t[0]
                    advantage_lst_sens.append([advantage])
                advantage_lst_sens.reverse()
                advantage_sens = torch.tensor(advantage_lst_sens, dtype=torch.float).to(args.device)

                # pi_sens = model.pi_sens_func(this_sens_state[:-1])
                # pi_a_sens = pi_sens.gather(1,emb_action_tensor[i][0: this_seq_len].unsqueeze(1))
                #
                # loss_sens = - torch.log(pi_a_sens) * advantage_sens
                # loss.append(torch.sum(loss_sens))

                # 修改：计算多知识点敏感度损失
                # 获取该学生的所有知识点动作、log概率和注意力权重
                all_emb_actions = torch.stack(
                    [emb_action_list[t][i] for t in range(this_seq_len)])  # [seq_len, max_concepts]
                all_log_probs = torch.stack(
                    [log_probs_list[t][i] for t in range(this_seq_len)])  # [seq_len, max_concepts]
                all_masks = torch.stack([mask_list[t][i] for t in range(this_seq_len)])  # [seq_len, max_concepts]
                all_attn_weights = torch.stack(
                    [attn_weights_list[t][i] for t in range(this_seq_len)])  # [seq_len, max_concepts]

                # 计算每个知识点的损失（只考虑有效知识点）
                per_concept_loss = -all_log_probs * advantage_sens

                # 应用注意力权重加权
                weighted_loss = per_concept_loss * all_attn_weights

                # 应用掩码并归一化
                valid_mask = all_masks.bool()  # [step,max_concepts]
                num_valid_per_step = valid_mask.sum(dim=1, keepdim=True)

                # 对每个时间步的有效知识点损失求和
                masked_loss = torch.where(valid_mask, weighted_loss, torch.zeros_like(weighted_loss))
                loss_sens = masked_loss.sum(dim=1) / num_valid_per_step.squeeze(1)

                loss.append(torch.sum(loss_sens))

                # 计算 BCE损失
                this_prob = logits_tensor[i][0: this_seq_len]  # 样本长度
                this_groud_truth = ground_truth_tensor[i][0: this_seq_len]  # 样本长度

                tracat_logits.append(this_prob)
                tracat_ground_truth.append(this_groud_truth)

            bce = BCELoss(torch.cat(tracat_logits, dim=0), torch.cat(tracat_ground_truth, dim=0))

            label_len = torch.cat(tracat_ground_truth, dim=0).size()[0]
            loss_l = sum(loss)
            # lamb=get_dynamic_lambda_dec(epoch)
            rl_part = args.lamb * (loss_l / label_len)
            # rl_part = lamb * (loss_l / label_len)

            loss = rl_part + bce
            log.info(f"Epoch: {epoch:03d},lamb:{args.lamb},RL Loss: {rl_part:.4f}, BCE Loss: {bce:.4f}, Ratio (RL/BCE): {rl_part / bce:.2f}")
            loss_all += loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        show_loss = loss_all / len(loaders['train'].dataset)
        acc, auc = evaluate(model, loaders['valid'], args)
        tacc, tauc = evaluate(model, loaders['test'], args)
        log.info(
            'Epoch: {:03d}, Loss: {:.7f}, valid acc: {:.7f}, valid auc: {:.7f}, test acc: {:.7f}, test auc: {:.7f}'.format(
                epoch, show_loss, acc, auc, tacc, tauc))

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(model, os.path.join(args.run_dir, 'params_%i.pt' % epoch))
        scheduler.step(auc)
        # Early stopping based on validation AUC
        if auc > best_valid_auc:
            best_valid_auc = auc
            no_improve = 0
            torch.save(model, os.path.join(args.run_dir, 'best_model.pt'))
        else:
            no_improve += 1
        if no_improve >= patience:
            log.info(f'Early stopping triggered after {epoch}  without improvement')
            break


def evaluate(model, loader, args):
    model.eval()
    eval_sigmoid = torch.nn.Sigmoid()
    y_list, prob_list, final_action = [], [], []

    for step, data in enumerate(loader):

        with torch.no_grad():
            x, y = batch_data_to_device(data, args.device)
        model.train()
        data_len = len(x[0])
        h = torch.zeros(data_len, args.dim).to(args.device)
        batch_probs, uni_prob_list, actual_label_list, states_list, reward_list = [], [], [], [], []
        H = None
        if 'eernna' in args.model:
            H = torch.zeros(data_len, 1, args.dim).to(args.device)
        else:
            H = torch.zeros(data_len, args.concept_num - 1, args.dim).to(args.device)
        rt_x = torch.zeros(data_len, 1, args.dim * 2).to(args.device)
        for seqi in range(0, args.seq_len):
            # v, concept_embeddings, attention_weights = model.get_ques_representation(
            #     x[1][seqi][6], x[1][seqi][2], x[1][seqi][5], x[1][seqi][5].size()[0], h=h
            # )
            v= model.get_ques_representation(
                x[1][seqi][6], x[1][seqi][2], x[1][seqi][5], x[1][seqi][5].size()[0]
            )
            ques_h = torch.cat([v, h], dim=1)
            flip_prob_emb = model.pi_cog_func(ques_h)

            m = Categorical(flip_prob_emb)
            emb_ap = m.sample()
            emb_p = model.cog_matrix[emb_ap, :]

            h_v, v, logits, rt_x = model.obtain_v(x[1][seqi], h, rt_x, emb_p)
            prob = eval_sigmoid(logits)
            out_operate_groundtruth = x[1][seqi][4]
            out_x_groundtruth = torch.cat([
                h_v.mul(out_operate_groundtruth.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1 - out_operate_groundtruth).repeat(1, h_v.size()[-1]).float())],
                dim=1)

            out_operate_logits = torch.where(prob > 0.5, torch.tensor(1).to(args.device),
                                             torch.tensor(0).to(args.device))
            out_x_logits = torch.cat([
                h_v.mul(out_operate_logits.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1 - out_operate_logits).repeat(1, h_v.size()[-1]).float())],
                dim=1)
            out_x = torch.cat([out_x_groundtruth, out_x_logits], dim=1)

            # # 把每个状态都单独进行控制，状态和题目表示公用一个
            # h_v_by_concept = model.get_concept_representation(x[1][seqi][6], x[1][seqi][2], x[1][seqi][5],
            #                                                   x[1][seqi][5].size()[0], h=h)
            #
            # # Step 1: 扩展为 [batch, max_concept, 1]
            # groundtruth_expanded = out_operate_groundtruth.unsqueeze(2).repeat(1, h_v_by_concept.size(1),
            #                                                                    h_v_by_concept.size(2))  # [batch, 1, 1]
            #
            # logits_expanded = out_operate_logits.unsqueeze(2).repeat(1, h_v_by_concept.size(1), h_v_by_concept.size(2))
            #
            # # Step 2: 执行与原来一样的拼接操作
            # groundtruth_by_concept = torch.cat([
            #     h_v_by_concept * groundtruth_expanded.float(),
            #     h_v_by_concept * (1 - groundtruth_expanded).float()
            # ], dim=-1)  # [batch, max_concept, dim*6]
            #
            # logits_by_concept = torch.cat([
            #     h_v_by_concept * logits_expanded.float(),
            #     h_v_by_concept * (1 - logits_expanded).float()
            # ], dim=-1)  # [batch, max_concept, dim*6]
            #
            # # Step 3: 拼接 groundtruth 和 logits 特征
            # out_x_by_concept = torch.cat([groundtruth_by_concept, logits_by_concept], dim=-1)  # [batch, max_concept, dim*12]
            #
            ground_truth = x[1][seqi][4].squeeze(-1)
            # # 修改：获取多知识点敏感度向量
            # sensitivity_vectors, emb_actions, log_probs = model.predict_multi_sensitivity(
            #     out_x_by_concept,  # 当前状态 [batch, dim*12]
            #     concept_embeddings,  # 知识点嵌入 [batch, max_concepts, dim]
            #     x[1][seqi][5]  # 知识点掩码 [batch, max_concepts]
            # )
            # # 使用注意力权重加权融合敏感度向量
            # # attention_weights: [batch, max_concepts]
            # # sensitivity_vectors: [batch, max_concepts, dim*2]
            # # 扩展注意力权重维度以进行加权
            # attn_weights_expanded = attention_weights.unsqueeze(-1)  # [batch, max_concepts, 1]
            # fused_vector = torch.sum(sensitivity_vectors * attn_weights_expanded, dim=1)  # [batch, dim*2]

            # # 使用融合的敏感度向量更新状态
            # h = model.update_state(h, v, fused_vector, ground_truth.unsqueeze(1))
            flip_prob_emb = model.pi_sens_func(out_x)

            m = Categorical(flip_prob_emb)
            emb_a = m.sample()
            emb = model.acq_matrix[emb_a,:]

            h = model.update_state(h, v, emb, ground_truth.unsqueeze(1))
            uni_prob_list.append(prob.detach())

        seq_num = x[0]
        prob_tensor = torch.cat(uni_prob_list, dim=1)
        for i in range(0, data_len):
            this_seq_len = seq_num[i]
            batch_probs.append(prob_tensor[i][0: this_seq_len])
        batch_t = torch.cat(batch_probs, dim=0)
        prob_list.append(batch_t)
        y_list.append(y)

    y_tensor = torch.cat(y_list, dim=0).int()
    hat_y_prob_tensor = torch.cat(prob_list, dim=0)

    acc = accuracy_score(y_tensor.cpu().numpy(), (hat_y_prob_tensor > 0.5).int().cpu().numpy())
    fpr, tpr, thresholds = metrics.roc_curve(y_tensor.cpu().numpy(), hat_y_prob_tensor.cpu().numpy(), pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return acc, auc


def get_dynamic_lambda_asc(epoch, warmup_epochs=5, decay_epochs=10, max_lambda=60, min_lambda=40):
    """lamda的动态调整函数，逐渐增加"""
    if epoch < warmup_epochs:
        return min_lambda
    elif epoch < warmup_epochs + decay_epochs:
        progress = (epoch - warmup_epochs) / decay_epochs
        lambda_val = min_lambda + progress * (max_lambda - min_lambda)
        return lambda_val
    else:
        return max_lambda

def get_dynamic_lambda_dec(epoch, warmup_epochs=5, decay_epochs=10, max_lambda=40, min_lambda=20):
    """lamda的动态调整函数，逐渐下降"""
    if epoch < warmup_epochs:
        return max_lambda
    elif epoch < warmup_epochs + decay_epochs:
        progress = (epoch - warmup_epochs) / decay_epochs
        lambda_val = max_lambda - progress * (max_lambda - min_lambda)
        return lambda_val
    else:
        return min_lambda
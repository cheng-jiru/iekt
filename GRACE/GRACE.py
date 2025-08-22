"""
@Author: jiru.cheng
@Time: 2025/7/24 19:13
@File: GRACE.py
"""

from AC import ActorCritic, Data_P
from PPO import PPO, Data_L
import gym
import networkx as nx
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys


def train(env, L_agent, P_agent, max_episode_num, batch_size):
    L_max_steps = 10
    L_rewards = []
    all_logs_episode_L = []
    all_logs_episode_P = []

    for episode in tqdm.tqdm(range(max_episode_num), desc="Apisode"):
        env.reset()
        logs_episode_L = []
        logs_episode_P = []

        init_profile, _, init_state = env.begin_episode()

        target = list(init_profile["target"])

        logs_episode_L.append({"learning targets": target})
        init_logs = init_profile["logs"]

        init_ques = []
        init_ans = []
        init_concept = []
        for log in init_logs:
            init_ques.append(int(log[0]))
            init_ans.append(int(log[1]))
            init_concept.append(log[2])

        logs_episode_L.append({"length of initial logs": init_profile["init_len"]})
        logs_episode_L.append({"init knowledge state": init_state})
        L_state = init_state
        P_state = init_state
        l_steps = 0
        l_knows = []
        know = 0

        while True:
            l_steps += 1
            learner = env._learner
            # 1. 获取动作
            available_actions = learner._chain_tracker.get_available_actions()

            for concept in available_actions:
                if env.scorer.response_concept_function(learner.state, concept):
                    learner._chain_tracker.update(concept)

            available_actions = learner._chain_tracker.get_available_actions()

            knows, _ = L_agent.take_action_by(L_state, available_actions, )

            l_knows.append(know)
            p_steps = 0
            p_step_reward = 0

            logs_know_P = []
            logs_know_P.append("learning step {} study {} ".format(l_steps, knows))

            # 先把练习题数定死
            tolerance = 20

            for i in tqdm.tqdm(range(int(tolerance)), desc="P-agent"):
                p_steps += 1
                ques = P_agent.take_action(knows, P_state)


if __name__ == '__main__':
    env = gym.make("KSS-v2")
    state_dim = env.action_space.n
    action_dim = env.action_space.n
    hidden_dim = 128
    actor_lr = 0.001
    critic_lr = 0.001
    gamma = 0.98

    lmbda = 0.95
    epochs = 10
    eps = 0.2

    max_episode_num = 30
    batch_size = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    L_agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                  lmbda, epochs, eps, gamma, device, batch_size)

    P_agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr,
                          critic_lr, gamma, env.learning_item_base.knowledge2item, device, batch_size)

    rewards = train(env, L_agent, P_agent, max_episode_num, batch_size)

    print(np.sum(rewards) / len(rewards))

    plt.plot(rewards)
    plt.show()

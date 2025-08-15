"""
@Author: jiru.cheng
@Time: 2025/8/5 16:33
@File: agent_utils.py
"""
import os
import numpy as np

from test_original import KtAgent


def get_graph_embeddings(env_name):
    cur_path = os.path.abspath(os.path.dirname(__file__))# node2vec
    if env_name == 'dbe':
        concept_embeddings_path = os.path.join(cur_path, '../concept_embedding_graph.npy')
        knowledge_embeddings_path = os.path.join(cur_path, '../exercise_embedding_graph.npy')
    else:
        raise ValueError('Wrong graph embedding data path')

    try:
        concept_graph_embeddings = np.load(concept_embeddings_path)
        knowledge_graph_embeddings= np.load(knowledge_embeddings_path)
    except FileNotFoundError:
        print(f'no {cur_path} yet')
    return concept_graph_embeddings,knowledge_graph_embeddings


def get_kt_model():
    model_path = '../run/h_number_graph/best_model.pt'
    agent = KtAgent(model_path, 4, dim=99)
    return agent
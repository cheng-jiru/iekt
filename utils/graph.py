"""
@Author: jiru.cheng
@Time: 2025/7/21 17:42
@File: graph.py
"""
import networkx as nx
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


def build_graph(file_pata):
    """
    构建知识点之间的图
    """
    df = pd.read_csv(file_pata)
    concept_graph = nx.DiGraph()
    for _, row in df.iterrows():
        concept_graph.add_edge(row['from_knowledgecomponent_id'], row['to_knowledgecomponent_id'])
    return concept_graph


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    # 适用于 DAG，构建层次结构布局
    from networkx.drawing.nx_agraph import graphviz_layout
    return graphviz_layout(G, prog='dot')  # 需要安装 graphviz


def is_subchain(shorter, longer):
    n, m = len(shorter), len(longer)
    if n > m:
        return False
    for i in range(m - n + 1):
        if longer[i:i + n] == shorter:
            return True
    return False


def filter_subchains(paths):
    filtered = []
    for path in paths:
        if not any(is_subchain(path, other) for other in paths if other != path):
            filtered.append(path)
    return filtered


def build_learning_chains_best_cover(concept_graph, target_concepts):
    roots = [n for n in concept_graph.nodes if concept_graph.in_degree(n) == 0]
    all_paths = []

    # Step 1: 获取所有到 target 的路径
    for tgt in target_concepts:
        for root in roots:
            try:
                paths = list(nx.all_simple_paths(concept_graph, root, tgt))
                all_paths.extend(paths)
            except nx.NetworkXNoPath:
                continue

    # Step 2: 给每条路径打标签：路径上有哪些目标点
    path_infos = []
    for path in all_paths:
        targets_in_path = [node for node in path if node in target_concepts]
        if targets_in_path:
            path_infos.append({
                "path": path,
                "targets": set(targets_in_path),
                "length": len(path)
            })

    # Step 3: 贪心选择路径，优先选择覆盖多目标点、路径短的
    selected = []
    covered = set()
    while covered < set(target_concepts):
        # 排除完全无用的路径
        candidates = [p for p in path_infos if len(p["targets"] - covered) > 0]
        if not candidates:
            break

        # 排序规则：优先目标多的，其次路径短
        candidates.sort(key=lambda x: (-len(x["targets"] - covered), x["length"]))
        best = candidates[0]
        selected.append(best["path"])
        covered |= best["targets"]

    # Step 4: 过滤子链
    return filter_subchains(selected)


def build_learning_chains_with_fallbacks(concept_graph, target_concepts):
    chains = []
    used_nodes = set()
    fallback_map = {}  # 节点 → 所有未被用掉的其他前驱

    for tgt in target_concepts:
        best_path = None
        best_length = float('inf')

        # 所有入度为0的节点尝试构建路径
        for start in concept_graph.nodes:
            if concept_graph.in_degree(start) == 0:
                try:
                    path = nx.shortest_path(concept_graph, source=start, target=tgt)
                    if any(n in used_nodes for n in path):
                        continue
                    if len(path) < best_length:
                        best_path = path
                        best_length = len(path)
                except:
                    continue

        if best_path:
            chains.append(best_path)
            used_nodes.update(best_path)

            # 记录每个节点潜在的备用前驱（除了主路径）
            for node in best_path[1:]:
                fallback_predecessors = set(concept_graph.predecessors(node)) - set(best_path)
                fallback_map[node] = fallback_predecessors
        else:
            # 无法建路径，就单节点成链
            if tgt not in used_nodes:
                chains.append([tgt])
                used_nodes.add(tgt)

    return chains, fallback_map


if __name__ == '__main__':
    # 构造示例图
    # G = nx.DiGraph()
    # edges = [
    #     ('A', 'B'), ('B', 'C'), ('C', 'D'), ('C', 'E'),
    #     ('F', 'G'), ('E', 'H'), ('H', 'I'), ('J', 'K')
    # ]
    # G.add_edges_from(edges)

    # 示例知识结构图
    G = build_graph("../KC_Relationships_mapped.csv")
    target_concepts = [55, 12, 83, 76, 16,3]
    print(G.number_of_nodes())
    chains = build_learning_chains_best_cover(G, target_concepts)
    print("生成的学习链：", chains)

    # 可视化
    pos = hierarchy_pos(G)
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
    plt.figure(figsize=(20, 14))
    nx.draw_networkx_edges(G, pos, arrows=True)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i, chain in enumerate(chains):
        chain_color = colors[i % len(colors)]
        # 画链上的节点
        nx.draw_networkx_nodes(G, pos, nodelist=chain, node_color=chain_color, node_size=600)
        # 画链上的节点标签，带序号
        labels = {node: f"{node}\n({idx + 1})" for idx, node in enumerate(chain)}
        nx.draw_networkx_labels(G, pos, labels, font_color='white', font_weight='bold')

    # 画剩余节点（不在链上的）
    rest_nodes = set(G.nodes) - set([n for chain in chains for n in chain])
    nx.draw_networkx_nodes(G, pos, nodelist=rest_nodes, node_color='lightgray', node_size=400)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in rest_nodes})

    plt.title("knowledge structure graph with learning chains")
    plt.axis('off')
    plt.show()

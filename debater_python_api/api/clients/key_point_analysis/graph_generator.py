import json
import logging
import pandas as pd
import numpy as np
from debater_python_api.api.clients.key_point_analysis.utils import create_dict_to_list, read_dicts_from_df


def create_graph_data(full_results_df, min_n_similar_matches=5, n_matches_samples=20):
    '''
    translates the result file (full result, not the summary) into a json that is loadable in the kpa-key-points-graph-ui
    :param kpa_result: a KpaResult instance
    :param min_n_similar_matches: minimal number of sentences in the key points intersection for creating an edge (relation between the key points).
    :paran n_matches_samples: maximal number of matching sentences to include for each key point
    :return: a json object with the graph data.
    '''

    def graph_to_graph_data(graph, n_sentences):
        for node in graph['nodes']:
            node['relative_val'] = float(node['n_matches']) / float(n_sentences)

        graph_data = []
        for n in graph['nodes']:
            id = n['id']
            graph_data.append({'type': 'node',
                               'data':
                                   {
                                       'id': str(id),
                                       'kp': str(n["kp"]),
                                       'n_matches': str(n["n_matches"]),
                                       'n_sentences': str(n_sentences),
                                       'relative_val': round(n['relative_val'], 3),
                                       'matches': n['matches']
                                   }})

        for i, e in enumerate(graph['edges']):
            graph_data.append(
                {'type': 'edge',
                 'data':
                     {'id': f'e{i}',
                      'source': str(e['source']),
                      'target': str(e['target']),
                      'score': round(e['score'], 2)
                      }})
        return graph_data

    def get_node(i, kp, kp_to_dicts, n_matches_samples):
        return {'id': i,
                'kp': kp,
                'n_matches': len(kp_to_dicts[kp]),
                'matches': [{'sentence_text': d["sentence_text"],
                             'match_score': float(d["match_score"])}
                            for d in kp_to_dicts[kp][:n_matches_samples]]
                }

    dicts, _ = read_dicts_from_df(full_results_df)
    n_sentences = len(set([d['sentence_text'] for d in dicts]))
    kp_to_dicts = create_dict_to_list([(d['kp'], d) for d in dicts])
    all_kps = set(kp_to_dicts.keys())
    all_kps = all_kps - {'none'}
    all_kps = sorted(list(all_kps), key=lambda kp: len(kp_to_dicts[kp]), reverse=True)
    kp_to_sentences = {kp: set([(d['comment_id'], d['sentence_id']) for d in kp_dicts]) for kp, kp_dicts in
                       kp_to_dicts.items()}

    # edge (i, j): what percentage of sentences in i are also sentences in j
    edge_to_score = {}
    for i in range(len(all_kps)):
        for j in range(len(all_kps)):
            if i == j:
                continue
            else:
                kp_i = all_kps[i]
                kp_j = all_kps[j]
                sents_i = kp_to_sentences[kp_i]
                sents_j = kp_to_sentences[kp_j]
                n_sentences_i_in_j = len([s for s in sents_i if s in sents_j])
                if n_sentences_i_in_j > min_n_similar_matches:
                    edge_to_score[(i, j)] = n_sentences_i_in_j / len(sents_i)
    nodes = [get_node(i, kp, kp_to_dicts, n_matches_samples) for i, kp in enumerate(all_kps)]
    edges = [{'source': i, 'target': j, 'score': edge_to_score[i, j]} for (i, j) in edge_to_score]
    graph = {'nodes': nodes, 'edges': edges}

    return graph_to_graph_data(graph, n_sentences)


def graph_data_to_hierarchical_graph_data(graph_data_json_file=None,
                                          graph_data=None,
                                          filter_small_nodes=-1.0,
                                          filter_min_relations=-1.0,
                                          filter_big_to_small_edges=True):
    if (graph_data_json_file is None and graph_data is None) or (
            graph_data_json_file is not None and graph_data is not None):
        logging.error('Please pass either graph_data_json_file or graph_data')

    if graph_data_json_file is not None:
        print(f'creating hierarchical graph from file: {graph_data_json_file}')
        graph_data = json.load(open(graph_data_json_file))

    nodes = [d for d in graph_data if d['type'] == 'node']
    edges = [d for d in graph_data if d['type'] == 'edge']

    if filter_small_nodes > 0.0:
        nodes = [n for n in nodes if
                 (round(float(n['data']['n_matches']) / float(n['data']['n_sentences']), 3) >= filter_small_nodes)]

    nodes_ids = [n['data']['id'] for n in nodes]
    id_to_node = {d['data']['id']: d for d in nodes}

    if filter_min_relations > 0.0:
        edges = [e for e in edges if float(e['data']['score']) >= filter_min_relations]

    edges = [e for e in edges if (e['data']['source'] in nodes_ids and e['data']['target'] in nodes_ids)]

    if filter_big_to_small_edges:
        edges = [e for e in edges if int(id_to_node[e['data']['source']]['data']['n_matches']) <= int(
            id_to_node[e['data']['target']]['data']['n_matches'])]

    source_target_to_score = {(e['data']['source'], e['data']['target']): e['data']['score'] for e in edges}

    # keep only edges to max parent
    id_to_parents = create_dict_to_list([(e['data']['source'], e['data']['target']) for e in edges])
    id_to_max_parent = {}
    for id, parents in id_to_parents.items():
        scores = [source_target_to_score[(id, parent)] if (id, parent) in source_target_to_score else 0.0 for parent in
                  parents]
        max_parent = parents[np.argmax(scores)]
        id_to_max_parent[id] = max_parent

    edges = [e for e in edges if
             e['data']['source'] in id_to_max_parent and id_to_max_parent[e['data']['source']] == e['data']['target']]
    return nodes + edges

# def hierarchical_graph_data_to_textual_bullets(graph_data_json_file=None, graph_data=None, out_file=None):
#     def get_hierarchical_bullets_aux(id_to_kids, id_to_node, id, tab, res):
#         msg = f'{"  " * tab} * {id_to_node[id]["data"]["kp"]} ({id_to_node[id]["data"]["n_matches"]} matches)'
#         res.append(msg)
#         if id in id_to_kids:
#             kids = id_to_kids[id]
#             kids = sorted(kids, key=lambda n: int(id_to_node[n]['data']['n_matches']), reverse=True)
#             for k in kids:
#                 get_hierarchical_bullets_aux(id_to_kids, id_to_node, k, tab + 1, res)
#
#     def get_hierarchical_bullets(roots, id_to_kids, id_to_node):
#         res = []
#         tab = 0
#         roots = sorted(roots, key=lambda n: int(id_to_node[n]['data']['n_matches']), reverse=True)
#         for root in roots:
#             get_hierarchical_bullets_aux(id_to_kids, id_to_node, root, tab, res)
#         return res
#
#     if (graph_data_json_file is None and graph_data is None) or (
#             graph_data_json_file is not None and graph_data is not None):
#         logging.error('Please pass either graph_data_json_file or graph_data')
#
#     if graph_data_json_file is not None:
#         print(f'creating hierarchical graph from file: {graph_data_json_file}')
#         graph_data = json.load(open(graph_data_json_file))
#
#     nodes = [d for d in graph_data if d['type'] == 'node']
#     edges = [d for d in graph_data if d['type'] == 'edge']
#
#     nodes_ids = [n['data']['id'] for n in nodes]
#     id_to_node = {d['data']['id']: d for d in nodes}
#
#     id_to_kids = create_dict_to_list([(e['data']['target'], e['data']['source']) for e in edges])
#     all_kids = set()
#     for id, kids in id_to_kids.items():
#         all_kids = all_kids.union(kids)
#
#     root_ids = set(nodes_ids).difference(all_kids)
#     bullets_txt = get_hierarchical_bullets(root_ids, id_to_kids, id_to_node)
#     if out_file is not None:
#         logging.info(f'saving textual bullets in file: {out_file}')
#         with open(out_file, 'w') as f:
#             for line in bullets_txt:
#                 f.write("%s\n" % line)
#     return bullets_txt


def filter_graph_by_relation_strength(graph_data, min_relation_strength):
    if min_relation_strength > 0:
        nodes = [d for d in graph_data if d['type'] == 'node']
        edges = [d for d in graph_data if d['type'] == 'edge']
        edges = [e for e in edges if float(e['data']['score']) >= min_relation_strength]
        graph_data = nodes + edges
    return graph_data


def get_hierarchical_graph_from_tree_and_subset_results(graph_data_hierarchical, subset_results_df,
                                                        filter_min_relations_for_text=0.4,
                                                        n_top_matches_in_graph=20):

    n_sentences = len(set(subset_results_df["sentence_text"]))
    results_kps = [kp for kp in set(subset_results_df["kp"]) if kp != "none"]

    nodes = [d for d in graph_data_hierarchical if d['type'] == 'node']
    edges = [d for d in graph_data_hierarchical if d['type'] == 'edge']
    if filter_min_relations_for_text > 0:
        edges = [e for e in edges if float(e['data']['score']) >= filter_min_relations_for_text]

    # change graph nodes to match new results:
    n_kps_in_new_only = len(set(results_kps).difference(set(n["data"]['kp'] for n in nodes)))
    if n_kps_in_new_only > 0:  # no new kps in results
        logging.warning(
            f"New result file contains {n_kps_in_new_only} key points not in the graph data. Not creating summaries.")
        return None

    subset_results_df = subset_results_df[subset_results_df["kp"] != "none"]
    kp_to_results = {k: v for k, v in subset_results_df.groupby(["kp"])}

    new_nodes = []
    for node in nodes:
        node_kp = node["data"]["kp"]
        kp_results = kp_to_results.get(node_kp, pd.DataFrame([], columns=subset_results_df.columns))
        n_matches = len(kp_results)
        sentences_text = list(kp_results["sentence_text"])
        scores = list(kp_results["match_score"])

        n_graph_matches = np.min([n_matches, n_top_matches_in_graph])
        matches = [{"sentence_text": sentences_text[i], "match_score": scores[i]} for i in range(n_graph_matches)]
        rel_val = float(n_matches) / float(n_sentences) if n_sentences else 0
        new_node = {"type": "node", "data": {"id": node["data"]["id"], "kp": node_kp, "n_matches": n_matches,
                                             "n_sentences": n_sentences,
                                             "relative_val": rel_val, "matches": matches}}
        new_nodes.append(new_node)

    # filter nodes with no matches
    removed_idxs = []
    for node in new_nodes:
        if node["data"]["n_matches"] == 0:
            id = node["data"]["id"]

            # edges: specific -> general
            sub_kps = [e["data"]["source"] for e in edges if e["data"]["target"] == id]
            top_kps = [e["data"]["target"] for e in edges if e["data"]["source"] == id]
            # assert len(top_kps) <= 1
            top_kp = top_kps[0] if len(top_kps) == 1 else None

            edges = list(filter(lambda e: id != e["data"]["source"] and id != e["data"]["target"], edges))
            if top_kp:  # connect top and sub kps
                edges.extend(
                    [{"type": "edge", "data": {"source": sub_kp, "target": top_kp, "score": -1}} for sub_kp in
                     sub_kps])
            removed_idxs.append(id)

    # squeeze nodes indices
    new_nodes = [n for n in new_nodes if n["data"]["id"] not in removed_idxs]
    remaining_idxs = [node["data"]["id"] for node in new_nodes]
    new_idxs = list(range(len(remaining_idxs)))
    old_to_new_idx = dict(zip(remaining_idxs, new_idxs))
    for node in new_nodes:
        node["data"]["id"] = old_to_new_idx[node["data"]["id"]]
    for e in edges:
        e["data"]["source"] = old_to_new_idx[e["data"]["source"]]
        e["data"]["target"] = old_to_new_idx[e["data"]["target"]]
        e["data"]["score"] = -1

    return new_nodes + edges

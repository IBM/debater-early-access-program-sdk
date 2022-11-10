import json
import logging
import os
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np

from debater_python_api.api.clients.key_point_analysis.KpaExceptions import KpaIllegalInputException
from debater_python_api.utils.kp_analysis_utils import create_dict_to_list, argmax


class KpAnalysisUtils:
    '''
    A class with static methods for utilities that assist with the key point analysis service.
    '''

    @staticmethod
    def print_result(result, print_matches=False):
        '''
        Prints the key point analysis result to console.
        :param result: the result, returned by method get_result in KpAnalysisTaskFuture.
        '''
        def print_kp(kp, n_matches, n_matches_subtree, depth, print_matches, keypoint_matching):
            spaces = '     ' * depth
            has_n_matches_subtree = n_matches_subtree is not None
            logging.info('%s    %d%s - %s' % (spaces, n_matches_subtree if has_n_matches_subtree else n_matches,
                                       (' - %d' % n_matches) if has_n_matches_subtree else '', kp))
            if print_matches:
                for match in keypoint_matching['matching']:
                    logging.info('%s    %s - %s' % (spaces + '    ', str(match['score']), match['sentence_text']))

        kp_to_n_matches_subtree = defaultdict(int)
        parents = list()
        parent_to_kids = defaultdict(list)
        for keypoint_matching in result['keypoint_matchings']:
            kp = keypoint_matching['keypoint']
            kp_to_n_matches_subtree[kp] += len(keypoint_matching['matching'])
            parent = keypoint_matching.get("parent", None)
            if parent is None or parent == 'root':
                parents.append(keypoint_matching)
            else:
                parent_to_kids[parent].append(keypoint_matching)
                kp_to_n_matches_subtree[parent] += len(keypoint_matching['matching'])

        parents.sort(key=lambda x: kp_to_n_matches_subtree[x['keypoint']], reverse=True)

        logging.info('Result:')
        for parent in parents:
            kp = parent['keypoint']
            print_kp(kp, len(parent['matching']), None if len(parent_to_kids[kp]) == 0 else kp_to_n_matches_subtree[kp], 0, print_matches, parent)
            for kid in parent_to_kids[kp]:
                kid_kp = kid['keypoint']
                print_kp(kid_kp, len(kid['matching']), None, 1, print_matches, kid)

    @staticmethod
    def print_report(user_report):
        '''
        Prints the user_report to console.
        :param user_report: the user report, returned by method get_full_report in KpAnalysisClient.
        '''
        logging.info('User Report:')
        comments_statuses = user_report['domains_status']
        logging.info('  Comments status by domain (%d domains):' % len(comments_statuses))
        if len(comments_statuses) == 0:
            logging.info('    User has no domains')
        else:
            for status in comments_statuses:
                logging.info(f'    Domain: {status["domain"]}, Domain Params: {status["domain_params"]}, Status: {status["data_status"]}')
        kp_analysis_statuses = user_report['kp_analysis_status']
        logging.info(f'  Key point analysis - jobs status ({len(kp_analysis_statuses)} jobs):')
        if len(kp_analysis_statuses) == 0:
            logging.info('    User has no key point analysis jobs history')
        else:
            for kp_analysis_status in kp_analysis_statuses:
                logging.info(f'    Job: {str(kp_analysis_status)}')

    @staticmethod
    def set_stance_to_result(result, stance):
        for keypoint_matching in result['keypoint_matchings']:
            keypoint_matching['stance'] = stance
        return result

    @staticmethod
    def merge_two_results(result1, result2):
        result = {'keypoint_matchings': result1['keypoint_matchings'] + result2['keypoint_matchings']}
        result['keypoint_matchings'].sort(key=lambda matchings: len(matchings['matching']), reverse=True)
        return result

    @staticmethod
    def write_result_to_csv(result, result_file):
        '''
        Writes the key point analysis result to file.
        Creates two files:
        * matches file: a file with all sentence-key point matches, saved in result_file path.
        * summary file: a summary file with all key points and their aggregated information, saved in result_file path with suffix: _kps_summary.csv.
        :param result: the result, returned by method get_result in KpAnalysisTaskFuture.
        :param result_file: a path to the file that will be created (should have a .csv suffix, otherwise it will be added).
        '''
        def _write_df_to_file(df, file):
            logging.info("Writing dataframe to: " + file)
            file_path = Path(file)
            if not os.path.exists(file_path.parent):
                logging.info('creating directory: %s' % str(file_path.parent))
                os.makedirs(file_path.parent)
            df.to_csv(file, index=False)

        if 'keypoint_matchings' not in result:
            logging.info("No keypoint matchings results")
            return

        if '.csv' not in result_file:
            result_file += '.csv'

        all_sentneces_ids = set([m['comment_id'] + "_" + str(m['sentence_id'])
                                 for keypoint_matching in result['keypoint_matchings'] for m in
                                 keypoint_matching['matching']])
        total_sentences = len(all_sentneces_ids)

        all_comment_ids = set([m["comment_id"] for keypoint_matching in result['keypoint_matchings'] for m in keypoint_matching['matching']])
        total_comments = len(all_comment_ids)
        all_mapped_comment_ids = set([m["comment_id"] for keypoint_matching in result['keypoint_matchings'] if
                                      keypoint_matching['keypoint'] != 'none' for m in keypoint_matching['matching']])

        total_unmapped_comments = total_comments - len(all_mapped_comment_ids)

        summary_rows = []
        matchings_rows = []
        kp_to_parent = {}
        kps_have_stance = False
        sentences_have_stance = False
        stance_keys = []
        for keypoint_matching in result['keypoint_matchings']:
            kp = keypoint_matching['keypoint']
            kp_stance = keypoint_matching.get('stance', None)
            n_sentences = len(keypoint_matching['matching'])
            sentence_coverage = n_sentences / total_sentences if total_sentences > 0 else 0.0
            if kp == "none":
                n_comments = total_unmapped_comments
            else:
                n_comments = len(set([m["comment_id"] for m in keypoint_matching['matching']]))
            comments_coverage = n_comments / total_comments if total_comments > 0 else 0.0

            summary_row = [kp, n_sentences, sentence_coverage, n_comments, comments_coverage]
            if kp_stance is not None:
                kps_have_stance = True
            if kps_have_stance:
                summary_row.append(kp_stance if kp_stance else "")

            kp_to_parent[kp] = keypoint_matching.get("parent", 'root')
            kp_scores = [0, 0, 0]
            for match in keypoint_matching['matching']:
                match_row = [kp, match["sentence_text"], match["score"], match["comment_id"], match["sentence_id"],
                             match["sents_in_comment"], match["span_start"], match["span_end"], match["num_tokens"],
                             match["argument_quality"],  match.get("kp_quality", 0)]

                if match["sentence_text"] == kp:
                    kp_q = match.get("kp_quality", 0)
                    kp_scores = [match["num_tokens"], match["argument_quality"], kp_q]

                if 'stance' in match:
                    stance_dict = match['stance']
                    stance_tups = list(stance_dict.items())
                    stance_keys = [t[0] for t in stance_tups]
                    stance_scores = [t[1] for t in stance_tups]
                    match_row.extend(stance_scores)
                    stance_conf = np.max(stance_scores)
                    selected_stance = stance_keys[np.argmax(stance_scores)]
                    match_row.extend([selected_stance, stance_conf])
                    sentences_have_stance = True
                if kp_stance is not None:
                    match_row.append(kp_stance)
                matchings_rows.append(match_row)

            summary_row.extend(kp_scores)
            summary_rows.append(summary_row)

        summary_rows = sorted(summary_rows, key=lambda x: x[1], reverse=True)
        summary_cols = ["kp", "#sentences", 'sentences_coverage', '#comments', 'comments_coverage']
        if kps_have_stance:
            summary_cols.append('stance')
        summary_cols.extend(["num_tokens", "argument_quality", "kp_quality"])

        summary_df = pd.DataFrame(summary_rows, columns=summary_cols)

        if len(set(kp_to_parent.values())) > 1:
            summary_df.loc[:, "parent"] = summary_df.apply(lambda x: kp_to_parent[x["kp"]], axis=1)
            parent_to_kps = {p: list(filter(lambda x: kp_to_parent[x] == p, kp_to_parent.keys()))
                             for p in set(kp_to_parent.values())}
            parent_to_kps.update({p: [] for p in set(parent_to_kps["root"]).difference(parent_to_kps.keys())})
            kp_to_n_args = dict(zip(summary_df["kp"], summary_df["#sentences"]))
            kp_to_n_args_sub = {kp: np.sum([kp_to_n_args[c_kp] for c_kp in set(parent_to_kps.get(kp, []) + [kp])])
                                for kp in kp_to_parent}
            kp_to_n_args_sub["root"] = np.sum(list(summary_df["#sentences"]))
            summary_df.loc[:, "#sents_in_subtree"] = summary_df.apply(lambda x: kp_to_n_args_sub[x["kp"]], axis=1)

            hierarchy_data = [[p, len(parent_to_kps[p]), kp_to_n_args_sub[p], parent_to_kps[p]] for p in parent_to_kps]
            hierarchy_df = pd.DataFrame(hierarchy_data,
                                        columns=["top_kp", "#level_2_kps", "#sents_in_subtree", "level_2_kps"])
            hierarchy_df.sort_values(by=["#sents_in_subtree"], ascending=False, inplace=True)

            hierarchy_file = result_file.replace(".csv", "_kp_hierarchy.csv")
            _write_df_to_file(hierarchy_df, hierarchy_file)

        summary_file = result_file.replace(".csv", "_kps_summary.csv")
        _write_df_to_file(summary_df, summary_file)

        matchings_cols = ["kp", "sentence_text", "match_score", 'comment_id', 'sentence_id', 'sents_in_comment', 'span_start',
                'span_end', 'num_tokens', 'argument_quality', 'kp_quality']
        if sentences_have_stance:
            matchings_cols.extend([f'{k}_score' for k in stance_keys] + ["selected_stance", "stance_conf"])
        if kps_have_stance:
            matchings_cols.append('kp_stance')
        match_df = pd.DataFrame(matchings_rows, columns=matchings_cols)
        _write_df_to_file(match_df, result_file)

    @staticmethod
    def write_sentences_to_csv(sentences, out_file):
        def get_selected_stance(r):
            stance_dict = dict(r["stance_dict"])
            stance_list = list(stance_dict.items())
            stance_list = sorted(stance_list, reverse=True, key=lambda item: item[1])
            stance, conf = stance_list[0]
            r["stance"] = stance
            r["stance_conf"] = conf
            return r

        if len(sentences) == 0:
            logging.info('there are no sentences, not saving file')
            return

        cols = list(sentences[0].keys())
        rows = [[s[col] for col in cols] for s in sentences]
        df = pd.DataFrame(rows, columns=cols)
        if "stance_dict" in cols:
            df = df.apply(lambda r: get_selected_stance(r), axis=1)
        df.to_csv(out_file, index=False)

    @staticmethod
    def init_logger():
        '''
        Inits the logger for more informative console prints.
        '''
        from logging import getLogger, getLevelName, Formatter, StreamHandler
        log = getLogger()
        log.setLevel(getLevelName('INFO'))
        log_formatter = Formatter("%(asctime)s [%(levelname)s] %(filename)s %(lineno)d: %(message)s")

        console_handler = StreamHandler()
        console_handler.setFormatter(log_formatter)
        log.handlers = []
        log.addHandler(console_handler)

    @staticmethod
    def create_domain_ignore_exists(client, domain, domain_params):
        try:
            client.create_domain(domain, domain_params)
            logging.info(f'domain: {domain} was created')
        except KpaIllegalInputException as e:
            if 'already exist' not in str(e):
                raise e
            logging.info(f'domain: {domain} already exists, domain_params are NOT updated.')

    @staticmethod
    def delete_domain_ignore_doesnt_exist(client, domain):
        try:
            client.delete_domain_cannot_be_undone(domain)
            logging.info(f'domain: {domain} was deleted')
        except KpaIllegalInputException as e:
            if 'doesn\'t have domain' not in str(e):
                raise e
            logging.info(f'domain: {domain} doesn\'t exist.')

    @staticmethod
    def create_graph_data_file_for_ui(result_file, min_n_similar_matches=5, n_matches_samples=20):
        '''
        translates the result file (full result, not the summary) into a json that is loadable in the kpa-key-points-graph-ui
        :param result_file: full results file (with sentence to keypoint mappings).
        :param min_n_similar_matches: minimal number of sentences in the key points intersection for creating an edge (relation between the key points).
        :return: creates a new json file (located near result_file with a new suffix).
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

        def create_dict_to_list(list_tups):
            res = defaultdict(list)
            for x, y in list_tups:
                res[x].append(y)
            return dict(res)

        def read_tups_from_csv(filename):
            logging.info(f'reading file: {filename}')
            input_df = pd.read_csv(filename)
            cols = list(input_df.columns)
            tups = list(input_df.to_records(index=False))
            tups = [tuple([str(v) if str(v) != 'nan' else '' for v in t]) for t in tups]
            return tups, cols

        def read_dicts_from_csv(filename):
            tups, cols = read_tups_from_csv(filename)
            dicts = [{} for _ in tups]
            dicts_tups = list(zip(dicts, tups))
            for i, c in enumerate(cols):
                for d, t in dicts_tups:
                    d[c] = t[i]
            return dicts, cols

        def get_node(i, kp, kp_to_dicts, n_matches_samples):
            return {'id': i,
                    'kp': kp,
                    'n_matches': len(kp_to_dicts[kp]),
                    'matches': [{'sentence_text': d["sentence_text"],
                                 'match_score': float(d["match_score"])}
                                for d in kp_to_dicts[kp][:n_matches_samples]]
                    }

        logging.info(f'Creating key points graph data-file for results file: {result_file}')
        dicts, _ = read_dicts_from_csv(result_file)

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

        graph_data = graph_to_graph_data(graph, n_sentences)

        out_file = result_file.replace('.csv', '_graph_data.json')
        logging.info(f'saving graph in file: {out_file}')
        with open(out_file, 'w') as f:
            json.dump(graph_data, f)
        return out_file

    @staticmethod
    def graph_data_to_hierarchical(result_graph_data_json,
                                   filter_small_nodes=-1,
                                   filter_min_relations=-1,
                                   filter_big_to_small_edges=True,
                                   save_hierarchical_textual_bullets=True,
                                   save_hierarchical_graph_data=True):
        def get_hierarchical_bullets_aux(id_to_kids, id_to_node, id, tab, res):
            msg = f'{"  " * tab} * {id_to_node[id]["data"]["kp"]} ({id_to_node[id]["data"]["n_matches"]} matches)'
            res.append(msg)
            if id in id_to_kids:
                kids = id_to_kids[id]
                kids = sorted(kids, key=lambda n: int(id_to_node[n]['data']['n_matches']), reverse=True)
                for k in kids:
                    get_hierarchical_bullets_aux(id_to_kids, id_to_node, k, tab + 1, res)

        def get_hierarchical_bullets(roots, id_to_kids, id_to_node):
            res = []
            tab = 0
            roots = sorted(roots, key=lambda n: int(id_to_node[n]['data']['n_matches']), reverse=True)
            for root in roots:
                get_hierarchical_bullets_aux(id_to_kids, id_to_node, root, tab, res)
            return res

        print(f'creating hierarchical graph from file: {result_graph_data_json}')
        data = json.load(open(result_graph_data_json))
        nodes = [d for d in data if d['type'] == 'node']
        edges = [d for d in data if d['type'] == 'edge']

        if filter_small_nodes > 0:
            nodes = [n for n in nodes if
                     (round(float(n['data']['n_matches']) / float(n['data']['n_sentences']), 3) >= filter_small_nodes)]
        nodes_ids = [n['data']['id'] for n in nodes]
        id_to_node = {d['data']['id']: d for d in nodes}

        if filter_min_relations > 0:
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
            scores = [source_target_to_score[(id, parent)] if (id, parent) in source_target_to_score else 0.0 for parent in parents]
            max_score_idx = argmax(scores)
            max_parent = parents[max_score_idx]
            id_to_max_parent[id] = max_parent

        edges = [e for e in edges if e['data']['source'] in id_to_max_parent and id_to_max_parent[e['data']['source']] == e['data']['target']]

        if save_hierarchical_textual_bullets:
            id_to_kids = create_dict_to_list([(e['data']['target'], e['data']['source']) for e in edges])
            all_kids = set()
            for id, kids in id_to_kids.items():
                all_kids = all_kids.union(kids)

            root_ids = set(nodes_ids).difference(all_kids)
            res = get_hierarchical_bullets(root_ids, id_to_kids, id_to_node)
            if '_graph_data.json' in result_graph_data_json:
                out_file = result_graph_data_json.replace('_graph_data.json', '_hierarchical.txt')
            else:
                out_file = result_graph_data_json.replace('.json', '_hierarchical.txt')
            logging.info(f'saving graph in file: {out_file}')
            with open(out_file, 'w') as f:
                for line in res:
                    f.write("%s\n" % line)

        if save_hierarchical_graph_data:
            graph_data = nodes + edges
            if '_graph_data.json' in result_graph_data_json:
                out_file = result_graph_data_json.replace('_graph_data.json', '_hierarchical_graph_data.json')
            else:
                out_file = result_graph_data_json.replace('.json', '_hierarchical_graph_data.json')
            logging.info(f'saving graph in file: {out_file}')
            with open(out_file, 'w') as f:
                json.dump(graph_data, f)

    @staticmethod
    def graph_data_to_hierarchical_graph_data(result_graph_data_json,
                                              filter_small_nodes=-1,
                                              filter_min_relations=-1):
        KpAnalysisUtils.graph_data_to_hierarchical(result_graph_data_json,
                                                   filter_small_nodes=filter_small_nodes,
                                                   filter_min_relations=filter_min_relations,
                                                   filter_big_to_small_edges=True,
                                                   save_hierarchical_graph_data=True,
                                                   save_hierarchical_textual_bullets=False)

    @staticmethod
    def graph_data_to_hierarchical_textual_bullets(result_graph_data_json,
                                              filter_small_nodes=-1,
                                              filter_min_relations=0.2):
        KpAnalysisUtils.graph_data_to_hierarchical(result_graph_data_json,
                                                   filter_small_nodes=filter_small_nodes,
                                                   filter_min_relations=filter_min_relations,
                                                   filter_big_to_small_edges=True,
                                                   save_hierarchical_graph_data=False,
                                                   save_hierarchical_textual_bullets=True)




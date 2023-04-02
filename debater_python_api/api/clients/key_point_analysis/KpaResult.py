import ast
import logging
import math
from collections import defaultdict

import pandas as pd
import numpy as np

from debater_python_api.api.clients.key_point_analysis.utils import read_dicts_from_df, create_dict_to_list, \
    write_df_to_file


class KpaResult:
    def __init__(self, result_json, result_df, summary_df, hierarchy_df, name=None):
        self.result_json = result_json
        self.result_df = result_df
        self.summary_df = summary_df
        self.hierarchy_df = hierarchy_df
        self.name = name if name else "kpa_results"

    @staticmethod
    def result_json_to_result_df(result):
        if 'keypoint_matchings' not in result:
            logging.info("No keypoint matchings results")
            return None, None

        matchings_rows = []
        kps_have_stance = False
        sentences_have_stance = False
        stance_keys = []
        for keypoint_matching in result['keypoint_matchings']:
            kp = keypoint_matching['keypoint']
            kp_stance = keypoint_matching.get('stance', None)
            if kp_stance is not None:
                kps_have_stance = True

            for match in keypoint_matching['matching']:
                match_row = [kp, match["sentence_text"], match["score"], match["comment_id"], match["sentence_id"],
                             match["sents_in_comment"], match["span_start"], match["span_end"], match["num_tokens"],
                             match["argument_quality"], match.get("kp_quality", 0)]

                if 'stance' in match:
                    stance_dict = match['stance']
                    stance_tups = list(stance_dict.items())
                    stance_keys = [t[0] for t in stance_tups]
                    stance_scores = [float(t[1]) for t in stance_tups]
                    match_row.extend(stance_scores)
                    stance_conf = np.max(stance_scores)
                    selected_stance = stance_keys[np.argmax(stance_scores)]
                    match_row.extend([selected_stance, stance_conf])
                    sentences_have_stance = True
                if kp_stance is not None:
                    match_row.append(kp_stance)
                matchings_rows.append(match_row)
        matchings_cols = ["kp", "sentence_text", "match_score", 'comment_id', 'sentence_id', 'sents_in_comment',
                          'span_start', 'span_end', 'num_tokens', 'argument_quality', 'kp_quality']
        if sentences_have_stance:
            matchings_cols.extend([f'{k}_score' for k in stance_keys] + ["selected_stance", "stance_conf"])
        if kps_have_stance:
            matchings_cols.append('kp_stance')
        return pd.DataFrame(matchings_rows, columns=matchings_cols)

    @staticmethod
    def result_df_to_summary_df(result_df):
        dicts, _ = read_dicts_from_df(result_df)
        if len(dicts) == 0:
            logging.info("No key points in results")
            return None

        kp_to_dicts = create_dict_to_list([(d['kp'], d) for d in dicts])

        total_sentences = KpaResult.get_number_of_unique_sentences_for_csv_results(result_df)

        all_comment_ids = set([d["comment_id"] for d in dicts])
        total_comments = len(all_comment_ids)
        all_mapped_comment_ids = set([d["comment_id"] for d in dicts if d['kp'] != 'none'])

        total_unmapped_comments = total_comments - len(all_mapped_comment_ids)

        summary_rows = []
        kps_have_stance = False
        for kp, kp_dicts in kp_to_dicts.items():
            n_sentences = len(kp_dicts)
            sentence_coverage = n_sentences / total_sentences if total_sentences > 0 else 0.0
            if kp == "none":
                n_comments = total_unmapped_comments
            else:
                n_comments = len(set([d["comment_id"] for d in kp_dicts]))
            comments_coverage = n_comments / total_comments if total_comments > 0 else 0.0
            summary_row = [kp, n_sentences, sentence_coverage, n_comments, comments_coverage]

            kp_stance = None
            if len(kp_dicts) > 0:
                if 'kp_stance' in kp_dicts[0]:
                    kps_have_stance = True
                    kp_stance = kp_dicts[0]['kp_stance']

            if kps_have_stance:
                summary_row.append(kp_stance if kp_stance else "")

            kp_scores = [0, 0, 0]
            for d in kp_dicts:
                if d["sentence_text"] == kp:
                    kp_scores = [d["num_tokens"], d["argument_quality"], d.get("kp_quality", 0)]

            summary_row.extend(kp_scores)
            summary_rows.append(summary_row)

        summary_rows = sorted(summary_rows, key=lambda x: x[1], reverse=True)
        summary_cols = ["kp", "#sentences", 'sentences_coverage', '#comments', 'comments_coverage']
        if kps_have_stance:
            summary_cols.append('stance')
        summary_cols.extend(["num_tokens", "argument_quality", "kp_quality"])
        return pd.DataFrame(summary_rows, columns=summary_cols)

    @staticmethod
    def update_dataframes_with_hierarchical_results(result_df, summary_df):
        '''
        Updates the summary_df with a summary of the hierarchical data
        and returns the hierarchy_df dataframe with more elaborated hierarchical results.
        '''
        dicts, _ = read_dicts_from_df(result_df)

        hierarchy_df = None
        kp_to_parent = {d['kp']: d.get("parent", 'root') for d in dicts}
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
        return summary_df, hierarchy_df

    @staticmethod
    def result_df_to_result_json(result_df):
        kp_to_matches = defaultdict(list)
        dicts, _ = read_dicts_from_df(result_df)
        for i, match in enumerate(dicts):
            kp = str(match['kp'])
            kp_stance = None
            del match['kp']
            if 'kp_stance' in match:
                kp_stance = match['kp_stance']
                del match['kp_stance']
            for k in match:
                match[k] = str(match[k])
            match['score'] = match['match_score']
            del match['match_score']

            if 'stance' in match:
                if isinstance(match['stance'], str):
                    match['stance'] = ast.literal_eval(match['stance'])
            else:
                stance = {}
                if 'pos_score' in match:
                    stance['pos'] = float(match['pos_score'])
                if 'neg_score' in match:
                    stance['neg'] = float(match['neg_score'])
                if 'sug_score' in match:
                    stance['sug'] = float(match['sug_score'])
                if 'neut_score' in match:
                    stance['neut'] = float(match['neut_score'])
                if len(stance) > 0:
                    match['stance'] = stance

            kp_to_matches[(kp, kp_stance)].append(match)
        result_json = {'keypoint_matchings': []}
        for (kp, stance), matchings in kp_to_matches.items():
            km = {'keypoint': kp, 'matching': matchings}
            if stance is not None:
                km['stance'] = stance
            result_json['keypoint_matchings'].append(km)
        return result_json

    @staticmethod
    def create_from_result_json(result_json, name=None):
        if 'keypoint_matchings' not in result_json:
            logging.info("No keypoint matchings results")
            return KpaResult(result_json, None, None, None, name)
        else:
            result_df = KpaResult.result_json_to_result_df(result_json)
            summary_df = KpaResult.result_df_to_summary_df(result_df)
            summary_df, hierarchy_df = KpaResult.update_dataframes_with_hierarchical_results(result_df, summary_df)
            return KpaResult(result_json, result_df, summary_df, hierarchy_df, name)

    @staticmethod
    def create_from_result_csv(result_csv, name=None):
        result_df = pd.read_csv(result_csv)
        result_json = KpaResult.result_df_to_result_json(result_df)
        summary_df = KpaResult.result_df_to_summary_df(result_df)
        summary_df, hierarchy_df = KpaResult.update_dataframes_with_hierarchical_results(result_df, summary_df)
        return KpaResult(result_json, result_df, summary_df, hierarchy_df, name)

    def write_to_file(self, result_file, also_hierarchy=True):
        if '.csv' not in result_file:
            result_file += '.csv'

        if self.result_df is not None:
            write_df_to_file(self.result_df, result_file)

        if self.summary_df is not None:
            summary_file = result_file.replace(".csv", "_kps_summary.csv")
            write_df_to_file(self.summary_df, summary_file)

        if also_hierarchy and self.hierarchy_df is not None:
            hierarchy_file = result_file.replace(".csv", "_kps_hierarchy.csv")
            write_df_to_file(self.hierarchy_df, hierarchy_file)

    def print_result(self, n_sentences_per_kp, title):
        '''
        Prints the key point analysis result to console.
        :param n_sentences_per_kp: number of top matched sentences to display for each key point
        :param title: title to print for the analysis
        '''
        def split_sentence_to_lines(sentence, max_len=90):
            if len(sentence) <= max_len:
                return ['- ' + sentence]

            lines = []
            line = None
            tokens = sentence.split(' ')
            for token in tokens:
                if line is None:
                    line = '- ' + token
                else:
                    if len(line + ' ' + token) <= max_len:
                        line += ' ' + token
                    else:
                        lines.append(line)
                        line = '  ' + token
            if line is not None:
                lines.append(line)
            return lines

        def split_sentences_to_lines(sentences, n_tabs):
            lines = []
            for sentence in sentences:
                lines.extend(split_sentence_to_lines(sentence))
            return [('\t' * n_tabs) + line for line in lines]

        def print_kp(kp, stance, n_matches, n_matches_subtree, depth, keypoint_matching, n_sentences_per_kp):
            has_n_matches_subtree = n_matches_subtree is not None
            print('%s%d%s - %s%s' % (('\t' * depth), n_matches_subtree if has_n_matches_subtree else n_matches,
                                     (' - %d' % n_matches) if has_n_matches_subtree else '', kp,
                                     '' if stance is None else ' - ' + stance))
            sentences = [match['sentence_text'] for match in keypoint_matching['matching']]
            sentences = sentences[1:(n_sentences_per_kp + 1)]  # first sentence is the kp itself
            lines = split_sentences_to_lines(sentences, depth)
            for line in lines:
                print('\t%s' % line)

        kp_to_n_matches_subtree = defaultdict(int)
        parents = list()
        parent_to_kids = defaultdict(list)
        for keypoint_matching in self.result_json['keypoint_matchings']:
            kp = keypoint_matching['keypoint']
            kp_to_n_matches_subtree[kp] += len(keypoint_matching['matching'])
            parent = keypoint_matching.get("parent", None)
            if parent is None or parent == 'root':
                parents.append(keypoint_matching)
            else:
                parent_to_kids[parent].append(keypoint_matching)
                kp_to_n_matches_subtree[parent] += len(keypoint_matching['matching'])

        parents.sort(key=lambda x: kp_to_n_matches_subtree[x['keypoint']], reverse=True)

        n_total_sentences = self.get_number_of_unique_sentences(include_unmatched=True)
        n_matched_sentences = self.get_number_of_unique_sentences(include_unmatched=False)

        print(title + ' coverage: %.2f' % (float(n_matched_sentences) / float(n_total_sentences) * 100.0))
        print(title + ' key points:')
        for parent in parents:
            kp = parent['keypoint']
            stance = None if 'stance' not in parent else parent['stance']
            if kp == 'none':
                continue
            print_kp(kp, stance, len(parent['matching']),
                     None if len(parent_to_kids[kp]) == 0 else kp_to_n_matches_subtree[kp], 0, parent, n_sentences_per_kp)
            for kid in parent_to_kids[kp]:
                kid_kp = kid['keypoint']
                kid_stance = None if 'stance' not in kid else kid['stance']
                print_kp(kid_kp, kid_stance, len(kid['matching']), None, 1, kid, n_sentences_per_kp)

    @staticmethod
    def get_unique_sent_id(sentence_dict):
        return f"{sentence_dict['comment_id']}_{sentence_dict['sentence_id']}"

    def get_number_of_unique_sentences(self, include_unmatched=True):
        return KpaResult.get_number_of_unique_sentences_for_json_results(self.result_json, include_unmatched)

    @staticmethod
    def get_number_of_unique_sentences_for_json_results(result_json, include_unmatched=True):
        total_sentences = set()
        for i, keypoint_matching in enumerate(result_json['keypoint_matchings']):
            matches = keypoint_matching['matching']
            matching_sents_ids = set([KpaResult.get_unique_sent_id(d) for d in matches])
            if keypoint_matching['keypoint'] != 'none' or include_unmatched:
                total_sentences = total_sentences.union(matching_sents_ids)
        return len(total_sentences)

    @staticmethod
    def get_number_of_unique_sentences_for_csv_results(result_df, include_unmatched=True):
        result_json = KpaResult.result_df_to_result_json(result_df)
        return KpaResult.get_number_of_unique_sentences_for_json_results(result_json, include_unmatched)

    def get_kp_to_n_matched_sentences(self, include_none=True):
        kps_n_args = {kp['keypoint']: len(kp['matching']) for kp in self.result_json['keypoint_matchings']
                       if kp['keypoint'] != 'none' or include_none}
        return kps_n_args

    def compare_with_other(self, other_results):
        title_1 = self.name
        result_1_total_sentences = self.get_number_of_unique_sentences(include_unmatched=True)
        kps1_n_args = self.get_kp_to_n_matched_sentences(include_none=False)
        kps1 = set(kps1_n_args.keys())

        title_2 = other_results.name
        result_2_total_sentences = other_results.get_number_of_unique_sentences(include_unmatched=True)
        kps2_n_args = other_results.get_kp_to_n_matched_sentences(include_none=False)
        kps2 = set(kps2_n_args.keys())

        kps_in_both = kps1.intersection(kps2)
        kps_in_both = sorted(list(kps_in_both), key=lambda kp: kps1_n_args[kp], reverse=True)
        cols = ['key point', f'{title_1}_n_sents', f'{title_1}_percent', f'{title_2}_n_sents', f'{title_2}_percent',
                'change_n_sents', 'change_percent']
        rows = []
        for kp in kps_in_both:
            sents1 = kps1_n_args[kp]
            sents2 = kps2_n_args[kp]
            percent1 = (sents1 / result_1_total_sentences) * 100.0
            percent2 = (sents2 / result_2_total_sentences) * 100.0
            rows.append([kp, sents1, f'{percent1:.2f}%', sents2, f'{percent2:.2f}%', str(math.floor((sents2 - sents1))),
                         f'{(percent2 - percent1):.2f}%'])
        kps1_not_in_2 = kps1 - kps2
        kps1_not_in_2 = sorted(list(kps1_not_in_2), key=lambda kp: kps1_n_args[kp], reverse=True)
        for kp in kps1_not_in_2:
            sents1 = kps1_n_args[kp]
            percent1 = (sents1 / result_1_total_sentences) * 100.0
            rows.append([kp, sents1, f'{percent1:.2f}%', '---', '---', '---', '---'])
        kps2_not_in_1 = kps2 - kps1
        kps2_not_in_1 = sorted(list(kps2_not_in_1), key=lambda kp: kps2_n_args[kp], reverse=True)
        for kp in kps2_not_in_1:
            sents2 = kps2_n_args[kp]
            percent2 = (sents2 / result_2_total_sentences) * 100.0
            rows.append([kp, '---', '---', sents2, f'{percent2:.2f}%', '---', '---'])
        comparison_df = pd.DataFrame(rows, columns=cols)
        return comparison_df

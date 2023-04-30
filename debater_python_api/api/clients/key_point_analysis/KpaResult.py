import ast
import json
import logging
import math
from collections import defaultdict

import pandas as pd
import numpy as np

from debater_python_api.api.clients.key_point_analysis.utils import read_dicts_from_df, create_dict_to_list, \
    write_df_to_file, get_unique_sent_id
CURR_RESULTS_VERSION = "2.0"

class KpaResult:
    def __init__(self, result_json, version=CURR_RESULTS_VERSION):
        self.result_json = result_json
        self.result_df = self.create_result_df()
        self.summary_df = self.result_df_to_summary_df()
        self.version = version

    def save(self, json_file):
        with open(json_file, 'w') as f:
            json.dump(self.result_json, f)

    @staticmethod
    def load(json_file):
        with open(json_file, 'r') as f:
            json_res = json.load(f)
            return KpaResult.create_from_result_json(json_res)

    def create_result_df(self):

        sentences_data = self.result_json["sentences_data"]
        matchings_rows = []
        kps_have_stance = False
        sentences_have_stance = False
        stance_keys = []
        for keypoint_matching in self.result_json['keypoint_matchings']:
            kp = keypoint_matching['keypoint']
            kp_quality = keypoint_matching.get('kp_quality', None)
            kp_stance = keypoint_matching.get('stance', None)
            if kp_stance is not None:
                kps_have_stance = True

            for match in keypoint_matching['matching']:
                score = match["score"]
                sent_identifier = match["sent_identifier"]
                sent_data = sentences_data[sent_identifier]
                match_row = [kp, sent_data["sentence_text"], score, sent_data["comment_id"], sent_data["sentence_id"],
                             sent_data["sents_in_comment"], sent_data["span_start"], sent_data["span_end"], sent_data["num_tokens"],
                             sent_data["argument_quality"], kp_quality]

                if 'stance' in sent_data:
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

    def result_df_to_summary_df(self):
        dicts, _ = read_dicts_from_df(self.result_df)
        if len(dicts) == 0:
            logging.info("No key points in results")
            return None

        kp_to_dicts = create_dict_to_list([(d['kp'], d) for d in dicts])

        total_sentences = self.get_number_of_unique_sentences()

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
    def result_df_to_result_json(result_df, new_version = True):
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
        if new_version:
            result_json = KpaResult.create_from_result_json(result_json)
        return result_json

    @staticmethod
    def create_from_result_json(result_json):
        if 'keypoint_matchings' not in result_json:
            logging.error("Faulty results json provided: does not contain 'keypoint_matchings'. returning empty results")
            return None # TODO raise Exception??
        else:
            if "version" in result_json:
                return KpaResult(result_json)
            else:
                result_json_v2 = KpaResult.convert_to_v2(result_json)
                return KpaResult(result_json_v2)

    @staticmethod
    def create_from_result_csv(result_csv):
        result_df = pd.read_csv(result_csv)
        result_json = KpaResult.result_df_to_result_json(result_df)
        return KpaResult(result_json)

    @staticmethod
    def convert_to_v2(result_json):
        sentences_data = {}
        new_matchings = []
        for keypoint_matching in result_json['keypoint_matchings']:
            kp = keypoint_matching['keypoint']
            kp_stance = keypoint_matching.get('stance', None)
            new_kp_matching = {'keypoint': kp, "matching": []}
            if kp_stance is not None:
                new_kp_matching["stance"] = kp_stance

            for match in keypoint_matching['matching']:
                if "kp_quality" not in new_kp_matching:
                    kpq = match.get("kp_quality", 0)
                    new_kp_matching["kp_quality"] = kpq

                sent_identifier = get_unique_sent_id(match)
                new_kp_matching["matching"].append({"sent_identifier":sent_identifier, "score":match["score"]})

                if sent_identifier not in sentences_data:
                    sentences_data[sent_identifier] = {c:match[c] for c in ["sentence_text","comment_id","sentence_id",
                                            "sents_in_comment","span_start","span_end","num_tokens","argument_quality",
                                            ]}
                if "stance" in match:
                    sentences_data[sent_identifier]["stance"] = dict(match["stance"])

            new_matchings.append(new_kp_matching)
        return {"keypoint_matchings":new_matchings, "sentences_data":sentences_data, "version":CURR_RESULTS_VERSION}


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

    def get_number_of_unique_sentences(self, include_unmatched=True):
        total_sentences = set()
        for i, keypoint_matching in enumerate(self.result_json['keypoint_matchings']):
            matches = keypoint_matching['matching']
            matching_sents_ids = set([get_unique_sent_id(d) for d in matches])
            if keypoint_matching['keypoint'] != 'none' or include_unmatched:
                total_sentences = total_sentences.union(matching_sents_ids)
        return len(total_sentences)

    def get_kp_to_n_matched_sentences(self, include_none=True):
        kps_n_args = {kp['keypoint']: len(kp['matching']) for kp in self.result_json['keypoint_matchings']
                       if kp['keypoint'] != 'none' or include_none}
        return kps_n_args

    def compare_with_other(self, other_results, this_title, other_title):
        result_1_total_sentences = self.get_number_of_unique_sentences(include_unmatched=True)
        kps1_n_args = self.get_kp_to_n_matched_sentences(include_none=False)
        kps1 = set(kps1_n_args.keys())

        result_2_total_sentences = other_results.get_number_of_unique_sentences(include_unmatched=True)
        kps2_n_args = other_results.get_kp_to_n_matched_sentences(include_none=False)
        kps2 = set(kps2_n_args.keys())

        kps_in_both = kps1.intersection(kps2)
        kps_in_both = sorted(list(kps_in_both), key=lambda kp: kps1_n_args[kp], reverse=True)
        cols = ['key point', f'{this_title}_n_sents', f'{this_title}_percent', f'{other_title}_n_sents', f'{other_title}_percent',
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

    def merge_with_other(self, other_results):
        combined_results_json = {'keypoint_matchings': self.result_json['keypoint_matchings'] + other_results.result_json['keypoint_matchings']}
        combined_results_json['keypoint_matchings'].sort(key=lambda matchings: len(matchings['matching']), reverse=True)
        return KpaResult.create_from_result_json(result_json = combined_results_json)

    def set_stance_to_result(self, stance):
        for keypoint_matching in self.result_json['keypoint_matchings']:
            keypoint_matching['stance'] = stance
        self.result_df["stance"] = stance

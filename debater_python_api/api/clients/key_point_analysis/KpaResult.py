import ast
import json
import logging
import math
from collections import defaultdict

import pandas as pd
import numpy as np
from keypoint_analysis.km_utils import init_logger

from KpaExceptions import KpaIllegalInputException
from debater_python_api.api.clients.key_point_analysis.utils import read_dicts_from_df, create_dict_to_list, \
    write_df_to_file, get_unique_sent_id, filter_dict_by_keys
from docx_generator import save_hierarchical_graph_data_to_docx
from graph_generator import create_graph_data, graph_data_to_hierarchical_graph_data, \
    get_hierarchical_graph_from_tree_and_subset_results, get_hierarchical_kps_data
import os
CURR_RESULTS_VERSION = "2.0"


class KpaResult:
    """
    Class to hold and process the results of a KPA job.
    """
    def __init__(self, result_json, filter_min_relations=0.4):
        """
        :param result_json: the result json returned from the key point client using get_results.
        """
        self.result_json = result_json
        self.result_df = self.create_result_df()
        self.version = CURR_RESULTS_VERSION
        self.stances = set(result_json["job_metadata"]["per_stance"].keys()).difference("no-stance")
        self.filter_min_relations_for_text = filter_min_relations
        self.kp_id_to_hierarchical_data = self.get_kp_id_to_hierarchical_data()
        self.summary_df = self.result_df_to_summary_df()

    @staticmethod
    def get_stance_from_server_stances(server_stances):

        if not server_stances or len(server_stances) == 0:
            return "no-stance"
        if server_stances == ["pos"]:
            return "pro"
        if set(server_stances) == set(["neg", "sug"]):
            return "con"

        raise KpaIllegalInputException(f'Unsupported stances {server_stances}')

    def get_matadata(self):
        return self.result_json["job_metadata"]

    def get_domain(self):
        return self.result_json["job_metadata"]["general"]["domain"]

    def get_job_ids(self):
        per_stance_metadata = self.result_json["job_metadata"]["per_stance"]
        return set(m["job_id"] for m in per_stance_metadata.values())

    def save(self, json_file):
        """
        Save to current results as a json file
        :param json_file: path to the json file to hold the results
        """
        logging.info(f"Writing results to: {json_file}")
        with open(json_file, 'w') as f:
            json.dump(self.result_json, f)

    @staticmethod
    def load(json_file):
        """
        Load results from a json file
        :param json_file: the file to load the results from, obtained using the save() method.
        :return: KpaResult that wraps the results.
        """
        logging.info(f"Loading results from: {json_file}")
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
            if kp == "none":
                continue
            kp_quality = keypoint_matching.get('kp_quality', None)
            kp_stance = keypoint_matching.get('stance', None)
            if kp_stance is not None:
                kps_have_stance = True

            for match in keypoint_matching['matching']:
                score = match["score"]
                comment_id = str(match["comment_id"])
                sent_id_in_comment = int(match["sentence_id"])
                comment_data = sentences_data[comment_id]
                sent_data = comment_data["sentences"][sent_id_in_comment]

                match_row = [kp, sent_data["sentence_text"], score, comment_id, sent_id_in_comment,
                             comment_data["sents_in_comment"], sent_data["span_start"], sent_data["span_end"], sent_data["num_tokens"],
                             sent_data["argument_quality"], kp_quality]

                if 'stance' in sent_data:
                    stance_dict = sent_data['stance']
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

        kp_to_id = {self.kp_id_to_hierarchical_data[id]["kp"]:id for id in self.kp_id_to_hierarchical_data }
        dicts, _ = read_dicts_from_df(self.result_df)

        kp_to_dicts = create_dict_to_list([(d['kp'], d) for d in dicts])

        general_metadata = self.result_json["job_metadata"]["general"]
        n_total_sentences = general_metadata["n_sentences"]
        all_mapped_sentences = set([get_unique_sent_id(d) for d in dicts if d['kp'] != 'none'])
        n_unmapped_sentences = n_total_sentences - len(all_mapped_sentences)

        n_total_comments = general_metadata["n_comments"]
        all_mapped_comment_ids = set([d["comment_id"] for d in dicts if d['kp'] != 'none'])
        n_unmapped_comments = n_total_comments - len(all_mapped_comment_ids)

        summary_rows = []
        summary_cols = ["kp", '#comments', 'comments_coverage', "#sentences", 'sentences_coverage', "parent", "n_comments_subtree","stance"]

        none_row = ["none", n_unmapped_comments, n_unmapped_comments/n_total_comments, n_unmapped_sentences, n_unmapped_sentences/n_total_sentences,
                    "-","-", "-"]
        summary_rows.append(none_row)

        for kp, kp_dicts in kp_to_dicts.items():
            if kp == "none":
                continue

            n_sentences = len(kp_dicts)
            sentence_coverage = n_sentences / n_total_sentences if n_total_sentences > 0 else 0.0
            n_comments = len(set([d["comment_id"] for d in kp_dicts]))
            comments_coverage = n_comments / n_total_comments if n_total_comments > 0 else 0.0

            kp_to_data = self.kp_id_to_hierarchical_data[kp_to_id[kp]]
            parent = kp_to_data['parent']
            n_comments_in_subtree = kp_to_data['n_matching_comments_in_subtree']
            kp_stance = kp_to_data.get('kp_stance', "")

            summary_row = [kp, n_comments, comments_coverage, n_sentences, sentence_coverage,
                           parent, n_comments_in_subtree, kp_stance]
            summary_rows.append(summary_row)

        summary_rows = sorted(summary_rows, key=lambda x: x[1], reverse=True)
        for stance in self.stances:
            stance_data = self.result_json["job_metadata"]["per_stance"][stance]
            stance_n_comments = stance_data["n_comments_stance"]
            stance_comment_coverage = stance_n_comments/n_total_comments
            stance_n_sentences = stance_data["n_sentences_stance"]
            stance_sentence_coverage = stance_n_sentences / n_total_sentences
            stance_row = [stance, stance_n_comments, stance_comment_coverage, stance_n_sentences, stance_sentence_coverage,
                          "","",stance]
            summary_rows.append(stance_row)
        total_row = ["total",n_total_comments, 1, n_total_sentences, 1, "","",""]
        summary_rows.append(total_row)

        return pd.DataFrame(summary_rows, columns=summary_cols)

    @staticmethod
    def create_from_result_json(result_json, filter_min_relations_for_text=0.4):
        """
        Create KpaResults from results_json
        :param result_json: the json object obtained from the client via "get_results"
        :return KpaResult object
        """
        if 'keypoint_matchings' not in result_json:
            raise KpaIllegalInputException("Faulty results json provided: does not contain 'keypoint_matchings'. returning empty results")
        else:
            try:
                version = result_json.get("version", "1.0")
                if version != CURR_RESULTS_VERSION:
                    result_json = KpaResult.convert_to_new_version(result_json, version, CURR_RESULTS_VERSION)
                return KpaResult(result_json, filter_min_relations_for_text)
            except Exception as e:
                logging.error("Could not create KpaResults from json.")
                raise e

    @staticmethod
    def convert_to_new_version(result_json, old_version, new_version):
        if old_version == "1.0" and new_version == "2.0":
            return KpaResult.convert_v1_to_v2(result_json)

        #can create more conversion methods for future json_result versions...
        raise KpaIllegalInputException(f"Unsupported results version old: {old_version}, new: {new_version}. "
                                       f"Supported: old = 1.0, new = 2.0")

    @staticmethod
    # If result_json is of the old version, convert to version 2.0
    # 1.0 - received from the server, must have only one stance (merging pro_con is only on v2)
    def convert_v1_to_v2(result_json):
        metadata = result_json["job_metadata"]
        kps_stance = KpaResult.get_stance_from_server_stances(metadata["stances"])
        sentences_data = {}
        new_matchings = []
        for keypoint_matching in result_json['keypoint_matchings']:
            kp = keypoint_matching['keypoint']

            new_kp_matching = {'keypoint': kp, "matching": []}
            if kps_stance != "no-stance":
                new_kp_matching["stance"] = kps_stance

            for match in keypoint_matching['matching']:
                if "kp_quality" not in new_kp_matching:
                    kpq = match.get("kp_quality", 0)
                    new_kp_matching["kp_quality"] = kpq

                #sent_identifier = get_unique_sent_id(match)
                comment_id = match["comment_id"]
                sent_id_in_comment = int(match["sentence_id"])

                #new_kp_matching["matching"].append({"sent_identifier":sent_identifier, "score":match["score"]})
                new_kp_matching["matching"].append({"comment_id":comment_id, "sentence_id":int(sent_id_in_comment), "score":match["score"]})
                if comment_id not in sentences_data:
                    sentences_data[comment_id] = {"sents_in_comment":match["sents_in_comment"], "sentences":{}}

                if sent_id_in_comment not in sentences_data[comment_id]["sentences"]:
                    sent_data = {c:match[c] for c in ["sentence_text", "span_start","span_end","num_tokens","argument_quality"]}
                    if "stance" in match:
                        sent_data["stance"] = dict(match["stance"])
                    sentences_data[comment_id]["sentences"][sent_id_in_comment] = sent_data.copy()

            new_matchings.append(new_kp_matching)


        per_stance_dict = {kps_stance: filter_dict_by_keys(metadata, ['description', 'run_params', 'job_id',
                                                                          'n_sentences_stance', 'n_comments_stance'])}

        metadata_v2 = {"general": filter_dict_by_keys(metadata, ['domain', 'user_id', 'n_sentences',
                                                                 'n_sentences_unfiltered', 'n_comments',
                                                                 'n_comments_unfiltered']),
                       "per_stance": per_stance_dict}
        return {"keypoint_matchings": new_matchings, "sentences_data": sentences_data, "version": CURR_RESULTS_VERSION,
                "job_metadata": metadata_v2}

    def generate_docx_report(self, output_dir, result_name,
                             n_matches_in_docx=50,
                             include_match_score_in_docx=False,
                             min_n_matches_in_docx=5,
                             kp_id_to_hierarchical_data=None):
        """
        creates <output_dir>/<result_name>_hierarchical.docx: This Microsoft Word document shows the key point hierarchy and matching sentences
        as a user-friendly report.
        :param output_dir: path to output directory
        :param result_name: name of the results to appear in the output files.
        :param n_matches_in_docx: number of top matches to write in the textual summary (docx file). Pass None for all matches.
        :param include_match_score_in_docx: when set to true, the match score between the sentence and the key point is added.
        :param min_n_matches_in_docx: remove key points with less than min_n_matches_in_docx matching sentences.
        :param kp_id_to_hierarchical_data: optional, should be set to None.
        """
        if not kp_id_to_hierarchical_data:
            kp_id_to_hierarchical_data = self.kp_id_to_hierarchical_data
        docx_file = os.path.join(output_dir, f'{result_name}_hierarchical.docx')
        meta_data = self.result_json["job_metadata"]
        save_hierarchical_graph_data_to_docx(full_result_df=self.result_df, kp_id_to_data=kp_id_to_hierarchical_data,
                                             result_filename=docx_file, meta_data=meta_data, n_matches=n_matches_in_docx,
                                             include_match_score=include_match_score_in_docx,
                                             min_n_matches=min_n_matches_in_docx)

    def generate_docx_report_using_given_tree(self, full_results,
                                              output_dir, result_name, n_top_matches_in_graph=20,
                                              n_matches_in_docx=50, include_match_score_in_docx=False,
                                              min_n_matches_in_docx=5):
        '''
        Create hierarchical result for kpa_result, using a precalculated hierarchical results.
        This is useful when we first create results using the whole data, and then want to calculate the
        hierarchical result of its subset while considering the already existing key points and hierarchy generated over the whole data.
        For example, when we have a large survey, we can first run over the entire data to create a hierarchical
        representation of the full results. Then when we want to evaluate a subset of the survey we can run over a subset of
        the survey using the same key points precomputed in the full survey. Then we create its hierarchical
        representation using the hierarchy of the full survey.
        :param full_results: KpaResult over the full data.
        :param output_dir: path to output directory
        :param result_name: name of the results to appear in the output files.
        :param n_top_matches_in_graph : optional, number of top matches to add to the graph_data file.
        :param n_matches_in_docx: optional, number of top matches to write in the textual summary (docx file). Pass None for all matches.
        :param include_match_score_in_docx: optional, when set to true, the match score between the sentence and the key point is added.
        :param min_n_matches_in_docx: optional, remove key points with less than min_n_matches_in_docx matching sentences.
        '''
        graph_full_data = create_graph_data(full_results.result_df, full_results.get_number_of_unique_sentences())
        hierarchical_graph_full_data = graph_data_to_hierarchical_graph_data(graph_data=graph_full_data)

        new_hierarchical_graph_data = get_hierarchical_graph_from_tree_and_subset_results(
            hierarchical_graph_full_data,
            self.result_df, self.filter_min_relations_for_text, n_top_matches_in_graph)

        new_kp_id_to_hierarchical_data = get_hierarchical_kps_data(self.result_df, new_hierarchical_graph_data, self.filter_min_relations_for_text)
        self.generate_docx_report(output_dir, result_name,
                            n_matches_in_docx=n_matches_in_docx,
                            include_match_score_in_docx=include_match_score_in_docx,
                            min_n_matches_in_docx=min_n_matches_in_docx,
                            kp_id_to_hierarchical_data = new_kp_id_to_hierarchical_data)

    def get_kp_id_to_hierarchical_data(self):
        graph_data = create_graph_data(self.result_df, n_sentences=self.get_number_of_unique_sentences())
        hierarchical_graph_data = graph_data_to_hierarchical_graph_data(graph_data=graph_data)
        return get_hierarchical_kps_data(self.result_df, hierarchical_graph_data, self.filter_min_relations_for_text)

    def export_to_all_outputs(self, output_dir, result_name,
                              n_matches_in_docx=50,
                              include_match_score_in_docx=False,
                              min_n_matches_in_docx=5
                              ):
        """
        Generates all the kpa available output types.
        :param output_dir: path to output directory
        :param result_name: name of the results to appear in the output files.
        :param n_matches_in_docx: optional, number of top matches to write in the textual summary (docx file). Pass None for all matches.
        :param include_match_score_in_docx: optional, when set to true, the match score between the sentence and the key point is added.
        :param min_n_matches_in_docx: optional, remove key points with less than min_n_matches_in_docx matching sentences.
        Creates 3 outout files:
             * <ouput_dir>/<result_name>.csv : full results as csv .
             * <ouput_dir>/<result_name>_kps_summary.csv : summary results as csv.
             *  <result_file>_hierarchical.docx: This Microsoft Word document shows the key point hierarchy and matching sentences
            as a user-friendly report.
        """

        result_file = os.path.join(output_dir, result_name+".csv")
        write_df_to_file(self.result_df, result_file)

        summary_file = result_file.replace(".csv", "_kps_summary.csv")
        write_df_to_file(self.summary_df, summary_file)

        self.generate_docx_report(output_dir, result_name,
                                  n_matches_in_docx=n_matches_in_docx,
                                  include_match_score_in_docx=include_match_score_in_docx,
                                  min_n_matches_in_docx=min_n_matches_in_docx)

    def print_result(self, n_sentences_per_kp, title, n_top_kps = None):
        '''
        Prints the key point analysis result to console.
        :param n_sentences_per_kp: number of top matched sentences to display for each key point
        :param title: title to print for the analysis
        :param n_top_kps: maximal number of kps to display.
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

        def print_kp(kp, stance, keypoint_matching, sentences_data, n_sentences_per_kp): #TODO!!! N_MATCHES INCLUDES KP ITSELF, PRINTS 1 LESS...
            print('%d - %s%s' % (len(keypoint_matching), kp, '' if stance is None else ' - ' + stance))
            sentences = []
            for match in keypoint_matching:
                comment_id = str(match["comment_id"])
                sent_id_in_comment = int(match["sentence_id"])
                sent = sentences_data[comment_id]['sentences'][sent_id_in_comment]["sentence_text"]
                sentences.append(sent)
            sentences = sentences[1:(n_sentences_per_kp + 1)]  # first sentence is the kp itself
            lines = split_sentences_to_lines(sentences, 0)
            for line in lines:
                print('\t%s' % line)

        keypoint_matchings = self.result_json["keypoint_matchings"]
        sentences_data = self.result_json["sentences_data"]

        n_total_sentences = self.get_number_of_unique_sentences(include_unmatched=True)
        n_matched_sentences = self.get_number_of_unique_sentences(include_unmatched=False)

        print(title + ' coverage: %.2f' % (float(n_matched_sentences) / float(n_total_sentences) * 100.0))
        print(title + ' key points:')
        n_top_kps = n_top_kps if n_top_kps else len(keypoint_matchings)
        for keypoint_matching in keypoint_matchings[:n_top_kps]:
            kp = keypoint_matching['keypoint']
            stance = None if 'stance' not in keypoint_matching else keypoint_matching['stance']
            if kp == 'none':
                continue
            print_kp(kp, stance, keypoint_matching['matching'], sentences_data, n_sentences_per_kp)

    def get_number_of_unique_sentences(self, include_unmatched=True):
        if include_unmatched:
            return self.result_json["job_metadata"]["general"]["n_sentences"]
        else:
            total_sentences = set()
            for i, keypoint_matching in enumerate(self.result_json['keypoint_matchings']):
                matching_sents_ids = set([get_unique_sent_id(d) for d in keypoint_matching['matching']])
                if keypoint_matching['keypoint'] != 'none':
                    total_sentences = total_sentences.union(matching_sents_ids)
            return len(total_sentences)

    def get_number_of_unique_comments(self, include_unmatched=True):
        if include_unmatched:
            return self.result_json["job_metadata"]["general"]["n_comments"]
        else:
            total_comments = set()
            for i, keypoint_matching in enumerate(self.result_json['keypoint_matchings']):
                if keypoint_matching['keypoint'] != 'none':
                    matching_sents_ids = set([d["comment_id"] for d in keypoint_matching['matching']])
                    total_comments = total_comments.union(matching_sents_ids)
            return len(total_comments)


    def get_kp_to_n_matched_comments(self, comments_subset = None):
        kp_n_comments = {}
        for keypoint_matching in self.result_json['keypoint_matchings']:
            kp = keypoint_matching["keypoint"]
            matching_comments_set = set(match["comment_id"] for match in keypoint_matching["matching"])

            if comments_subset:
                matching_comments_set = set(filter(lambda x:x in comments_subset, matching_comments_set))

            n_matches = len(matching_comments_set)
            kp_n_comments[kp] = n_matches
        return kp_n_comments

    def compare_with_comment_subsets(self, comments_subsets_dicts):
        """
        :param subset_dicts: dict from subest_name to set of comment ids
        """
        results_to_total_comments = {"full": self.get_number_of_unique_comments()}
        results_to_total_comments.update({title: len(comments_subsets_dicts[title]) for title in comments_subsets_dicts})

        titles = ["full"] + sorted(comments_subsets_dicts.keys(), key=lambda x: results_to_total_comments[x], reverse=True)

        results_to_kp_to_n_comments = {"full": self.get_kp_to_n_matched_comments()}
        results_to_kp_to_n_comments.update(
            {title: self.get_kp_to_n_matched_comments(comments_subset=comment_ids) for title, comment_ids in comments_subsets_dicts.items()})
        return self._get_comparison_df(results_to_kp_to_n_comments, results_to_total_comments, titles)

    def compare_with_other_results(self, this_title, other_results_dict):
        """
        Compare this result with other results.
        :param this_title: title to be associated with this result.
        :param other_results_dict: dictionary of result_title as keys and KpaResults as values.
        :return dataframe containing the number and percentage of the comments matched to each key point in each result,
        and the change percentage if comparing to a single result.
        """
        results_to_total_comments = {this_title:self.get_number_of_unique_comments()}
        results_to_total_comments.update({title:result.get_number_of_unique_comments() for title,result in other_results_dict.items()})
        titles = [this_title] + sorted(other_results_dict.keys(), key=lambda x:results_to_total_comments[x], reverse=True)

        results_to_kp_to_n_comments = {this_title:self.get_kp_to_n_matched_comments()}
        results_to_kp_to_n_comments.update({title:result.get_kp_to_n_matched_comments()  for title,result in other_results_dict.items()})
        return self._get_comparison_df(results_to_kp_to_n_comments, results_to_total_comments, titles)

    def _get_comparison_df(self, results_to_kp_to_n_comments, results_to_total_comments, titles):
        ordered_kps = []
        cols = ['key point']
        for title in titles:
            n_comments_per_kp_per_title = results_to_kp_to_n_comments[title]
            new_kps = set(n_comments_per_kp_per_title.keys()).difference(set(ordered_kps))
            new_kps = sorted(new_kps, key= lambda x:n_comments_per_kp_per_title[x] ,reverse=True)
            ordered_kps += new_kps
            cols.extend([f"{title}_n_comments", f"{title}_precent"])

        if "none" in ordered_kps:
            ordered_kps.remove("none")

        add_change = False
        if len(titles) == 2:
            cols.append("change_percent")
            add_change = True

        rows = []
        title_to_precent = {}
        total_row = ["total"]
        for i,kp in enumerate(ordered_kps):
            row = [kp]
            for title in titles:
                n_comments_title_kp = results_to_kp_to_n_comments[title].get(kp,0)
                percent_comments_title_kp = (100*n_comments_title_kp/results_to_total_comments[title]) if n_comments_title_kp else 0
                row.extend([n_comments_title_kp,f'{percent_comments_title_kp:.2f}%'])
                title_to_precent[title] = percent_comments_title_kp
                if i == 0:
                    total_row.extend([results_to_total_comments[title] , "1"])
            if add_change:
                change_percent = title_to_precent[titles[1]] - title_to_precent[titles[0]]
                row.append(f'{change_percent:.2f}%')
                if i==0:
                    total_row.append("")
            rows.append(row)
        rows.append(total_row)
        comparison_df = pd.DataFrame(rows, columns=cols)
        return comparison_df

    @staticmethod
    def get_merged_pro_con_results(pro_result, con_result):
        assert pro_result.stances == {"pro"}, f"Pro results stances must be ['pro'], given {pro_result.stances}"
        assert con_result.stances == {"con"}, f"Con results stances must be ['con'], given {con_result.stances}"

        # pro and con results must be with identical data/params, except for the stance
        assert con_result.result_json["job_metadata"]["general"] == pro_result.result_json["job_metadata"]["general"]
        keypoint_matchings = []
        none_matchings = []
        for result in [pro_result, con_result]:
            for keypoint_matching in result.result_json["keypoint_matchings"]:
                if keypoint_matching["keypoint"] == "none":
                    none_matchings.extend(keypoint_matching["matching"])
                else:
                    keypoint_matchings.append(keypoint_matching)
        keypoint_matchings.append({'keypoint':'none', 'matching':none_matchings})
        keypoint_matchings.sort(key=lambda matchings: len(matchings['matching']), reverse=True)

        sentences_data = pro_result.result_json['sentences_data']
        for comment_id, comment_data_dict in con_result.result_json["sentences_data"].items():
            if comment_id not in sentences_data:
                 sentences_data[comment_id] = {"sents_in_comment":comment_data_dict["sents_in_comment"], "sentences":{}}
            for sent_id_in_comment, sent_data in comment_data_dict["sentences"].items():
                 sentences_data[comment_id]["sentences"][sent_id_in_comment] = sent_data

        pro_metadata = pro_result.result_json["job_metadata"]
        con_metadata = con_result.result_json["job_metadata"]
        new_metadata = {"general":pro_metadata["general"],
                        "per_stance":{"pro":pro_metadata["per_stance"]["pro"], "con":con_metadata["per_stance"]["con"]}}
        combined_results_json = {'keypoint_matchings':keypoint_matchings, "sentences_data":sentences_data,
                                 "version":CURR_RESULTS_VERSION,
                                 "job_metadata":new_metadata}
        return KpaResult.create_from_result_json(combined_results_json)

if __name__ == "__main__":
    init_logger()
    result_csv = "/Users/lilache/Library/CloudStorage/Box-Box/interview_analysis/austin_results/new_match_models/per_year_results/2018/standalone/Q25/austin_2018_Q25_new_models_neg.csv"
    #result_csv = "/Users/lilache/Library/CloudStorage/Box-Box/interview_analysis/debater_p_results/2022/final/v0_multi_kps_sbert_stage2_15/eng_kp_input_2022_simplified_multi_kps_merged_kpa_results.csv"
    # kpa_json_results_old = KpaResult.result_df_to_result_json(pd.read_csv(result_csv), new_version=False)
    # with open("old_json.json", 'w') as f:
    #     json.dump(kpa_json_results_old, f)
    #
    # new_json = KpaResult.convert_to_v2(kpa_json_results_old)
    # with open("new_json2.json", 'w') as f:
    #     json.dump(new_json, f)
    kpa_result = KpaResult.create_from_result_csv(result_csv)
    kpa_result.save("new_austin_result.json")
    #kpa_result.print_result(2, "test")
    kpa_result.export_to_all_outputs("","test_all_outputs")
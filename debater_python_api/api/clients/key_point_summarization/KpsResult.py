import ast
import json
import logging
import math
from collections import defaultdict
from typing import Optional, Dict, List

import pandas as pd
import numpy as np

from debater_python_api.api.clients.key_point_summarization.KpsExceptions import KpsIllegalInputException
from debater_python_api.api.clients.key_point_summarization.utils import read_dicts_from_df, create_dict_to_list, \
    write_df_to_file, get_unique_sent_id, filter_dict_by_keys, update_row_with_stance_data
from debater_python_api.api.clients.key_point_summarization.docx_generator import save_hierarchical_graph_data_to_docx
from debater_python_api.api.clients.key_point_summarization.graph_generator import create_graph_data, graph_data_to_hierarchical_graph_data, \
    get_hierarchical_graph_from_tree_and_subset_results, get_hierarchical_kps_data
import os
CURR_RESULTS_VERSION = "2.0"


class KpsResult:
    """
    Class to hold and process the results of a KPS job.
    """
    def __init__(self, result_json, filter_min_relations=0.4):
        """
        :param result_json: the result json returned from the key point client using get_results.
        """
        self.result_json = result_json
        kps = self.get_key_points()
        if len(kps) == 0:
            logging.info("No key points found. Returning empty KpsResult.")
        self.version = CURR_RESULTS_VERSION
        self.stances = set(result_json["job_metadata"]["per_stance"].keys()).difference({"no-stance"})
        self.result_df = self._create_result_df()
        self.filter_min_relations = filter_min_relations
        self.kp_id_to_hierarchical_data = self._get_kp_id_to_hierarchical_data()
        self.summary_df = self._result_df_to_summary_df()

    def get_unmapped_sentences_df(self):
        """
        return a df with all the sentences not matched during the job (either filtered for length/stance or no matching key point was found).
        """
        unmatched_sentences = self.result_json['unmatched_sentences']
        sentences_data = self.result_json['sentences_data']
        data_cols = ['span_start', 'span_end', 'num_tokens', 'argument_quality','sentence_text','stance']
        rows = []

        for s in unmatched_sentences:
            comment_id = str(s["comment_id"])
            sent_id_in_comment = str(s["sentence_id"])
            comment_data = sentences_data[comment_id]
            sent_data = comment_data["sentences"][sent_id_in_comment]
            sent_row = [comment_id, sent_id_in_comment, comment_data["sents_in_comment"]] + [sent_data[c] for c in data_cols]
            rows.append(sent_row)

        cols = ['comment_id', 'sentence_id', "sents_in_comment"] + data_cols

        unmatched_df = pd.DataFrame(rows, columns=cols)
        unmatched_df = unmatched_df.rename(columns={"stance":"stance_dict"})
        unmatched_df = unmatched_df.apply(lambda r: update_row_with_stance_data(r), axis=1)
        return unmatched_df

    def get_domain(self):
        return self.result_json["job_metadata"]["general"]["domain"]

    def get_stance_to_job_id(self):
        """
        Return a dictionary with the mapping from each stance to its job id
        """
        per_stance_metadata = self.result_json["job_metadata"]["per_stance"]
        return {k: per_stance_metadata[k]["job_id"] for k in per_stance_metadata}

    def save(self, json_file: str):
        """
        Save to current results as a json file
        :param json_file: path to the json file to hold the results
        """
        logging.info(f"Writing results to: {json_file}")
        with open(json_file, 'w') as f:
            json.dump(self.result_json, f)

    @staticmethod
    def load(json_file: str, filter_min_relations:Optional[float] = 0.4):
        """
        Load results from a json file
        :param json_file: the file to load the results from, obtained using the save() method.
        :return: KpsResult object.
        """
        logging.info(f"Loading results from: {json_file}")
        with open(json_file, 'r') as f:
            json_res = json.load(f)
            return KpsResult.create_from_result_json(json_res, filter_min_relations)

    def _create_result_df(self):
        sentences_data = self.result_json["sentences_data"]
        matchings_rows = []
        kps_have_stance = False
        sentences_have_stance = False
        stance_keys = []
        kps_to_stance = defaultdict(lambda: set())
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
                sent_id_in_comment = str(match["sentence_id"])
                comment_data = sentences_data[comment_id]
                sent_data = comment_data["sentences"][sent_id_in_comment]

                match_row = [kp, sent_data["sentence_text"], score, comment_id, int(sent_id_in_comment),
                             comment_data["sents_in_comment"], sent_data["span_start"], sent_data["span_end"], sent_data["num_tokens"],
                             sent_data["argument_quality"], kp_quality, sent_data["kp_quality"]]

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
                    kps_to_stance[kp].add(kp_stance)
                matchings_rows.append(match_row)
        matchings_cols = ["kp", "sentence_text", "match_score", 'comment_id', 'sentence_id', 'sents_in_comment',
                          'span_start', 'span_end', 'num_tokens', 'argument_quality', 'kp_quality', "sent_kp_quality"]
        if sentences_have_stance:
            matchings_cols.extend([f'{k}_score' for k in stance_keys] + ["selected_stance", "stance_conf"])
        if kps_have_stance:
            matchings_cols.append('kp_stance')
        results_df = pd.DataFrame(matchings_rows, columns=matchings_cols)

        if len(self.stances) > 1:
            kp_with_two_stances = list(filter(lambda x: len(kps_to_stance[x]) > 1, kps_to_stance.keys()))
            if len(kp_with_two_stances) > 0:
                results_df.loc[:,"kp"] = results_df.apply(lambda r: self._get_kp_stance_str(r['kp'], r['kp_stance']) if r['kp'] in kp_with_two_stances else r['kp'], axis = 1)
        return results_df

    def _get_kp_stance_str(self, kp, stance):
        return f"{kp} ({stance})"

    def _result_df_to_summary_df(self):

        kp_to_id = {self.kp_id_to_hierarchical_data[id]["kp"]:id for id in self.kp_id_to_hierarchical_data }
        dicts, _ = read_dicts_from_df(self.result_df)
        dicts = list(filter(lambda d: d["kp"]!= "none", dicts))

        kp_to_dicts = create_dict_to_list([(d['kp'], d) for d in dicts])

        n_total_sentences = self._get_number_of_unique_sentences()
        n_total_comments = self._get_number_of_unique_comments()

        summary_rows = []
        summary_cols = ["key_point", '#comments', 'comments_coverage', "#sentences", 'sentences_coverage', "stance","kp_id","parent_id", "n_comments_subtree"]

        for kp, kp_dicts in kp_to_dicts.items():

            n_sentences = len(kp_dicts)
            sentence_coverage = n_sentences / n_total_sentences if n_total_sentences > 0 else 0.0
            n_comments = len(set([d["comment_id"] for d in kp_dicts]))
            comments_coverage = n_comments / n_total_comments if n_total_comments > 0 else 0.0

            kp_id = kp_to_id[kp]
            kp_to_data = self.kp_id_to_hierarchical_data[kp_id]
            parent = kp_to_data['parent']
            n_comments_in_subtree = kp_to_data['n_matching_comments_in_subtree']
            kp_stance = kp_to_data.get('kp_stance', "")

            summary_row = [kp, n_comments, comments_coverage, n_sentences, sentence_coverage, kp_stance,
                           kp_id, parent, n_comments_in_subtree]
            summary_rows.append(summary_row)

        summary_rows = sorted(summary_rows, key=lambda x: x[summary_cols.index("#comments")], reverse=True)

        for stance in self.stances:
            if stance == "no-stance":
                continue
            stance_data = self.result_json["job_metadata"]["per_stance"][stance]
            stance_n_comments = stance_data["n_comments_stance"]
            stance_comment_coverage = stance_n_comments/n_total_comments
            stance_n_sentences = stance_data["n_sentences_stance"]
            stance_sentence_coverage = stance_n_sentences / n_total_sentences

            stance_row = ["*total_"+stance, stance_n_comments, stance_comment_coverage, stance_n_sentences, stance_sentence_coverage,
                          "","","",""]
            summary_rows.append(stance_row)

            stance_dicts = list(filter(lambda d:d["kp_stance"] == stance, dicts))
            stance_n_mapped_comments =  len(set([d["comment_id"] for d in stance_dicts]))
            stance_n_mapped_sentences = len(set([get_unique_sent_id(d) for d in stance_dicts]))
            stance_mapped_comment_coverage = stance_n_mapped_comments / n_total_comments
            stance_mapped_sentence_coverage = stance_n_mapped_sentences / n_total_sentences
            stance_mapped_row = ["*matched_" + stance, stance_n_mapped_comments, stance_mapped_comment_coverage, stance_n_mapped_sentences,
                          stance_mapped_sentence_coverage, "", "", "", ""]
            summary_rows.append(stance_mapped_row)

        total_row = ["*total",n_total_comments, 1, n_total_sentences, 1, "","","",""]
        summary_rows.append(total_row)

        n_mapped_sentences = len(set([get_unique_sent_id(d) for d in dicts]))
        n_mapped_comments =  len(set([d["comment_id"] for d in dicts]))
        total_mapped_row = ["*matched", n_mapped_comments, n_mapped_comments / n_total_comments, n_mapped_sentences,
                    n_mapped_sentences / n_total_sentences, "", "","", ""]
        summary_rows.append(total_mapped_row)

        return pd.DataFrame(summary_rows, columns=summary_cols)

    @staticmethod
    def create_from_result_json(result_json, filter_min_relations=0.4):
        """
        Create KpsResults from result_json
        :param result_json: the json object obtained from the client via "get_result" or "get_result_from_futures"
        :param filter_min_relations: minimal relation score between key points to be considered related in the hierarchy
        :return: KpsResult object
        """
        if 'keypoint_matchings' not in result_json:
            raise KpsIllegalInputException("Faulty results json provided: does not contain 'keypoint_matching'.")
        if "job_metadata" not in result_json:
            raise KpsIllegalInputException("Faulty result_json provided: does not contain 'job_metadata'. result_json must be"
                                           "the json object returned by the server when running with SDK version >= 5.0.0."
                                           "Converting older json results is not supported. to generate the old"
                                           "results for the given json object please use SDK version <= 4.3.2. To generate new result"
                                           "please rerun the KPS job using the current SDK")
        else:
            try:
                version = result_json.get("version", "1.0")
                if version != CURR_RESULTS_VERSION:
                    result_json = KpsResult._convert_to_new_version(result_json, version, CURR_RESULTS_VERSION)
                return KpsResult(result_json, filter_min_relations)
            except Exception as e:
                logging.error("Could not create KpsResults from json.")
                raise e

    def get_job_metadata(self):
        """ Return the metadata associated with the (or jobs in case of merged results)"""
        return self.result_json['job_metadata']

    @staticmethod
    def _convert_to_new_version(result_json, old_version, new_version):
        if old_version == "1.0" and new_version == "2.0":
            return KpsResult._convert_v1_to_v2(result_json)

        #can create more conversion methods for future json_result versions...
        raise KpsIllegalInputException(f"Unsupported results version old: {old_version}, new: {new_version}. "
                                       f"Supported: old = 1.0, new = 2.0")

    def generate_docx_report(self, output_dir: str, result_name: str,
                             n_matches_in_docx: Optional[int] = 50,
                             include_match_score_in_docx: Optional[bool] = False,
                             kp_id_to_hierarchical_data=None):
        """
        creates <output_dir>/<result_name>_hierarchical.docx: This Microsoft Word document shows the key point hierarchy and matching sentences
        as a user-friendly report.
        :param output_dir: path to output directory
        :param result_name: name of the results to appear in the output files.
        :param n_matches_in_docx: number of top matches to write in the textual summary (docx file). Pass None for all matches.
        :param include_match_score_in_docx: when set to true, the match score between the sentence and the key point is added.
        :param kp_id_to_hierarchical_data: optional, should be set to None.
        """
        if not kp_id_to_hierarchical_data:
            kp_id_to_hierarchical_data = self.kp_id_to_hierarchical_data
        docx_file = os.path.join(output_dir, f'{result_name}_hierarchical.docx')
        meta_data = self.result_json["job_metadata"]
        save_hierarchical_graph_data_to_docx(full_result_df=self.result_df, kp_id_to_data=kp_id_to_hierarchical_data,
                                             result_filename=docx_file, meta_data=meta_data, n_matches=n_matches_in_docx,
                                             include_match_score=include_match_score_in_docx,
                                             min_n_matches=0)

    def generate_docx_report_using_given_tree(self, full_results,
                                              output_dir, result_name, n_top_matches_in_graph=20,
                                              n_matches_in_docx=50, include_match_score_in_docx=False):
        '''
        Create hierarchical result for this result, using a precalculated hierarchical results.
        This is useful when we first create results using the whole data, and then want to calculate the
        hierarchical result of its subset while considering the already existing key points and hierarchy generated over the whole data.
        For example, when we have a large survey, we can first run over the entire data to create a hierarchical
        representation of the full results. Then when we want to evaluate a subset of the survey we can run over a subset of
        the survey using the same key points precomputed in the full survey. Then we create its hierarchical
        representation using the hierarchy of the full survey.
        :param full_results: KpsResult over the full data.
        :param output_dir: path to output directory
        :param result_name: name of the results to appear in the output files.
        :param n_top_matches_in_graph : optional, number of top matches to add to the graph_data file.
        :param n_matches_in_docx: optional, number of top matches to write in the textual summary (docx file). Pass None for all matches.
        :param include_match_score_in_docx: optional, when set to true, the match score between the sentence and the key point is added.
        '''
        graph_full_data = create_graph_data(full_results.result_df, full_results._get_number_of_unique_sentences())
        hierarchical_graph_full_data = graph_data_to_hierarchical_graph_data(graph_data=graph_full_data)

        new_hierarchical_graph_data = get_hierarchical_graph_from_tree_and_subset_results(
            hierarchical_graph_full_data,
            self.result_df, self.filter_min_relations, n_top_matches_in_graph)

        new_kp_id_to_hierarchical_data = get_hierarchical_kps_data(self.result_df, new_hierarchical_graph_data, self.filter_min_relations)
        self.generate_docx_report(output_dir, result_name,
                            n_matches_in_docx=n_matches_in_docx,
                            include_match_score_in_docx=include_match_score_in_docx,
                            kp_id_to_hierarchical_data = new_kp_id_to_hierarchical_data)

    def export_to_all_outputs(self, output_dir: str, result_name: str,
                              n_matches_in_docx: Optional[int] = 50,
                              include_match_score_in_docx: Optional[bool] = False,
                              ):
        """
        Generates all the kps available output types.
        :param output_dir: path to output directory
        :param result_name: name of the results to appear in the output files.
        :param n_matches_in_docx: optional, number of top matches to write in the textual summary (docx file). Pass None for all matches.
        :param include_match_score_in_docx: optional, when set to true, the match score between the sentence and the key point is added.
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
                                  include_match_score_in_docx=include_match_score_in_docx)

    def print_result(self, n_sentences_per_kp: int, title: str, n_top_kps:Optional[int]=None):
        '''
        Prints the key point summarization result to console. For each kp, display the number of matched comments, stance
        and top matching sentences.
        :param n_sentences_per_kp: number of top matched sentences to display for each key point
        :param title: title to print for the summarization
        :param n_top_kps: Optional, maximal number of kps to display.
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

        def print_kp(kp, stance, keypoint_matching, sentences_data, n_sentences_per_kp, n_comments):

            sentences = []
            for match in keypoint_matching:
                comment_id = str(match["comment_id"])
                sent_id_in_comment = str(match["sentence_id"])
                sent = sentences_data[comment_id]['sentences'][sent_id_in_comment]["sentence_text"]
                sentences.append(sent)
            print('%d - %s%s' % (n_comments, kp, '' if stance is None else ' - ' + stance))

            sentences = sentences[1:(n_sentences_per_kp + 1)]  # first sentence is the kp itself
            lines = split_sentences_to_lines(sentences, 0)
            for line in lines:
                print('\t%s' % line)

        keypoint_matchings = self.result_json["keypoint_matchings"]
        sentences_data = self.result_json["sentences_data"]

        n_total_comments = self._get_number_of_unique_comments(include_unmatched=True)
        n_matched_comments = self._get_number_of_unique_comments(include_unmatched=False)

        stance_str = self.get_stance_str()
        print(f'{title} results, stance: {stance_str} ')
        print(f'n_comments: {n_total_comments}')
        print('Coverage (all comments): %.2f' % (
                float(n_matched_comments) / float(n_total_comments) * 100.0))

        if len(self.stances) == 1:
            n_comments_with_stance = self.result_json["job_metadata"]["per_stance"][list(self.stances)[0]]["n_comments_stance"]
            if n_comments_with_stance == 0:
                cov_from_stance = np.NAN
            else:
                cov_from_stance = float(n_matched_comments) / float(n_comments_with_stance) * 100.0
            print(f'Coverage ({stance_str} comments): %.2f' % cov_from_stance)

        kp_stance_to_n_comments = self._get_kp_stance_to_n_matched_comments()

        sorted_kps_stance = list(sorted(kp_stance_to_n_comments.keys(), key=lambda x: kp_stance_to_n_comments[x], reverse=True))
        total_n_kps = len(sorted_kps_stance)
        n_top_kps = n_top_kps if n_top_kps else total_n_kps
        n_displayed_kps = np.min([n_top_kps, total_n_kps])
        sorted_kps = sorted_kps_stance[:n_displayed_kps]
        print(f'Displaying {n_displayed_kps} key points out of {total_n_kps}:')

        kp_stance_to_matching = {(keypoint_matching['keypoint'],keypoint_matching.get('stance')):  keypoint_matching for keypoint_matching in keypoint_matchings}

        for kp_stance in sorted_kps:
            keypoint_matching = kp_stance_to_matching[kp_stance]
            #stance = None if 'stance' not in keypoint_matching else keypoint_matching['stance']
            n_comments = kp_stance_to_n_comments[kp_stance]
            print_kp(kp_stance[0],kp_stance[1], keypoint_matching['matching'], sentences_data, n_sentences_per_kp, n_comments)

    def _get_number_of_unique_sentences(self, include_unmatched=True):
        if include_unmatched:
            return self.result_json["job_metadata"]["general"]["n_sentences"]
        else:
            total_sentences = set()
            for i, keypoint_matching in enumerate(self.result_json['keypoint_matchings']):
                matching_sents_ids = set([get_unique_sent_id(d) for d in keypoint_matching['matching']])
                if keypoint_matching['keypoint'] != 'none':
                    total_sentences = total_sentences.union(matching_sents_ids)
            return len(total_sentences)

    def _get_comparison_df(self, results_to_kp_to_n_comments, results_to_total_comments, titles):
        ordered_kps = []
        cols = ['key point', 'stance']

        for title in titles:
            n_comments_per_kp_per_title = results_to_kp_to_n_comments[title]
            new_kps = set(n_comments_per_kp_per_title.keys()).difference(set(ordered_kps))
            new_kps = sorted(new_kps, key=lambda x: n_comments_per_kp_per_title[x], reverse=True)
            ordered_kps += new_kps
            cols.extend([f"{title}_n_comments", f"{title}_percent"])

        add_change = False
        if len(titles) == 2:
            cols.append("change_percent")
            add_change = True

        rows = []
        title_to_precent = {}
        total_row = ["total", ""]
        for i, kp_stance in enumerate(ordered_kps):
            row = [kp_stance[0], kp_stance[1]]

            for title in titles:
                n_comments_title_kp = results_to_kp_to_n_comments[title].get(kp_stance, 0)
                percent_comments_title_kp = (
                            100 * n_comments_title_kp / results_to_total_comments[title]) if n_comments_title_kp else 0
                row.extend([n_comments_title_kp, f'{percent_comments_title_kp:.2f}%'])
                title_to_precent[title] = percent_comments_title_kp
                if i == 0:
                    total_row.extend([results_to_total_comments[title], "1"])
            if add_change:
                change_percent = title_to_precent[titles[1]] - title_to_precent[titles[0]]
                row.append(f'{change_percent:.2f}%')
                if i == 0:
                    total_row.append("")

            rows.append(row)
        if len(rows) > 0:
            rows.append(total_row)
        comparison_df = pd.DataFrame(rows, columns=cols)

        if (set(comparison_df["stance"]) == {None}):
            comparison_df = comparison_df[[c for c in comparison_df.columns if c != "stance"]]
        return comparison_df

    def compare_with_comment_subsets(self, comments_subsets_dict:Dict[str, List[str]], include_full:Optional[bool] = True):
        """
        Compare the full result with the results generated from comment subsets. This is useful in order to compare the
        key points' prevalence among different subsets of the full data.
        :param comments_subsets_dict: dictionary of string titles as keys and the set of comment ids that corresponds to each title.
        Comment ids that are not part of this result are ignored.
        :param include_full: Optional, default True, whether to include the full results in the comparison.
        :return: a dataframe containing the number and percentage of the comments matched to each key point in the full result and
        in each comments' subset, and the change percentage if comparing to a single subset.
        """
        results_to_total_comments = {"full": self._get_number_of_unique_comments()} if include_full else {}
        results_to_total_comments.update({title: len(comments_subsets_dict[title]) for title in comments_subsets_dict})

        titles = (["full"] if include_full else []) + sorted(comments_subsets_dict.keys(), key=lambda x: results_to_total_comments[x], reverse=True)

        results_to_kp_to_n_comments = {"full": self._get_kp_stance_to_n_matched_comments()} if include_full else {}
        results_to_kp_to_n_comments.update(
            {title: self._get_kp_stance_to_n_matched_comments(comments_subset=comment_ids) for title, comment_ids in comments_subsets_dict.items()})

        return self._get_comparison_df(results_to_kp_to_n_comments, results_to_total_comments, titles)

    def compare_with_other_results(self, this_title : str, other_results_dict : Dict[str,'KpsResult']):
        """
        Compare this result with other results.
        :param this_title: title to be associated with the current result.
        :param other_results_dict: dictionary of result titles as keys and KpsResult objects as values.
        :return: a dataframe containing the number and percentage of the comments matched to each key point in each result,
        and the change percentage if comparing to a single result.
        """
        results_to_total_comments = {this_title:self._get_number_of_unique_comments()}
        results_to_total_comments.update({title:result._get_number_of_unique_comments() for title,result in other_results_dict.items()})
        titles = [this_title] + sorted(other_results_dict.keys(), key=lambda x:results_to_total_comments[x], reverse=True)

        results_to_kp_to_n_comments = {this_title:self._get_kp_stance_to_n_matched_comments()}
        results_to_kp_to_n_comments.update({title:result._get_kp_stance_to_n_matched_comments()  for title,result in other_results_dict.items()})

        return self._get_comparison_df(results_to_kp_to_n_comments, results_to_total_comments, titles)

    @staticmethod
    def get_merged_pro_con_results(pro_result: 'KpsResult', con_result: 'KpsResult'):
        """
        Merge a pro KpsResult and a con KpsResult to a single merged KpsResult.
        The two results must be obtained by the same user, over the same domain and the same set of comments.
        :param pro_result: KpsResult generated from running on stance "PRO".
        :param con_result: KpsResult generated from running on stance "CON".
        """
        logging.info("Merging positive and negative KpsResults.")
        assert pro_result.stances == {"pro"}, f"pro_result must be generated from running on stance 'PRO', given {pro_result.stances}"
        assert con_result.stances == {"con"}, f"con_result must be generated from running on stance 'CON', given {con_result.stances}"
        con_meta = con_result.result_json["job_metadata"]["general"]
        pro_meta = pro_result.result_json["job_metadata"]["general"]
        for k in ["user_id","domain"]:
            assert con_meta[k] == pro_meta[k], f'pro_result and con_result must have the same {k}, but given {pro_meta[k]} for pro and {con_meta[k]} for con'
        assert con_meta["n_comments"] == pro_meta['n_comments'], f'pro_result and con_result must be generated from running on' \
                                                                f' the same set of comments, but pro was run over {pro_meta["n_comments"]}' \
                                                                f' comments and con over {con_meta["n_comments"]}'

        keypoint_matchings = []
        none_matchings = []
        for result in [pro_result, con_result]:
            for keypoint_matching in result.result_json["keypoint_matchings"]:
                if keypoint_matching["keypoint"] == "none":
                    none_matchings.extend(keypoint_matching["matching"])
                else:
                    keypoint_matchings.append(keypoint_matching)

        if len(none_matchings) > 0: # if there were non keypoints in original results, for backwards compatibility
            keypoint_matchings.append({'keypoint':'none', 'matching':none_matchings})
        keypoint_matchings.sort(key=lambda matchings: len(matchings['matching']), reverse=True)

        sentences_data = pro_result.result_json['sentences_data']
        for comment_id, comment_data_dict in con_result.result_json["sentences_data"].items():
            if comment_id not in sentences_data:
                 sentences_data[comment_id] = {"sents_in_comment":comment_data_dict["sents_in_comment"], "sentences":{}}
            for sent_id_in_comment, sent_data in comment_data_dict["sentences"].items():
                 sentences_data[comment_id]["sentences"][sent_id_in_comment] = sent_data

        pro_unmatched_sents = set([tuple(d.items()) for d in pro_result.result_json["unmatched_sentences"]])
        con_unmatched_sents = set([tuple(d.items()) for d in con_result.result_json["unmatched_sentences"]])
        unmatched_sents = pro_unmatched_sents.intersection(con_unmatched_sents)
        unmatched_sents = [dict(s_data) for s_data in unmatched_sents]

        pro_metadata = pro_result.result_json["job_metadata"]
        con_metadata = con_result.result_json["job_metadata"]
        new_metadata = {"general":pro_metadata["general"],
                        "per_stance":{"pro":pro_metadata["per_stance"]["pro"], "con":con_metadata["per_stance"]["con"]}}
        combined_results_json = {'keypoint_matchings':keypoint_matchings, "sentences_data":sentences_data,
                                 "version":CURR_RESULTS_VERSION,
                                 "job_metadata":new_metadata, "unmatched_sentences":unmatched_sents}
        filter_min_relations = np.min([pro_result.filter_min_relations, con_result.filter_min_relations])
        return KpsResult.create_from_result_json(combined_results_json, filter_min_relations=filter_min_relations)

    @staticmethod
    # If result_json is of the old server version, convert to version 2.0
    # 1.0 - received from the server, must have only one stance (merging pro_con is only on v2)
    def _convert_v1_to_v2(result_json):
        metadata = result_json["job_metadata"]
        kps_stance = KpsResult._get_stance_from_server_stances(metadata["stances"])
        sentences_data = {}

        unmapped_sentences = result_json.get("unused_sentences", []) # for server backward compatibility
        unmapped_sentences_ids = []
        for s in unmapped_sentences:
            KpsResult._add_sentence_to_sentences_data(s, sentences_data)
            unmapped_sentences_ids.append({"comment_id":s["comment_id"], "sentence_id":str(s["sentence_id"])})

        new_matchings = []
        for keypoint_matching in result_json['keypoint_matchings']:
            kp = keypoint_matching['keypoint']

            new_kp_matching = {'keypoint': kp, "matching": []}
            if kps_stance != "no-stance":
                new_kp_matching["stance"] = kps_stance

            for match in keypoint_matching['matching']:

                if kp == "none":
                    unmapped_sentences_ids.append({"comment_id": match["comment_id"], "sentence_id": str(match["sentence_id"])})

                if "kp_quality" not in new_kp_matching:
                    kpq = match.get("kp_quality", 0)
                    new_kp_matching["kp_quality"] = kpq

                KpsResult._add_sentence_to_sentences_data(match, sentences_data)

                new_kp_matching["matching"].append(
                    {"comment_id": match["comment_id"], "sentence_id": str(match["sentence_id"]), "score": match["score"]})

            if kp != 'none':
                new_matchings.append(new_kp_matching)

        per_stance_dict = {kps_stance: filter_dict_by_keys(metadata, ['description', 'run_params', 'job_id',
                                                                      'n_sentences_stance', 'n_comments_stance'])}

        metadata_v2 = {"general": filter_dict_by_keys(metadata, ['domain', 'user_id', 'n_sentences',
                                                                 'n_sentences_unfiltered', 'n_comments',
                                                                 'n_comments_unfiltered','domain_params']),
                       "per_stance": per_stance_dict}
        return {"keypoint_matchings": new_matchings, "sentences_data": sentences_data, "version": CURR_RESULTS_VERSION,
                "job_metadata": metadata_v2, "unmatched_sentences":unmapped_sentences_ids}

    @staticmethod
    def _add_sentence_to_sentences_data(match, sentences_data):
        comment_id = match["comment_id"]
        sent_id_in_comment = str(match["sentence_id"])
        if comment_id not in sentences_data:
            sentences_data[comment_id] = {"sents_in_comment": match["sents_in_comment"], "sentences": {}}

        if sent_id_in_comment not in sentences_data[comment_id]["sentences"]:
            sent_data = {c: match.get(c) for c in
                         ["sentence_text", "span_start", "span_end", "num_tokens", "argument_quality", "kp_quality"]}
            if "stance" in match:
                sent_data["stance"] = dict(match["stance"])
            sentences_data[comment_id]["sentences"][sent_id_in_comment] = sent_data.copy()

    def _get_number_of_unique_comments(self, include_unmatched=True):
        if include_unmatched:
            return self.result_json["job_metadata"]["general"]["n_comments"]
        else:
            total_comments = set()
            for i, keypoint_matching in enumerate(self.result_json['keypoint_matchings']):
                if keypoint_matching['keypoint'] != 'none':
                    matching_sents_ids = set([d["comment_id"] for d in keypoint_matching['matching']])
                    total_comments = total_comments.union(matching_sents_ids)
            return len(total_comments)

    def _get_kp_stance_to_n_matched_comments(self, comments_subset = None):
        kp_stance_n_comments = {}
        for keypoint_matching in self.result_json['keypoint_matchings']:
            kp = keypoint_matching["keypoint"]
            kp_stance = keypoint_matching.get("stance")
            matching_comments_set = set(match["comment_id"] for match in keypoint_matching["matching"])

            if comments_subset:
                matching_comments_set = set(filter(lambda x:x in comments_subset, matching_comments_set))

            n_matches = len(matching_comments_set)
            kp_stance_n_comments[(kp,kp_stance)] = n_matches

        return kp_stance_n_comments

    def get_key_points(self):
        return [keypoint_matching["keypoint"] for keypoint_matching in self.result_json['keypoint_matchings'] if keypoint_matching["keypoint"] != "none"]


    def _get_kp_id_to_hierarchical_data(self):
        graph_data = create_graph_data(self.result_df, n_sentences=self._get_number_of_unique_sentences())
        hierarchical_graph_data = graph_data_to_hierarchical_graph_data(graph_data=graph_data, filter_min_relations=self.filter_min_relations)
        return get_hierarchical_kps_data(self.result_df, hierarchical_graph_data, self.filter_min_relations)

    @staticmethod
    def _get_stance_from_server_stances(server_stances):

        if not server_stances or len(server_stances) == 0:
            return "no-stance"
        if server_stances == ["pos"]:
            return "pro"
        if set(server_stances) == set(["neg", "sug"]):
            return "con"

        raise KpsIllegalInputException(f'Unsupported stances {server_stances}')

    def get_result_with_top_kp_per_sentence(self):
        """
        Return the same KpsResult with up to one matching key point per sentence (the key point with the highest matching score).
        No hierarchy is generated in this setting.
        :return: new KpsResult after choosing the top kp for each sentence.
        """
        new_results_df = self.result_df.sort_values(by=["match_score"], ascending=[False])
        top_preds_df = pd.concat([group.head(1) for _,group in new_results_df.groupby(by=["comment_id","sentence_id"])])

        kp_to_args = list(zip(top_preds_df["kp"],zip(top_preds_df["comment_id"], top_preds_df["sentence_id"])))
        kp_to_args = create_dict_to_list(kp_to_args)
        new_meta_data = self.result_json["job_metadata"].copy()
        new_meta_data["general"]["top_kp_per_sent"] = True

        new_keypoint_matchings = []
        for keypoint_matching in self.result_json["keypoint_matchings"]:
            new_keypoint_matching = keypoint_matching.copy()
            kp = new_keypoint_matching["keypoint"]
            if kp != "none":
                kp_stance = new_keypoint_matching.get('stance')
                arg_for_kp = kp_to_args.get(kp)
                if arg_for_kp is None:
                    kp_stance = self._get_kp_stance_str(kp, kp_stance)
                    arg_for_kp = kp_to_args.get(kp_stance)
                if arg_for_kp:
                    new_keypoint_matching["matching"] = list(filter(lambda x: (x["comment_id"],int(x["sentence_id"])) in arg_for_kp, new_keypoint_matching["matching"]))
            if len(new_keypoint_matching["matching"]) > 0:
                new_keypoint_matchings.append(new_keypoint_matching)
        new_keypoint_matchings.sort(key=lambda x: len(x["matching"]), reverse=True)

        new_json = {"sentences_data": self.result_json["sentences_data"].copy(), "version": self.result_json["version"],
                    "job_metadata":new_meta_data, "keypoint_matchings":new_keypoint_matchings,
                    "unmatched_sentences": self.result_json["unmatched_sentences"]}

        return KpsResult.create_from_result_json(new_json)

    def get_stance_str(self):
        if len(self.stances) == 0:
            return "no-stance"
        elif len(self.stances) == 1:
            return list(self.stances)[0]
        return "pro and con"
    #
    # def get_kp_to_stance(self):
    #      keypoint_matchings = self.result_json["keypoint_matchings"]
    #      return {keypoint_matching['keypoint']:keypoint_matching.get("stance", None) for keypoint_matching in keypoint_matchings}

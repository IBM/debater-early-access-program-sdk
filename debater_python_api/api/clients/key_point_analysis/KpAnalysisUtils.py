import json
import logging
import math
from collections import defaultdict
from typing import Optional, List
import pandas as pd
import numpy as np

from debater_python_api.api.clients.key_point_analysis.KpaExceptions import KpaIllegalInputException
from debater_python_api.api.clients.key_point_analysis.KpaResult import KpaResult
from debater_python_api.api.clients.key_point_analysis.docx_generator import save_hierarchical_graph_data_to_docx
from debater_python_api.api.clients.key_point_analysis.utils import create_dict_to_list, read_dicts_from_df

class KpAnalysisUtils:
    '''
    A class with static methods for utilities that assist with the key point analysis service.
    '''
    @staticmethod
    def print_report(user_report, job_statuses: Optional[List[str]] = None, active_domains_only: Optional[bool] = True):
        '''
        Prints the user_report to console. :param user_report: the user report, returned by method get_full_report in
        KpAnalysisClient.
        :param job_statuses: optional, print only jobs with the listed statuses. possible statuses:
        'PENDING','PROCESSING','CANCELED','ERROR','DONE'
        :param active_domains_only: optional, print only jobs from domains that were not deleted. default: true.
        '''
        logging.info('User Report:')
        comments_statuses = user_report['domains_status']
        logging.info('  Comments status by domain (%d domains):' % len(comments_statuses))
        active_domains = []
        if len(comments_statuses) == 0:
            logging.info('    User has no domains')
        else:
            for status in comments_statuses:
                domain = status["domain"]
                active_domains.append(domain)
                logging.info(f'    Domain: {domain}, Domain Params: {status["domain_params"]}, Status: {status["data_status"]}')
        kp_analysis_statuses = user_report['kp_analysis_status']
        logging.info(f'  Key point analysis - jobs status:')
        if len(kp_analysis_statuses) == 0:
            logging.info('    User has no key point analysis jobs history')
        else:
            n_total_jobs = len(kp_analysis_statuses)
            if active_domains_only:
                kp_analysis_statuses = list(filter(lambda x:x["domain"] in active_domains, kp_analysis_statuses))
            if job_statuses and len(job_statuses) >0:
                kp_analysis_statuses = list(filter(lambda x: x["status"] in job_statuses, kp_analysis_statuses))
            n_displayed_jobs = len(kp_analysis_statuses)
            logging.info(f'  Displaying {n_displayed_jobs} jobs out of {n_total_jobs}: {"only active" if active_domains_only else "all"} domains, '
                         f'{"all" if not job_statuses else job_statuses} jobs statuses.')

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
    def write_result_to_csv(result_json, result_file, also_hierarchy=True):
        '''
        Writes the key point analysis result to file.
        Creates two files:
        * matches file: a file with all sentence-key point matches, saved in result_file path.
        * summary file: a summary file with all key points and their aggregated information, saved in result_file path with suffix: _kps_summary.csv.
        :param result: the result, returned by method get_result in KpAnalysisTaskFuture.
        :param result_file: a path to the file that will be created (should have a .csv suffix, otherwise it will be added).
        '''
        if 'keypoint_matchings' not in result_json:
            logging.info("No keypoint matchings results")
            return

        kpa_result = KpaResult.create_from_result_json(result_json)
        kpa_result.write_to_file(result_file, also_hierarchy)

    @staticmethod
    def compare_results(result_1, result_2, title_1='result1', title_2='result2'):
        kpa_results_1 = KpaResult.create_from_result_json(result_1, name=title_1)
        kpa_results_2 = KpaResult.create_from_result_json(result_2, name=title_2)
        return kpa_results_1.compare_with_other(kpa_results_2)

    @staticmethod
    def print_result(result_json, n_sentences_per_kp, title):
        KpaResult.create_from_result_json(result_json).print_result(n_sentences_per_kp=n_sentences_per_kp,
                                                                    title=title)

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
        if "stance_dict" in cols and sentences[0]["stance_dict"]:
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
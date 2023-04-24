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
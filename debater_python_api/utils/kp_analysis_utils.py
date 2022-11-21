import logging
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

def print_kps_summary(result):
    keypoint_matchings = result['keypoint_matchings']
    for keypoint_matching in keypoint_matchings:
        kp = keypoint_matching['keypoint']
        num_args = len(keypoint_matching['matching'])
        logging.info(kp + ' - ' + str(num_args))


def print_report(report):
    print('comments status:')
    comments_statuses = report['comments_status']
    for domain in comments_statuses:
        print('domain: %s, status: %s ' % (domain, str(comments_statuses[domain])))

    print('')
    print('kp_analysis jobs status:')
    kp_analysis_statuses = report['kp_analysis_status']
    for kp_analysis_status in kp_analysis_statuses:
        print(str(kp_analysis_status))


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'{prefix} |{bar}| {percent}% {suffix}\n')
    # Print New Line on Complete
    if iteration == total:
        print()


def create_dict_to_list(list_tups):
    res = defaultdict(list)
    for x, y in list_tups:
        res[x].append(y)
    return dict(res)

def write_df_to_file(df, file):
    logging.info("Writing dataframe to: " + file)
    file_path = Path(file)
    if not os.path.exists(file_path.parent):
        logging.info('creating directory: %s' % str(file_path.parent))
        os.makedirs(file_path.parent)
    df.to_csv(file, index=False)

import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def create_dict_to_list(list_tups):
    res = defaultdict(list)
    for x, y in list_tups:
        res[x].append(y)
    return dict(res)

def read_tups_from_df(input_df):
    cols = list(input_df.columns)
    tups = list(input_df.to_records(index=False))
    tups = [tuple([str(v) if str(v) != 'nan' else '' for v in t]) for t in tups]
    return tups, cols

def read_tups_from_csv(filename):
    logging.info(f'reading file: {filename}')
    input_df = pd.read_csv(filename)
    return read_tups_from_df(input_df)

def tups_to_dicts(tups, cols):
    dicts = [{} for _ in tups]
    dicts_tups = list(zip(dicts, tups))
    for i, c in enumerate(cols):
        for d, t in dicts_tups:
            d[c] = t[i]
    return dicts, cols


def read_dicts_from_df(df):
    tups, cols = read_tups_from_df(df)
    return tups_to_dicts(tups, cols)

def read_dicts_from_csv(filename):
    tups, cols = read_tups_from_csv(filename)
    return tups_to_dicts(tups, cols)

def trunc_float(fnum, points):
    for i in range(points):
        fnum *= 10

    fnum = float(int(fnum))
    for i in range(points):
        fnum /= 10

    return round(fnum, points)

def get_all_files_in_dir(path):
    files = os.listdir(path)
    files = [os.path.join(path,f) for f in files]
    return [f for f in files if os.path.isfile(f)]

def write_df_to_file(df, file):
    logging.info("Writing dataframe to: " + file)
    file_path = Path(file)
    if not os.path.exists(file_path.parent):
        logging.info('creating directory: %s' % str(file_path.parent))
        os.makedirs(file_path.parent)
    df.to_csv(file, index=False)


def update_row_with_stance_data(r):
    if "stance_dict" in r and len(r["stance_dict"]) > 0:
        stance_dict = dict(r["stance_dict"])
        stance_list = list(stance_dict.items())
        stance_list = sorted(stance_list, reverse=True, key=lambda item: item[1])
        stance, conf = stance_list[0]
        r["selected_stance"] = stance
        r["stance_conf"] = conf
    return r


def get_unique_sent_id(sentence_dict):
    return f"{sentence_dict['comment_id']}_{sentence_dict['sentence_id']}"


def get_cid_and_sid_from_sent_identifier(sent_identifier):
    splits = sent_identifier.split("_")
    sent_id = int(splits[-1])
    cid = "_".join(splits[:-1])
    return cid, sent_id


def validate_api_key_or_throw_exception(apikey):
    if len(apikey) != 35 or re.match('^[0-9a-zA-Z]+$', apikey) is None:
        raise ValueError("api key is not valid")
    return True


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

def get_default_request_header(apikey):
    return {'accept-encoding': 'gzip, deflate',
            'content-type': 'application/json',
            'charset': 'UTF-8',
            'apikey': apikey}


def filter_dict_by_keys(d, keys):
    return {k:d[k] for k in keys if k in d}


def is_list_of_strings(lst):
    return isinstance(lst, list) and len([a for a in lst if not isinstance(a, str)]) == 0

# sort by val (descending), with key (ascending) as tie breaker
def sort_dict_items_by_value_then_key(d, val_reverse = True, key_reverse = False):
    d_items = list(d.items())
    d_items.sort(key=lambda x: x[0], reverse = key_reverse)
    d_items.sort(key=lambda x: x[1], reverse = val_reverse)
    return d_items
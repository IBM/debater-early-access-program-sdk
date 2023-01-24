import logging
import os
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

import logging
import os
from collections import defaultdict

import pandas as pd


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

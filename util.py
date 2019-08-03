import numpy as np
import pandas as pd


def get_data(limit=None):
    data = pd.read_csv('/media/work/Work/ALL_FINAL_CODE/data/train.csv')
    minist_data = data.values
    x = minist_data[:, 1:] / 255
    y = minist_data[:, 0]
    if limit is not None:
        x, y = x[:limit], y[:limit]
    return x, y


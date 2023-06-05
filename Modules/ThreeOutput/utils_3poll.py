import os
from re import S

os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

import numpy as np
import pandas as pd
from tqdm  import tqdm
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image
from torch.utils.data import Dataset

import torch
import random

import xarray as xr
import rioxarray

from train_utils_3poll import eval_metrics

def read_param_file(filepath):
    with open(filepath, "r") as f:
        output = f.read()
    return output

def step(x, y_samples, model, loss, optimizer):

    y_no2, y_o3, y_pm10 = y_samples
    y_hat_1,y_hat_2,y_hat_3 = model(x)

    y_train = torch.stack([y_no2, y_o3, y_pm10])
    y_epoch = torch.stack([y_hat_1,y_hat_2,y_hat_3])
    loss_epoch = loss(y_epoch,y_train.to("cuda:0"))

    optimizer.zero_grad()
    loss_epoch.backward()
    optimizer.step()

    metric_results = eval_metrics(y_train.detach().cpu(),y_epoch.detach().cpu())
    #print(metric_results)

    return loss_epoch.detach().cpu(), metric_results

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_data(datadir, samples_file):
    """load samples to memory, returns array of samples and array of stations
    each sample is a dict
    this version loads all samples from one station in one go (e.g. for multiple months), s.t. the S5P data for the station is only read once"""

    if not isinstance(samples_file, pd.DataFrame):
        samples_df = pd.read_csv(samples_file, index_col="idx")
    else:
        samples_df = samples_file
    samples_df = samples_df[np.isnan(samples_df.no2) == False]

    print("Available columns from samples_file :")
    print(samples_df.columns)

    samples = []
    stations = {}
    try:
        # here we assume that all S5P data for one station is stored in one .netcdf file
        # so it's faster to access the samples on a per station basis and only opening the
        # .netcdf file once
        for station in tqdm(samples_df.AirQualityStation.unique()):
            station_obs = samples_df[samples_df.AirQualityStation == station]
            s5p_path = station_obs.s5p_path.unique().item()
            s5p_data = xr.open_dataset(os.path.join(datadir, "sentinel-5p", s5p_path)).rio.write_crs(4326)

            for idx in station_obs.index.values:
                sample = samples_df.loc[idx].to_dict() # select by index value, not position
                sample["idx"] = idx
                sample["s5p"] = s5p_data.tropospheric_NO2_column_number_density.values.squeeze()
                samples.append(sample)
                stations[sample["AirQualityStation"]] = np.load(os.path.join(datadir, "sentinel-2", sample["img_path"]))

            s5p_data.close()

    except IndexError as e:
        print(e)
        print("idx:", idx)
        print()

    #print(samples)
    return samples, stations

def none_or_true(value):
    if value == 'None':
        return None
    elif value == "True":
        return True
    return value

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
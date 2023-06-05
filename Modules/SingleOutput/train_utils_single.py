import os

os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

import sys
import copy
from tqdm  import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_single import NO2PredictionDataset
from transforms_single import ChangeBandOrder, ToTensor, DatasetStatistics, Normalize, Randomize
from model_single import get_model

def eval_metrics(y, y_hat):
    r2 = r2_score(y, y_hat)
    mae = mean_absolute_error(y, y_hat)
    mse = mean_squared_error(y, y_hat)

    return [r2, mae, mse]

def split_samples(samples, stations, test_size=0.25, val_size=0.25):
    stations_train, stations_test = train_test_split(stations, test_size=test_size)
    real_val_size = val_size / (1 - test_size)
    stations_train, stations_val = map(set, train_test_split(stations_train, test_size=real_val_size))
    stations_test = set(stations_test)

    samples_train = [s for s in samples if s["AirQualityStation"] in stations_train]
    samples_test = [s for s in samples if s["AirQualityStation"] in stations_test]
    samples_val = [s for s in samples if s["AirQualityStation"] in stations_val]

    return samples_train, samples_val, samples_test, stations_train, stations_val, stations_test

def split_samples_df(samples, test_size=0.2, val_size=0.2):
    """split pd.DF s.t. all samples of a given station
    are either in the train or test set """
    stations = samples.AirQualityStation.unique()
    stations_train, stations_test = train_test_split(stations, test_size=test_size)
    real_val_size = val_size / (1 - test_size)
    stations_train, stations_val = map(set, train_test_split(stations_train, test_size=real_val_size))
    stations_test = set(stations_test)

    samples_train = samples[samples.AirQualityStation.isin(stations_train)]
    samples_val = samples[samples.AirQualityStation.isin(stations_val)]
    samples_test = samples[samples.AirQualityStation.isin(stations_test)]

    return samples_train, samples_val, samples_test

def test(model, dataloader, device, datastats, StudyPollutant):
    model.eval()
    measurements=[]
    predictions=[]

    with torch.no_grad():
        for idx, sample in enumerate(dataloader):
            img = sample["img"].float().to(device)
            s5p = sample["s5p"].float().unsqueeze(dim=1).to(device)

            Onehot1 = sample["rural"]
            Onehot2 = sample["suburban"]
            Onehot3 = sample["urban"]
            Onehot4 = sample["traffic"]
            Onehot5 = sample["industrial"]
            Onehot6 = sample["background"]
            tabular = [sample["Altitude"], sample["PopulationDensity"], Onehot1, Onehot2, Onehot3, Onehot4, Onehot5,Onehot6]
            tabular = torch.stack(tabular,dim=1).float().to(device)

            model_input = {"img": img, "s5p": s5p, "tabular": tabular}

            y = sample[StudyPollutant].float().to(device).squeeze()
            y_hat = model(model_input).squeeze()

            measurements.append(y.cpu().numpy().item())
            predictions.append(y_hat.cpu().numpy().item())

    # measurements = np.array(measurements)
    # predictions = np.array(predictions)
    if StudyPollutant == "no2":
        measurements = Normalize.undo_no2_standardization(datastats, np.array(measurements))
        predictions = Normalize.undo_no2_standardization(datastats, np.array(predictions))
    elif StudyPollutant == "o3":
        measurements = Normalize.undo_o3_standardization(datastats, np.array(measurements))
        predictions = Normalize.undo_o3_standardization(datastats, np.array(predictions))
    elif StudyPollutant == "co":
        measurements = Normalize.undo_co_standardization(datastats, np.array(measurements))
        predictions = Normalize.undo_co_standardization(datastats, np.array(predictions))
    elif StudyPollutant == "so2":
        measurements = Normalize.undo_so2_standardization(datastats, np.array(measurements))
        predictions = Normalize.undo_so2_standardization(datastats, np.array(predictions))
    elif StudyPollutant == "pm10":
        measurements = Normalize.undo_pm10_standardization(datastats, np.array(measurements))
        predictions = Normalize.undo_pm10_standardization(datastats, np.array(predictions))
    return measurements, predictions



def test_plotter(output_directory,test_y, test_y_hat, train_y, train_y_hat, StudyPollutant):
    # create plot
    img, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    for ax in (ax1, ax2):
        #ax.set_xlim((0, 100))
        #ax.set_ylim((0, 100))
        ax.axline((0, 0), slope=1, c="red")
        ax.set_aspect('equal')
    ax1.scatter(test_y, test_y_hat, s=2)
    ax1.set_xlabel("Measurements")
    ax1.set_ylabel("Predictions")
    ax1.set_title(StudyPollutant.upper()+" test")
    ax2.scatter(train_y, train_y_hat, s=2)
    ax2.set_title(StudyPollutant.upper()+" train")
    ax2.set_xlabel("Measurements")
    ax2.set_ylabel("Predictions")
    plt.savefig(output_directory + "/predictions.png")
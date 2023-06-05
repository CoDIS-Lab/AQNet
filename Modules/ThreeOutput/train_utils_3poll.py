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

from dataset_3poll import NO2PredictionDataset
from transforms_3poll import ChangeBandOrder, ToTensor, DatasetStatistics, Normalize, Randomize
from model_3poll import get_model

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

def test(model, dataloader, device, datastats):
    model.eval()

    measurements_no2 =[]
    measurements_o3 =[]
    measurements_pm10 =[]

    predictions_no2 =[]
    predictions_o3 =[]
    predictions_pm10 =[]

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

            y_hat_1,y_hat_2,y_hat_3 = model(model_input)
            y_hat_no2 = y_hat_1.squeeze()
            y_hat_o3 = y_hat_2.squeeze()
            y_hat_pm10 = y_hat_3.squeeze()

            y_no2 = sample["no2"].float().to(device).squeeze()
            y_o3 = sample["o3"].float().to(device).squeeze()
            y_pm10 = sample["pm10"].float().to(device).squeeze()

            measurements_no2.append(y_no2.cpu().numpy().item())
            measurements_o3.append(y_o3.cpu().numpy().item())
            measurements_pm10.append(y_pm10.cpu().numpy().item())

            predictions_no2.append(y_hat_no2.cpu().numpy().item())
            predictions_o3.append(y_hat_o3.cpu().numpy().item())
            predictions_pm10.append(y_hat_pm10.cpu().numpy().item())

    measurements_no2 = Normalize.undo_no2_standardization(datastats, np.array(measurements_no2))
    measurements_o3 = Normalize.undo_o3_standardization(datastats, np.array(measurements_o3))
    measurements_pm10 = Normalize.undo_pm10_standardization(datastats, np.array(measurements_pm10))

    predictions_no2 = Normalize.undo_no2_standardization(datastats, np.array(predictions_no2))
    predictions_o3 = Normalize.undo_o3_standardization(datastats, np.array(predictions_o3))
    predictions_pm10 = Normalize.undo_pm10_standardization(datastats, np.array(predictions_pm10))

    measurements = {"no2":measurements_no2,
                    "o3":measurements_o3,
                    "pm10":measurements_pm10}
    predictions = {"no2":predictions_no2,
                    "o3":predictions_o3,
                    "pm10":predictions_pm10}

    return measurements, predictions



def test_plotter(output_directory,test_y, test_y_hat, train_y, train_y_hat):
    # create plot
    img, axs = plt.subplots(3, 2, figsize=(12, 12))
    #img.tight_layout()
    img.subplots_adjust(hspace=1)
    img.subplots_adjust(wspace=0.25)

    no2_lim = 80
    o3_lim = 120
    c0_lim = 2.5
    so2_lim = 40
    pm10_lim = 80

    #long winded plotting
    #no2
    axs[0, 0].scatter(test_y["no2"], test_y_hat["no2"], s=2)
    axs[0, 0].set_title("no2 test")
    axs[0, 0].set_xlabel("Measurements")
    axs[0, 0].set_ylabel("Predictions")
    # axs[0, 0].set_xlim((0, no2_lim))
    # axs[0, 0].set_ylim((0, no2_lim))
    axs[0, 0].set_aspect('equal')
    axs[0, 0].axline((0, 0), slope=1, c ="red")
    axs[0, 1].scatter(train_y["no2"], train_y_hat["no2"], s=2)
    axs[0, 1].set_title("no2 train")
    axs[0, 1].set_xlabel("Measurements")
    axs[0, 1].set_ylabel("Predictions")
    # axs[0, 1].set_xlim((0, no2_lim))
    # axs[0, 1].set_ylim((0, no2_lim))
    axs[0, 1].set_aspect('equal')
    axs[0, 1].axline((0, 0), slope=1, c ="red")
    #o3
    axs[1, 0].scatter(test_y["o3"], test_y_hat["o3"], s=2)
    axs[1, 0].set_title("o3 test")
    axs[1, 0].set_xlabel("Measurements")
    axs[1, 0].set_ylabel("Predictions")
    # axs[1, 0].set_xlim((0, o3_lim))
    # axs[1, 0].set_ylim((0, o3_lim))
    axs[1, 0].set_aspect('equal')
    axs[1, 0].axline((0, 0), slope=1, c ="red")
    axs[1, 1].scatter(train_y["o3"], train_y_hat["o3"], s=2)
    axs[1, 1].set_title("o3 train")
    axs[1, 1].set_xlabel("Measurements")
    axs[1, 1].set_ylabel("Predictions")
    # axs[1, 1].set_xlim((0, o3_lim))
    # axs[1, 1].set_ylim((0, o3_lim))
    axs[1, 1].set_aspect('equal')
    axs[1, 1].axline((0, 0), slope=1, c ="red")
    #pm10
    axs[2, 0].scatter(test_y["pm10"], test_y_hat["pm10"], s=2)
    axs[2, 0].set_title("pm10 test")
    axs[2, 0].set_xlabel("Measurements")
    axs[2, 0].set_ylabel("Predictions")
    # axs[2, 0].set_xlim((0, pm10_lim))
    # axs[2, 0].set_ylim((0, pm10_lim))
    axs[2, 0].set_aspect('equal')
    axs[2, 0].axline((0, 0), slope=1, c ="red")
    axs[2, 1].scatter(train_y["pm10"], train_y_hat["pm10"], s=2)
    axs[2, 1].set_title("pm10 train")
    axs[2, 1].set_xlabel("Measurements")
    axs[2, 1].set_ylabel("Predictions")
    # axs[2, 1].set_xlim((0, pm10_lim))
    # axs[2, 1].set_ylim((0, pm10_lim))
    axs[2, 1].set_aspect('equal')
    axs[2, 1].axline((0, 0), slope=1, c ="red")

    plt.savefig(output_directory + "/predictions.png")
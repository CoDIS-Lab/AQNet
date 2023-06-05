import os

os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

import sys
import copy
import random
import argparse
from datetime import datetime
from distutils.util import strtobool

import mlflow
import numpy as np
import pandas as pd
from tqdm  import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset_single import NO2PredictionDataset
from transforms_single import ChangeBandOrder, ToTensor, DatasetStatistics, Normalize, Randomize
from model_single import get_model
from utils_single import load_data, none_or_true, dotdict, set_seed, step
from train_utils_single import eval_metrics, split_samples,  test, test_plotter

parser = argparse.ArgumentParser(description='train_singlemodel')
network = "mobilenet_v3_small"
tabular_switch = True
S5p_switch = True
StudyPollutant = "no2" #3 pollutants : no2, o3, pm10 

tab_label = "SatDataOnly"
if tabular_switch == True:
    tab_label = "SatAndTabData"

##### block for complete training
parser.add_argument('--samples_file', default="./data/multimodal/samples_multimodal_3polls.csv", type=str)
parser.add_argument('--datadir', default="/lustre/home/ca-okarakus/AQNet/Datadir/eea", type=str)
parser.add_argument('--result_dir', default="results", type=str)
parser.add_argument('--checkpoint', default=True, type=str)
parser.add_argument('--epochs', default=30, type=int) #default=10
parser.add_argument('--batch_size', default=5, type=int) #default=50
parser.add_argument('--runs', default=5, type=int)

# block for testing
# parser.add_argument('--samples_file', default="data/multimodal/samples_S2S5P_2018_2020_epa_experiments.csv", type=str)
# parser.add_argument('--datadir', default="epa", type=str)
# parser.add_argument('--result_dir', default="results/TestingDump", type=str)
# parser.add_argument('--checkpoint', default=True, type=str)
# parser.add_argument('--epochs', default=3, type=int) #default=10
# parser.add_argument('--batch_size', default=5, type=int) #default=50
# parser.add_argument('--runs', default=1, type=int)
#
# # training parameters
parser.add_argument('--early_stopping', default="True", type=str)
parser.add_argument('--weight_decay_lambda', default=0.001, type=float)
parser.add_argument('--learning_rate', default=0.001, type=float)

args = parser.parse_args()
bool_args = ["early_stopping"]
config = dotdict({k : strtobool(v) if k in bool_args else v for k,v in vars(args).items()})

# set internal parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

assert config.checkpoint in [True, None]
if config.checkpoint is True:
    checkpoint_name = "ImageNet"
else: checkpoint_name = "FromScratch"

experiment = "_".join([datetime.today().strftime('%Y-%m-%d-%H:%M'), checkpoint_name])

# generate output folder and ID
experiment_folder_name = "_".join([datetime.today().strftime('%Y%m%d_%H%M'),str(config.epochs),"epochs",StudyPollutant,tab_label])
output_directory = os.path.join(config.result_dir,experiment_folder_name) #"results/ComparingS2BackboneModels"
os.mkdir(output_directory)
experiment_id = mlflow.create_experiment(experiment)

# print config info to cmd
print(config.samples_file)
print(config.datadir)
print(config.checkpoint)
print(device)
print("Start loading samples...")

# load data and instantiate objects
samples, stations = load_data(config.datadir, config.samples_file)
loss = nn.MSELoss()
datastats = DatasetStatistics()
tf = transforms.Compose([ChangeBandOrder(), Normalize(datastats), Randomize(), ToTensor()])

# set up performances lists
performances_test = []
performances_val = []
performances_train = []

# MODEL TRAINING
for run in tqdm(range(1, config.runs+1), unit="run"):

    # fix a different seed for each run
    seed = run

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("samples_file", config.samples_file)
        mlflow.log_param("datadir", config.datadir)
        mlflow.log_param("batch_size", config.batch_size)
        mlflow.log_param("result_dir", config.result_dir)
        mlflow.log_param("pretrained_checkpoint", config.checkpoint)
        mlflow.log_param("device", device)
        mlflow.log_param("early_stopping", config.early_stopping)
        mlflow.log_param("learning_rate", config.learning_rate)
        mlflow.log_param("run", run)
        mlflow.log_param("weight_decay", config.weight_decay_lambda)
        mlflow.log_param("epochs", config.epochs)
        mlflow.log_param("seed", seed)

        # set the seed for this run
        set_seed(seed)

        # initialize dataloaders + model
        print("POLLUTANT TO BE STUDIED = "+StudyPollutant)
        print("Initializing dataset")
        samples_train, samples_val, samples_test, stations_train, stations_val, stations_test = split_samples(samples, list(stations.keys()), 0.25, 0.25)
        print("First stations_train:", list(stations_train)[:10])

        dataset_test = NO2PredictionDataset(config.datadir, samples_test, transforms=tf, station_imgs=stations)
        dataset_train = NO2PredictionDataset(config.datadir, samples_train, transforms=tf, station_imgs=stations)
        dataset_val = NO2PredictionDataset(config.datadir, samples_val, transforms=tf, station_imgs=stations)

        dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, num_workers=0, shuffle=True, pin_memory=False)
        dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=0, shuffle=False, pin_memory=False)
        dataloader_val = DataLoader(dataset_val, batch_size=1, num_workers=0, shuffle=False, pin_memory=False)
        dataloader_train_for_testing = DataLoader(dataset_train, batch_size=1, num_workers=0, shuffle=False, pin_memory=False)

        # instantiate model
        print("Initializing model")
        model = get_model(device, network,tabular_switch,S5p_switch, config.checkpoint)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay_lambda)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5, threshold=1e6, min_lr=1e-7, verbose=True)

        print("Dataset Lengths - Train, Val, Test: "+str([len(dataset_train),len(dataset_val),len(dataset_test)]))

        print("Start training")
        # train the model
        for epoch in range(config.epochs):
            model.train()

            # set up blank lists
            loss_history = []
            loss_epoch = []
            r2_epoch = []
            mae_epoch = []
            mse_epoch = []

            for idx, sample in enumerate(dataloader_train):
                # read in satellite data
                model_input = sample["img"].float().to(device)
                s5p = sample["s5p"].float().unsqueeze(dim=1).to(device)

                # create tabular data
                Onehot1 = sample["rural"]
                Onehot2 = sample["suburban"]
                Onehot3 = sample["urban"]
                Onehot4 = sample["traffic"]
                Onehot5 = sample["industrial"]
                Onehot6 = sample["background"]

                tabular = [sample["Altitude"],sample["PopulationDensity"],Onehot1,Onehot2,Onehot3,Onehot4,Onehot5,Onehot6]
                #tabular = [sample["Altitude"],sample["PopulationDensity"]]
                tabular = torch.stack(tabular,dim=1).float().to(device)

                # set up model input and ground truth
                if S5p_switch == True:
                    if tabular_switch == True:
                        model_input = {"img": model_input, "s5p": s5p, "tabular": tabular}
                    else:
                        model_input = {"img" : model_input, "s5p" : s5p}
                else:
                    if tabular_switch == True:
                        model_input = {"img": model_input, "tabular": tabular}
                    else:
                        model_input = {"img" : model_input}

                y = sample[StudyPollutant].float()
                loss_batch, metric_results = step(model_input, y, model, loss, optimizer)
                loss_epoch.append(loss_batch.item())
                r2_epoch.append(metric_results[0])
                mae_epoch.append(metric_results[1])
                mse_epoch.append(metric_results[2])

            # epoch steps
            loss_epoch = np.array(loss_epoch).mean()
            r2_train_epoch = np.array(r2_epoch).mean()
            mae_train_epoch = np.array(mae_epoch).mean()
            mse_train_epoch = np.array(mse_epoch).mean()

            scheduler.step(loss_epoch)
            torch.cuda.empty_cache()
            loss_history.append(loss_epoch)

            # validation
            val_y, val_y_hat = test( model, dataloader_val, device, datastats, StudyPollutant)
            valid_val = (val_y_hat < 100) & (val_y_hat > 0)
            # valid_val = (val_y_hat["no2"] > 0)&(val_y_hat["no2"] < 100)&\
            #             (val_y_hat["o3"] > 0)&(val_y_hat["o3"] < 150)&\
            #             (val_y_hat["co"] > 0)&(val_y_hat["co"] < 5 )&\
            #             (val_y_hat["so2"] > 0)&(val_y_hat["so2"] < 50)&\
            #             (val_y_hat["pm10"] > 0)&(val_y_hat["pm10"] < 100) #(val_y_hat < 100) & (val_y_hat > 0)

            eval_val = eval_metrics(val_y, val_y_hat)
            print("Fraction of valid estimates:", sum(valid_val)/len(valid_val))

            #stop training if evaluation performance does not increase
            if epoch > 25 and sum(valid_val) > len(valid_val) - 5:
                if eval_val[0] > np.mean([performances_val[-3][2], performances_val[-2][2], performances_val[-1][2]]):
                    # performance on evaluation set is decreasing
                    print(f"Early stop at epoch: {epoch}")
                    mlflow.log_param("early_stop_epoch", epoch)
                    break

            # log results
            print(f"Epoch: {epoch}, {eval_val}")
            performances_val.append([run, epoch] + eval_val)
            mlflow.log_metrics({"val_r2_epoch" : eval_val[0], "val_mae_epoch" : eval_val[1], "val_mse_epoch" : eval_val[2]}, step=epoch)
            mlflow.log_metrics({"train_loss_epoch" : loss_epoch, "train_r2_epoch" : r2_train_epoch, "train_mae_epoch" : mae_train_epoch, "train_mse_epoch" : mse_train_epoch}, step=epoch)
            mlflow.log_metric("current_epoch", epoch, step=epoch)

        # test run statistics
        test_y, test_y_hat = test(model, dataloader_test, device, datastats, StudyPollutant)
        train_y, train_y_hat = test(model, dataloader_train_for_testing, device, datastats, StudyPollutant)

        valid = (test_y_hat < 100) & (test_y_hat > 0)
        valid_train = (train_y_hat < 100) & (train_y_hat > 0)

        eval_test = eval_metrics(test_y, test_y_hat)
        eval_train = eval_metrics(train_y, train_y_hat)

        #create fit graphs
        test_plotter(output_directory,test_y, test_y_hat, train_y, train_y_hat, StudyPollutant)

        mlflow.log_metric("test_r2", eval_test[0])
        mlflow.log_metric("test_mae", eval_test[1])
        mlflow.log_metric("test_mse", eval_test[2])

        performances_test.append(eval_test)
        performances_train.append(eval_train)

        mlflow.log_artifacts(output_directory) # log everything that was written to the artifacts directory

# set up performance dfs
performances_val = pd.DataFrame(performances_val, columns=["run", "epoch", "r2", "mae", "mse"])
performances_test = pd.DataFrame(performances_test, columns=["r2", "mae", "mse"])
performances_train = pd.DataFrame(performances_train, columns=["r2", "mae", "mse"])

# save results
print("Writing results...")
performances_test.to_csv(os.path.join(output_directory, "_".join([str(checkpoint_name), "test", str(config.epochs), "epochs"]) + ".csv"), index=False)
performances_train.to_csv(os.path.join(output_directory, "_".join([str(checkpoint_name), "train", str(config.epochs), "epochs"]) + ".csv"), index=False)
performances_val.to_csv(os.path.join(output_directory, "_".join([str(checkpoint_name), "val", str(config.epochs), "epochs"]) + ".csv"), index=False)

# save the model
print("Writing model...")
torch.save(model.state_dict(), os.path.join(output_directory, "_".join([str(checkpoint_name), str(config.epochs), "epochs"]) + ".model"))
print("done.")
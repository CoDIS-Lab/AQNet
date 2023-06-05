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

from dataset_3poll import NO2PredictionDataset
from transforms_3poll import ChangeBandOrder, ToTensor, DatasetStatistics, Normalize, Randomize
from model_3poll import get_model
from utils_3poll import load_data, none_or_true, dotdict, set_seed, step
from train_utils_3poll import eval_metrics, split_samples,  test, test_plotter

parser = argparse.ArgumentParser(description='train_multimodel')
network = "mobilenetv3small" #alternatively: "resnet50","resnet18"

# block for complete training
parser.add_argument('--samples_file', default="data/multimodal/samples_multimodal_3polls.csv", type=str)
parser.add_argument('--datadir', default="/lustre/home/ca-okarakus/AQNet/Datadir/eea", type=str)
parser.add_argument('--result_dir', default="results", type=str)
parser.add_argument('--checkpoint', default=True, type=str)
parser.add_argument('--epochs', default=30, type=int) 
parser.add_argument('--batch_size', default=15, type=int)
parser.add_argument('--runs', default=5, type=int)
parser.add_argument('--tabular', default="True", type=str)


# block for testing
# parser.add_argument('--samples_file', default="data/multimodal/samples_S2S5P_2018_2020_epa_experiments.csv", type=str)
# parser.add_argument('--datadir', default="epa", type=str)
# parser.add_argument('--result_dir', default="results/TestingDump", type=str)
# parser.add_argument('--checkpoint', default=None, type=str)
# parser.add_argument('--epochs', default=3, type=int) #default=10
# parser.add_argument('--batch_size', default=15, type=int) #default=50
# parser.add_argument('--runs', default=5, type=int)
#
# # training parameters
parser.add_argument('--early_stopping', default="True", type=str)
parser.add_argument('--weight_decay_lambda', default=0.001, type=float)
parser.add_argument('--learning_rate', default=0.001, type=float)

args = parser.parse_args()
bool_args = ["early_stopping", "tabular"]
config = dotdict({k : strtobool(v) if k in bool_args else v for k,v in vars(args).items()})

S5p_switch = True
tabular_switch = config.tabular
tab_label = "SatDataOnly"
if tabular_switch == True:
    tab_label = "SatAndTabData"

print(tab_label)
bbebebebebebeb


# set internal parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

assert config.checkpoint in [True, None]
if config.checkpoint is True:
    checkpoint_name = "ImageNet"
else: checkpoint_name = "FromScratch"

experiment = "_".join([datetime.today().strftime('%Y-%m-%d-%H:%M'), checkpoint_name])

# generate output folder and ID
experiment_folder_name = "_".join([datetime.today().strftime('%Y%m%d_%H%M'),str(config.epochs),"epochs","no2o3pm10",tab_label])
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
            r2_epoch_no2 = []
            r2_epoch_o3 = []
            r2_epoch_pm10 = []

            mae_epoch = []
            mae_epoch_no2 = []
            mae_epoch_o3 = []
            mae_epoch_pm10 = []

            mse_epoch = []
            mse_epoch_no2 = []
            mse_epoch_o3 = []
            mse_epoch_pm10 = []

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

                y_1 = sample["no2"].float()
                y_2 = sample["o3"].float()
                y_3 = sample["pm10"].float()

                y_samples = [y_1, y_2, y_3]
                loss_batch, metric_results = step(model_input, y_samples, model, loss, optimizer)

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
            val_y, val_y_hat = test( model, dataloader_val, device, datastats)
            valid_val = (val_y_hat["no2"] > 0)&(val_y_hat["no2"] < 100)&\
                        (val_y_hat["o3"] > 0)&(val_y_hat["o3"] < 150)&\
                        (val_y_hat["pm10"] > 0)&(val_y_hat["pm10"] < 100)

            eval_val_no2 = eval_metrics(val_y["no2"], val_y_hat["no2"])
            eval_val_o3 = eval_metrics(val_y["o3"], val_y_hat["o3"])
            eval_val_pm10 = eval_metrics(val_y["pm10"], val_y_hat["pm10"])
            eval_val = {"no2":eval_val_no2,"o3":eval_val_o3,"pm10":eval_val_pm10}
            #print("Fraction of valid estimates:", sum(valid_val)/len(valid_val))

            #stop training if evaluation performance does not increase
            # if epoch > 25 and sum(valid_val) > len(valid_val) - 5:
            #     if eval_val[0] > np.mean([performances_val[-3][2], performances_val[-2][2], performances_val[-1][2]]):
            #         # performance on evaluation set is decreasing
            #         print(f"Early stop at epoch: {epoch}")
            #         mlflow.log_param("early_stop_epoch", epoch)
            #         break

            # log results
            print(f"Epoch: {epoch} 'no2; r2,mae,mse' {eval_val['no2']}")
            print(f"Epoch: {epoch} 'o3; r2,mae,mse' {eval_val['o3']}")
            print(f"Epoch: {epoch} 'pm10; r2,mae,mse' {eval_val['pm10']}")

            performances_val.append([run, epoch] + eval_val["no2"] + eval_val["o3"] + eval_val["pm10"] )
            mlflow.log_metrics({"val_r2_epoch_no2" : eval_val["no2"][0],"val_mae_epoch_no2" : eval_val["no2"][1],"val_mse_epoch_no2" : eval_val["no2"][2],
                                "val_r2_epoch_o3" : eval_val["o3"][0], "val_mae_epoch_o3" : eval_val["o3"][1],"val_mse_epoch_o3" : eval_val["o3"][2],
                                "val_r2_epoch_pm10": eval_val["pm10"][0],"val_mae_epoch_pm10": eval_val["pm10"][1],"val_mse_epoch_pm10": eval_val["pm10"][2]}, step=epoch)
            # mlflow.log_metrics({"train_loss_epoch" : loss_epoch,"train_r2_epoch_no2" : r2_train_epoch_no2,"train_r2_epoch_o3" : r2_train_epoch_o3,
            #                     "train_r2_epoch_co" : r2_train_epoch_co,"train_r2_epoch_so2" : r2_train_epoch_so2,"train_r2_epoch_pm10" : r2_train_epoch_pm10,
            #                     "train_mae_epoch_no2" : mae_train_epoch_no2,"train_mae_epoch_o3" : mae_train_epoch_o3,"train_mae_epoch_co" : mae_train_epoch_co,
            #                     "train_mae_epoch_so2" : mae_train_epoch_so2,"train_mae_epoch_pm10" : mae_train_epoch_pm10,
            #                     "train_mse_epoch_no2" : mse_train_epoch_no2,"train_mse_epoch_o3" : mse_train_epoch_o3,"train_mse_epoch_co" : mse_train_epoch_co,
            #                     "train_mse_epoch_so2" : mse_train_epoch_so2,"train_mse_epoch_pm10" : mse_train_epoch_pm10}, step=epoch)
            mlflow.log_metric("current_epoch", epoch, step=epoch)

        # test run statistics
        test_y, test_y_hat = test(model, dataloader_test, device, datastats)
        train_y, train_y_hat = test(model, dataloader_train_for_testing, device, datastats)

        valid_test = (test_y_hat["no2"] > 0)&(test_y_hat["o3"] > 0)&(test_y_hat["pm10"] > 0)
        valid_train = (train_y_hat["no2"] > 0)&(train_y_hat["o3"] > 0)&(train_y_hat["pm10"] > 0)

        eval_test_no2 = eval_metrics(test_y["no2"], test_y_hat["no2"])
        eval_test_o3 = eval_metrics(test_y["o3"], test_y_hat["o3"])
        eval_test_pm10 = eval_metrics(test_y["pm10"], test_y_hat["pm10"])
        eval_test = {"no2": eval_test_no2,"o3": eval_test_o3,"pm10": eval_test_pm10}

        eval_train_no2 = eval_metrics(train_y["no2"], train_y_hat["no2"])
        eval_train_o3 = eval_metrics(train_y["o3"], train_y_hat["o3"])
        eval_train_pm10 = eval_metrics(train_y["pm10"], train_y_hat["pm10"])
        eval_train = {"no2": eval_train_no2,"o3": eval_train_o3,"pm10": eval_train_pm10}

        test_plotter(output_directory,test_y, test_y_hat, train_y, train_y_hat)

        mlflow.log_metric("test_r2", eval_test_no2[0])
        mlflow.log_metric("test_mae", eval_test_no2[1])
        mlflow.log_metric("test_mse", eval_test_no2[2])

        performances_test.append([eval_test_no2[0],eval_test_no2[1],eval_test_no2[2],
                                  eval_test_o3[0],eval_test_o3[1],eval_test_o3[2],
                                  eval_test_pm10[0],eval_test_pm10[1],eval_test_pm10[2]])
        performances_train.append([eval_train_no2[0],eval_train_no2[1],eval_train_no2[2],
                                  eval_train_o3[0],eval_train_o3[1],eval_train_o3[2],
                                  eval_train_pm10[0],eval_train_pm10[1],eval_train_pm10[2]])

        mlflow.log_artifacts(output_directory) # log everything that was written to the artifacts directory

# set up performance dfs
performances_val = pd.DataFrame(performances_val, columns=["run", "epoch",
                                                           "r2_no2", "mae_no2", "mse_no2",
                                                           "r2_o3", "mae_o3", "mse_o3",
                                                           "r2_pm10", "mae_pm10", "mse_pm10"])
performances_test = pd.DataFrame(performances_test,columns=["r2_no2", "mae_no2", "mse_no2",
                                                           "r2_o3", "mae_o3", "mse_o3",
                                                           "r2_pm10", "mae_pm10", "mse_pm10"])
performances_train = pd.DataFrame(performances_train,columns=["r2_no2", "mae_no2", "mse_no2",
                                                           "r2_o3", "mae_o3", "mse_o3",
                                                           "r2_pm10", "mae_pm10", "mse_pm10"])

# save results
print("Writing results...")
performances_test.to_csv(os.path.join(output_directory, "_".join([str(checkpoint_name), "test", str(config.epochs), "epochs"]) + ".csv"), index=False)
performances_train.to_csv(os.path.join(output_directory, "_".join([str(checkpoint_name), "train", str(config.epochs), "epochs"]) + ".csv"), index=False)
performances_val.to_csv(os.path.join(output_directory, "_".join([str(checkpoint_name), "val", str(config.epochs), "epochs"]) + ".csv"), index=False)

# save the model
print("Writing model...")
torch.save(model.state_dict(), os.path.join(output_directory, "_".join([str(checkpoint_name), str(config.epochs), "epochs"]) + ".model"))
print("done.")
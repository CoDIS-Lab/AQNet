import os

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

import xarray as xr
import rioxarray

class NO2PredictionDataset(Dataset):

    def __init__(self, datadir, samples, transforms=None, station_imgs=None):
        self.datadir = datadir
        self.transforms = transforms
        self.station_imgs = station_imgs # dict of AirQualityStation -> S2 image
        self.samples = samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.station_imgs is not None:
            sample["img"] = self.station_imgs.get(sample["AirQualityStation"])
            
        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.samples)

    def display_sample(self, sample, title=None):
        img = sample["img"]
        band_data = self._normalize_for_display(img)
        fig, axs = plt.subplots(1, 2, figsize=(7,7))
        s2_ax = axs[0]
        s2_ax.imshow(band_data[:, :, [3,2,1]])
        s2_ax.set_title("Sentinel2 data")

        im = axs[1].imshow(sample["s5p"])
        axs[1].set_title("Sentinel-5P data")
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        if title is not None:
            fig.suptitle(title)

        plt.show()

    def _normalize_for_display(self, band_data):
        band_data = reshape_as_image(np.array(band_data))
        lower_perc = np.percentile(band_data, 2, axis=(0,1))
        upper_perc = np.percentile(band_data, 98, axis=(0,1))
        return (band_data - lower_perc) / (upper_perc - lower_perc)
import os
import glob
import time
import datetime
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pytensor
import pytensor.tensor as pt
from IPython.core.display import HTML

from models import model_utils


class Model(ABC):
    def __init__(self, data=None, coords=None, description=""):
        self.data = data
        self.coords = coords
        self.model = None
        self.trace = None
        self.alphabet = list("abcdefghjklmnopqrs")
        self.V = len(self.alphabet)
        self.mean_seq_len = 40
        self.mode_map = {}
        self.num_modes = 0
        self.description = description
        self.model_name = self.__class__.__name__
        self.sample_dir = "samples"

    @abstractmethod
    def build_model(self, x_0_obs=None, x_t_obs=None, lengths_obs=None, **kwargs):
        pass

    def _get_filename(self, description=""):
        desc = description or self.description
        desc_str = f"_{desc}" if desc else ""
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(self.sample_dir, f"{self.model_name}{desc_str}_{ts}.nc")

    def _get_latest_filepath(self, description=""):
        desc = description or self.description
        desc_str = f"_{desc}" if desc else ""
        pattern = os.path.join(self.sample_dir, f"{self.model_name}{desc_str}_*.nc")
        files = glob.glob(pattern)

        if not files:
            return None

        return max(files, key=os.path.getctime)

    def save(self, description=""):
        if self.trace is None:
            raise ValueError("No trace to save. Fit the model first.")

        os.makedirs(self.sample_dir, exist_ok=True)
        filepath = self._get_filename(description)
        self.trace.to_netcdf(filepath)
        return filepath

    def load(self, filepath=None, description=""):
        if filepath is None:
            filepath = self._get_latest_filepath(description)

        if filepath is None or not os.path.exists(filepath):
            raise FileNotFoundError("No matching trace file found to load.")

        self.trace = az.from_netcdf(filepath)
        return self.trace

    def fit(self, data_dict=None, draws=1000, tune=1000, use_latest=False, description="", backup=False, **kwargs):
        if use_latest:
            try:
                return self.load(description=description)
            except FileNotFoundError:
                pass

        if self.model is None or data_dict:
            data_dict = data_dict or {}
            self.build_model(**data_dict)

        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune, **kwargs)

        if not backup:
            self.save(description=description)

        return self.trace

    def visualize_model(self):
        return pm.model_to_graphviz(self.model)

    def prior_predict(self, n_samples=1):
        with self.model:
            samples = []
            for _ in range(n_samples):
                samples.append(pm.sample_prior_predictive(draws=1).prior)
        return samples

    def pitches(self, sample, reverse=True):
        pitch_0 = sample["x_0"].to_series()
        pitch_series = sample["x_t+1"].to_series()
        sequence = pd.concat([pitch_0, pitch_series])
        melody = "-".join(self.alphabet[pitch] for pitch in sequence)
        if reverse:
            melody = melody[::-1]
        return melody

    def show_melody(self, sample, reverse=True):
        return HTML(
            f"<div style='font-family:Volpiano;font-size:3em;padding:2em;display:flex;justify-content:center;align-items:center;'>{self.pitches(sample, reverse)}</div>"
        )

    def A(self, samples, heatmap=False):
        if heatmap:
            model_utils.plot_heatmap(samples.A.mean(chain=0, draw=0))
        else:
            return samples.prior.A.mean(chain=0, draw=0)

    def predict(self, points):
        pass
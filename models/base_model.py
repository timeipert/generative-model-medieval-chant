import pymc as pm
from pymc.pytensorf import collect_default_updates
import numpy as np
import pytensor
import pytensor.tensor as pt
from abc import ABC, abstractmethod
from models import model_utils
import pandas as pd
from IPython.core.display import HTML
import time
import datetime


class Model(ABC):
    def __init__(self, data=None, coords=None):
        self.data = data
        self.coords = coords
        self.model = None
        self.trace = None
        self.alphabet = list("abcdefghjklmnopqrs")
        self.V = len(self.alphabet)
        self.mean_seq_len = 40
        self.mode_map = {}
        self.num_modes = 0

    @abstractmethod
    def build_model(self, x_0_obs=None, x_t_obs=None, lengths_obs=None, **kwargs):
        pass

    def fit(self, data_dict=None, draws=1000, tune=1000, **kwargs):
        if self.model is None or data_dict:
            data_dict = data_dict or {}
            self.build_model(**data_dict)

        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune, **kwargs)
        return self.trace

    def normalize_modes(self, raw_m):
        unique_modes = sorted(list(set(raw_m)))
        mode_map = {mode: i for i, mode in enumerate(unique_modes)}
        self.mode_map = mode_map
        self.num_modes = len(unique_modes)

        return [mode_map[m] for m in raw_m]

    def prepare_data(self, raw_sequences, raw_m):
        clean_sequences = [self.number_sequence(seq) for seq in raw_sequences]
        valid_indices = self.get_valid_indices(clean_sequences)

        valid_sequences = [clean_sequences[i] for i in valid_indices][::-1]
        x_0_data, x_t_data, lengths_data = self.get_observation_data(valid_sequences)
        data_dict = {
            "x_0_obs": x_0_data,
            "x_t_obs": x_t_data,
            "lengths_obs": lengths_data
        }
        if raw_m is not None:
            normalized_m = self.normalize_modes(raw_m)
            valid_m = [normalized_m[i] for i in valid_indices][::-1]
            data_dict["m_obs"] = np.array(valid_m, dtype=np.int64)
            data_dict["num_modes"] = self.num_modes

        return data_dict

    def number_sequence(self, sequence):
        return [self.alphabet.index(pitch) for pitch in sequence if pitch in self.alphabet]

    def get_observation_data(self, sequences):
        lengths = [len(seq) - 1 for seq in sequences]
        max_len = max(lengths)
        num_seqs = len(sequences)

        x_0_data = np.array([seq[0] for seq in sequences], dtype=np.int64)
        x_t_data = np.full((max_len, num_seqs), -1, dtype=np.int64)

        for i, seq in enumerate(sequences):
            seq_len = lengths[i]
            x_t_data[:seq_len, i] = seq[1:]

        return x_0_data, x_t_data, np.array(lengths, dtype=np.int64)

    def get_valid_indices(self, clean_sequences):
        return [i for i, seq in enumerate(clean_sequences) if len(seq) > 1]

    def _save_trace(self, trace):
        ts = time.time()
        d = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        trace.to_netcdf(f"samples/sampled_model_{d}.nc")

    def fit_from_raw(self, raw_sequences, raw_m=None, draws=1000, tune=1000, backup=False, **kwargs):
        data_dict = self.prepare_data(raw_sequences, raw_m)
        trace = self.fit(data_dict=data_dict, draws=draws, tune=tune, **kwargs)

        if not backup:
            self._save_trace(trace)

        return trace

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
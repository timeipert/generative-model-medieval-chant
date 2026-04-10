import datetime
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import pymc as pm
import arviz as az
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
        self.sample_dir = Path("samples")

    @abstractmethod
    def build_model(self, x_0_obs=None, x_t_obs=None, lengths_obs=None, **kwargs):
        pass

    def _get_filename(self, description=""):
        active_desc = description or self.description
        desc_suffix = f"_{active_desc}" if active_desc else ""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        return self.sample_dir / f"{self.model_name}{desc_suffix}_{timestamp}.nc"

    def _get_latest_filepath(self, description=""):
        active_desc = description or self.description
        desc_suffix = f"_{active_desc}" if active_desc else ""
        pattern = f"{self.model_name}{desc_suffix}_*.nc"

        matching_files = list(self.sample_dir.glob(pattern))

        if not matching_files:
            return None

        return max(matching_files, key=lambda path: path.stat().st_ctime)

    def save(self, description=""):
        if self.trace is None:
            raise ValueError("No trace to save. Fit the model first.")

        self.sample_dir.mkdir(parents=True, exist_ok=True)
        filepath = self._get_filename(description)
        self.trace.to_netcdf(filepath)

        return str(filepath)

    def load(self, filepath=None, description=""):
        if filepath is None:
            filepath = self._get_latest_filepath(description)

        if filepath is None or not Path(filepath).exists():
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
            model_inputs = data_dict or {}
            self.build_model(**model_inputs)

        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                idata_kwargs={"log_likelihood": True},
                **kwargs
            )

        if not backup:
            self.save(description=description)

        return self.trace

    def visualize_model(self):
        return pm.model_to_graphviz(self.model)

    def prior_predict(self, n_samples=1):
        with self.model:
            idata = pm.sample_prior_predictive(draws=n_samples)

        prior_dataset = idata.prior
        return [prior_dataset.sel(draw=i, chain=0) for i in range(n_samples)]

    def pitches(self, sample, reverse=True):
        initial_pitch = sample["x_0"].to_series()
        sequence_pitches = sample["x_t+1"].to_series()
        full_sequence = pd.concat([initial_pitch, sequence_pitches])

        melody_string = "-".join(self.alphabet[pitch] for pitch in full_sequence)

        if reverse:
            return melody_string[::-1]

        return melody_string

    def show_melody(self, sample, reverse=True):
        melody_string = self.pitches(sample, reverse)
        container_style = (
            "font-family:Volpiano; "
            "font-size:3em; "
            "padding:2em; "
            "display:flex; "
            "justify-content:center; "
            "align-items:center;"
        )

        return HTML(f"<div style='{container_style}'>{melody_string}</div>")

    def A(self, samples, heatmap=False):
        target_dataset = samples.prior if hasattr(samples, "prior") else samples
        avg_matrix = target_dataset.A.mean(dim=["chain", "draw"])

        if heatmap:
            model_utils.plot_heatmap(avg_matrix)
        else:
            return avg_matrix

    def predict(self, points):
        pass
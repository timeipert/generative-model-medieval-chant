import pymc as pm
from pymc.pytensorf import collect_default_updates
import numpy as np
import pytensor
import pytensor.tensor as pt
from models import base_model


class Model1(base_model.Model):
    def __init__(self, data=None, coords=None):
        super().__init__(data=data, coords=coords)
        self.alpha = np.full(self.V, 1.5)
        self.beta = np.full((self.V, self.V), 1.5)

    @staticmethod
    def step_fn(t, state_tm1, A, end_t):
        state_t = pm.Categorical.dist(p=A[state_tm1])

        # variable length like here: https://discourse.pymc.io/t/using-discretemarkovchain-with-batch-dimensions-and-data-of-different-lengths/16611/2
        end_state = pm.DiracDelta.dist(-1, shape=state_t.shape)
        state_t = pt.where(t >= end_t, end_state, state_t)

        return state_t, collect_default_updates([state_t])

    @staticmethod
    def scan_dist(init_state, A, end_t, size):
        states, _ = pytensor.scan(
            fn=Model1.step_fn,
            sequences=[pt.arange(end_t.max())],
            outputs_info=[init_state],
            non_sequences=[A, end_t],
            strict=True,
        )
        return states

    def build_inference_model(self, x_0_obs, x_t_obs, lengths_obs, **kwargs):
        with pm.Model() as model:
            pi = pm.Dirichlet("pi", a=self.alpha)
            A = pm.Dirichlet("A", a=self.beta)

            pm.Categorical("x_0", p=pi, observed=x_0_obs)

            x_0_reshaped = pt.shape_padleft(x_0_obs)
            x_full = pt.concatenate([x_0_reshaped, x_t_obs], axis=0)
            x_tm1 = x_full[:-1]
            x_t = x_full[1:]

            # Mask for variable length
            t_idx = pt.arange(x_t_obs.shape[0])
            mask = t_idx[:, None] < lengths_obs[None, :]

            valid_x_tm1 = x_tm1[mask]
            valid_x_t = x_t[mask]

            pm.Categorical("transitions", p=A[valid_x_tm1], observed=valid_x_t)

        self.model = model
        return model



    def build_generative_model(self, **kwargs):
        with pm.Model() as model:
            pi = pm.Dirichlet("pi", a=self.alpha)
            A = pm.Dirichlet("A", a=self.beta)

            x_0 = pm.Categorical("x_0", p=pi)
            length = pm.Poisson("length", self.mean_seq_len)

            pm.CustomDist(
                "x_t+1",
                x_0,
                A,
                length,
                dist=Model1.scan_dist,
            )

        self.model = model
        return model

    def build_model(self, x_0_obs=None, x_t_obs=None, lengths_obs=None, **kwargs):
        if x_t_obs is not None:
            return self.build_inference_model(x_0_obs, x_t_obs, lengths_obs, **kwargs)
        else:
            return self.build_generative_model(**kwargs)
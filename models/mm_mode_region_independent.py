import pymc as pm
import numpy as np
import pytensor
import pytensor.tensor as pt
from models import base_model


class MarkovModeRegionIndependent(base_model.Model):
    def __init__(self, data=None, coords=None):
        super().__init__(data=data, coords=coords)
        self.alpha = np.ones(self.V)
        self.beta = np.ones((self.V, self.V))
        self.M = 8
        self.R = 3

    @staticmethod
    def step_fn(t, state_tm1, transition_p, end_t):
        V = transition_p.shape[-1]
        idx = pt.clip(state_tm1, 0, V - 1)

        if transition_p.ndim == 3:
            batch_idx = pt.arange(idx.shape[0])
            p = transition_p[batch_idx, idx]
        else:
            p = transition_p[idx]

        state_t = pm.Categorical.dist(p=p)

        end_state = pm.DiracDelta.dist(-1, shape=state_t.shape)
        state_t = pt.where(t >= end_t, end_state, state_t)

        return state_t, pm.pytensorf.collect_default_updates([state_t])

    @staticmethod
    def scan_dist(init_state, transition_p, end_t, size):
        states, _ = pytensor.scan(
            fn=MarkovModeRegionIndependent.step_fn,
            sequences=[pt.arange(end_t.max())],
            outputs_info=[init_state],
            non_sequences=[transition_p, end_t],
            strict=True,
        )
        return states

    def logp(self, x_tm1, x_t, mask, A, m_val, r_val):
        m_b = pt.alloc(m_val, x_tm1.shape[0], m_val.shape[0])
        r_b = pt.alloc(r_val, x_tm1.shape[0], r_val.shape[0])

        # Region, Mode, From_Pitch, To_Pitch
        step_logps = pt.log(A[r_b, m_b, x_tm1, x_t])

        return pt.sum(step_logps * mask)

    def _setup_priors(self):
        nu_global = pm.Dirichlet("nu_global", a=np.ones(self.M))
        kappa = pm.Exponential("kappa", 1.0) # variance from global nu...
        nu = pm.Dirichlet("nu", a=nu_global * kappa, shape=(self.R, self.M))

        # 4D Dirichlet Prior for Transition Matrices (R, M, V, V)
        beta_rm = np.tile(self.beta, (self.R, self.M, 1, 1))
        A = pm.Dirichlet("A", a=beta_rm)
        pi = pm.Dirichlet("pi", a=self.alpha)

        return nu, A, pi

    def _build_inference_components(self, pi, nu, A, x_0_obs, x_t_obs, lengths_obs, m_obs, region_obs):
        pm.Categorical("x_0", p=pi, observed=x_0_obs)

        x_0_reshaped = pt.shape_padleft(x_0_obs)
        x_full = pt.concatenate([x_0_reshaped, x_t_obs], axis=0)
        x_tm1 = x_full[:-1]
        x_t = x_full[1:]

        t_idx = pt.arange(x_t_obs.shape[0])
        mask = t_idx[:, None] < lengths_obs[None, :]

        pm.Categorical("m", p=nu[region_obs], observed=m_obs)

        seq_ll = self.logp(x_tm1, x_t, mask, A, m_obs, region_obs)
        pm.Potential("ll", seq_ll)

    def _build_generative_components(self, pi, nu, A, m_obs, region_obs):
        # 1. Handle Region Shape
        if region_obs is not None:
            r = pt.as_tensor_variable(region_obs)
            batch_shape = r.shape
        else:
            r = pm.Categorical("region", p=np.ones(self.R) / self.R)
            batch_shape = None

        # 2. Handle Mode Shape
        if m_obs is not None:
            m = pm.Categorical("m", p=nu[r], observed=m_obs)
        else:
            m = pm.Categorical("m", p=nu[r], shape=batch_shape)

        transition_p = pm.Deterministic("A_m", A[r, m])

        # 3. Handle Initial State & Length Shapes
        x_0 = pm.Categorical("x_0", p=pi, shape=batch_shape)
        length = pm.Poisson("length", self.mean_seq_len, shape=batch_shape)

        pm.CustomDist(
            "x_t+1",
            x_0,
            transition_p,
            length,
            dist=self.__class__.scan_dist,
        )

    def build_model(self, x_0_obs=None, x_t_obs=None, lengths_obs=None, m_obs=None, region_obs=None, **kwargs):
        with pm.Model() as model:
            nu, A, pi = self._setup_priors()

            if x_t_obs is not None:
                self._build_inference_components(pi, nu, A, x_0_obs, x_t_obs, lengths_obs, m_obs, region_obs)
            else:
                self._build_generative_components(pi, nu, A, m_obs, region_obs)

        self.model = model
        return model
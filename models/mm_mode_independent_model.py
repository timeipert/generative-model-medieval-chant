import pymc as pm
import numpy as np
import pytensor
import pytensor.tensor as pt
from models import base_model


class MarkovModeIndependentModel(base_model.Model):
    def __init__(self, data=None, coords=None):
        super().__init__(data=data, coords=coords)
        self.alpha = np.ones(self.V)
        self.beta = np.ones((self.V, self.V))
        self.M = 8

    @staticmethod
    def step_fn(t, state_tm1, transition_p, end_t):
        V = transition_p.shape[0]
        idx = pt.clip(state_tm1, 0, V - 1)

        state_t = pm.Categorical.dist(p=transition_p[idx])

        # variable length like here: https://discourse.pymc.io/t/using-discretemarkovchain-with-batch-dimensions-and-data-of-different-lengths/16611/2
        end_state = pm.DiracDelta.dist(-1, shape=state_t.shape)
        state_t = pt.where(t >= end_t, end_state, state_t)

        return state_t, pm.pytensorf.collect_default_updates([state_t])

    @staticmethod
    def scan_dist(init_state, transition_p, end_t, size):
        states, _ = pytensor.scan(
            fn=MarkovModeIndependentModel.step_fn,
            sequences=[pt.arange(end_t.max())],
            outputs_info=[init_state],
            non_sequences=[transition_p, end_t],
            strict=True,
        )
        return states

    def _calc_conditional_logp(self, x_tm1, x_t, mask, A, m_val):
        m_b = pt.alloc(m_val, x_tm1.shape[0], m_val.shape[0])
        step_logps = pt.log(A[m_b, x_tm1, x_t])

        return pt.sum(step_logps * mask)


    def build_model(self, x_0_obs=None, x_t_obs=None, lengths_obs=None, m_obs=None, **kwargs):
        with pm.Model() as model:
            nu = pm.Dirichlet("nu", a=np.ones(self.M))

            beta_m = np.tile(self.beta, (self.M, 1, 1))
            A = pm.Dirichlet("A", a=beta_m)
            pi = pm.Dirichlet("pi", a=self.alpha)

            if x_t_obs is not None:
                pm.Categorical("x_0", p=pi, observed=x_0_obs)

                x_0_reshaped = pt.shape_padleft(x_0_obs)
                x_full = pt.concatenate([x_0_reshaped, x_t_obs], axis=0)
                x_tm1 = x_full[:-1]
                x_t = x_full[1:]

                t_idx = pt.arange(x_t_obs.shape[0])
                mask = t_idx[:, None] < lengths_obs[None, :]

                pm.Categorical("m", p=nu, observed=m_obs)
                seq_ll = self._calc_conditional_logp(x_tm1, x_t, mask, A, m_obs)
                pm.Potential("conditional_ll", seq_ll)


            else:
                if m_obs is not None:
                    m = pm.Categorical("m", p=nu, observed=m_obs)
                else:
                    m = pm.Categorical("m", p=nu)

                transition_p = pm.Deterministic("A_m", A[m])
                x_0 = pm.Categorical("x_0", p=pi)
                length = pm.Poisson("length", self.mean_seq_len)

                pm.CustomDist(
                    "x_t+1",
                    x_0,
                    transition_p,
                    length,
                    dist=MarkovModeIndependentModel.scan_dist,
                )

        self.model = model
        return model
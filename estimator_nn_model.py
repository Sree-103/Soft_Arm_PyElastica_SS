"""The neural estimator's ARCHITECTURE and inference path -- SHARED by training and runtime.

SAME REASON estimator_features.py EXISTS. If the training script and the environment each
defined their own MLP, a silent mismatch -- a layer reordered, dropout inserted somewhere
else, the target rescaling forgotten -- would load without error (state_dict keys would
still line up for many such changes) and quietly produce wrong masses. Both sides import
this file, so there is exactly one definition.

THREE THINGS THE RUNTIME MUST NOT GET WRONG, all handled here:
  1. every ensemble member is averaged -- using one member throws away most of the benefit,
     since ensembling is what made the network competitive in the first place;
  2. each member is rescaled by ITS OWN y_mu/y_sd, because members were fitted on different
     inner splits and so standardised the target slightly differently;
  3. eval() + inference_mode(), or dropout stays active and the estimate becomes random.

TORCH THREADS ARE PINNED TO 1. The estimator runs inside training environments, of which
many run in parallel. Torch defaults to one thread per core in EVERY process, so N workers
would each try to use all cores and spend their time fighting each other. The network is
29->32->1, eight times: it is microseconds of arithmetic and wants exactly one thread.
"""
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Must stay byte-compatible with the state dicts train_estimator_nn.py saves."""

    def __init__(self, d_in, hidden, dropout):
        super().__init__()
        L, p = [], d_in
        for h in hidden:
            L += [nn.Linear(p, h), nn.ReLU()]
            if dropout > 0:
                L += [nn.Dropout(dropout)]
            p = h
        L += [nn.Linear(p, 1)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load(path, feature_names=None):
    """Load a saved ensemble. Returns a bundle for predict_pct."""
    torch.set_num_threads(1)
    d = torch.load(path, map_location="cpu", weights_only=False)
    if feature_names is not None:
        assert list(d["feature_names"]) == list(feature_names), (
            "the neural estimator was trained on a different feature set than "
            "estimator_features.py now defines -- refusing to run, it would be fed inputs "
            "it never saw."
        )
    hp = d["hp"]
    members = []
    for m in d["members"]:
        net = MLP(int(d["d_in"]), tuple(hp["hidden"]), float(hp["dropout"]))
        net.load_state_dict(m["state"])
        net.eval()                      # dropout OFF; without this the estimate is random
        members.append((net, float(m["y_mu"]), float(m["y_sd"])))
    return {
        "members": members,
        "mu": np.asarray(d["mu"], dtype=np.float64),
        "sd": np.asarray(d["sd"], dtype=np.float64),
        "hp": hp,
        "n_members": len(members),
        "held_out_mae_pp": float(d.get("held_out_mae_pp", float("nan"))),
    }


def predict_pct(bundle, feats):
    """Feature vector (29,) or matrix (N,29) -> payload as PERCENT of rod mass."""
    f = np.atleast_2d(np.asarray(feats, dtype=np.float64))
    z = torch.tensor((f - bundle["mu"]) / bundle["sd"], dtype=torch.float32)
    with torch.inference_mode():
        p = np.mean([(net(z) * ys + ym).numpy()
                     for net, ym, ys in bundle["members"]], axis=0)
    return float(p[0]) if np.ndim(feats) == 1 else p

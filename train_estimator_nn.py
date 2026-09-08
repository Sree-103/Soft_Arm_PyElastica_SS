"""Neural-network payload estimator -- trained properly, tuned, and honestly scored.

WHAT THIS IS. A drop-in alternative to the ridge estimator: same 29 features, same target
(payload as PERCENT of rod mass), same output contract, so it can fill observation entry 56
without anything else in the pipeline changing shape.

WHY THIS SCRIPT IS NOT JUST "nn.Sequential + Adam". A first, careless MLP lost to ridge by
34% on this dataset, and the reason was never the architecture: there are 2040 rows but only
204 CONFIGURATIONS, and the ten payloads recorded at one arm pose are near-duplicates. The
effective sample size for generalising to a NEW pose is therefore ~204, and a 6k-parameter
network against 204 independent samples memorises poses. Everything below exists to fight
that specific problem:

    target standardisation  a network predicting 0..45 with a default init starts with a
                            huge bias error and wastes early epochs fixing scale
    Huber loss              the metric is MAE, so optimising plain MSE optimises the wrong
                            thing and lets a few hard configurations dominate the gradient
    input noise             Gaussian jitter on standardised features each batch. On small
                            tabular data this is the single most effective regulariser
                            available -- it is ridge-like smoothing applied through the
                            data rather than the weights
    dropout + weight decay  tuned, not guessed
    cosine LR schedule      lets the net anneal into a minimum instead of bouncing
    early stopping          on configuration-held-out validation
    ENSEMBLING              N independently seeded members, each with its OWN inner split,
                            averaged. Ensembling is the reliable win on small data and it
                            also removes the seed-to-seed variance that would otherwise
                            make the reported accuracy depend on a lucky run

HYPERPARAMETER SEARCH, and the trap it avoids. Selection uses ONLY the training payloads,
and its validation split holds out CONFIGURATIONS **and** PAYLOADS {15, 35} together, so the
selection criterion mimics the deployment condition -- unseen mass in an unseen pose. Tuning
against a validation set that shares payloads with training would pick hyperparameters that
interpolate poses well and generalise to new masses badly, which is the failure that matters
here. The held-out payloads {10, 20, 30, 40} are touched exactly once, at the end.

Everything is overridable, so the search can be skipped and a configuration pinned:
    TRIALS=40 ENSEMBLE=8 EPOCHS=800 python train_estimator_nn.py
    HIDDEN=64,64 LR=1e-3 WD=1e-3 DROPOUT=0.1 NOISE=0.1 TRIALS=0 python train_estimator_nn.py

Usage:  python train_estimator_nn.py [data_glob] [out.pt]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import glob
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import estimator_features as ef

torch.set_num_threads(int(os.environ.get("THREADS", "4")))

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/calib/shard*.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/estimator_nn.pt"
TRIALS = int(os.environ.get("TRIALS", "40"))        # 0 = skip search, use the pinned config
ENSEMBLE = int(os.environ.get("ENSEMBLE", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "800"))
PATIENCE = int(os.environ.get("PATIENCE", "80"))
OUTER_FOLDS = int(os.environ.get("FOLDS", "8"))

TRAIN_PCTS = {0., 5., 15., 25., 35., 45.}
TEST_PCTS = {10., 20., 30., 40.}
SEL_PCTS = {15., 35.}                 # held out INSIDE the search, to mimic unseen masses

# Pinned configuration, used when TRIALS=0 and as the search's starting point.
PINNED = dict(
    hidden=tuple(int(x) for x in os.environ.get("HIDDEN", "64,64").split(",")),
    lr=float(os.environ.get("LR", "1e-3")),
    wd=float(os.environ.get("WD", "1e-3")),
    dropout=float(os.environ.get("DROPOUT", "0.1")),
    noise=float(os.environ.get("NOISE", "0.10")),
    batch=int(os.environ.get("BATCH", "64")),
)

# ------------------------------------------------------------------ data
F, Y, C, G, ROD, META = [], [], [], [], None, None
for f in sorted(glob.glob(DATA)):
    d = np.load(f)
    F.append(ef.extract_batch(d))
    Y.append(np.maximum(np.asarray(d["mass_pct"], float), 0.0))
    C.append(np.asarray(d["seed"], np.int64))
    G.append(np.asarray(d["regime"], np.int64))
    if ROD is None:
        ROD = float(d["meta_rod_mass"])
        META = {k: float(d[k]) for k in d.files if k.startswith("meta_")}
X = np.vstack(F); Y = np.concatenate(Y); C = np.concatenate(C); G = np.concatenate(G)
keep = G != 2                         # never queried at runtime; see train_estimator.py
X, Y, C = X[keep], Y[keep], C[keep]
PP2KG = ROD / 100.0
D_IN = X.shape[1]
is_tr_m = np.isin(Y, sorted(TRAIN_PCTS))
is_te_m = np.isin(Y, sorted(TEST_PCTS))
CFGS = np.unique(C)
print(f"{len(X)} samples, {len(CFGS)} configurations, {D_IN} features, rod {ROD:.3f} kg")


# ------------------------------------------------------------------ model
# Imported, NOT redefined. The runtime builds its networks from this same class, so a local
# copy here could drift out of step with it and load mismatched weights without erroring.
from estimator_nn_model import MLP


def fit(Ztr, ytr, Zva, yva, hp, seed, epochs=EPOCHS, patience=PATIENCE):
    """One member. Returns the model plus the (mu, sd) used to standardise the TARGET."""
    torch.manual_seed(seed)
    ym, ys = float(ytr.mean()), float(ytr.std()) + 1e-9
    m = MLP(Ztr.shape[1], hp["hidden"], hp["dropout"])
    opt = torch.optim.AdamW(m.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.tensor(Ztr, dtype=torch.float32)
    yt = torch.tensor((ytr - ym) / ys, dtype=torch.float32)
    Xv = torch.tensor(Zva, dtype=torch.float32)
    yv = torch.tensor(yva, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    n, bs = len(Xt), hp["batch"]
    best, state, bad = np.inf, None, 0
    for _ in range(epochs):
        m.train()
        idx = torch.randperm(n, generator=g)
        for b in range(0, n, bs):
            s = idx[b:b + bs]
            xb = Xt[s]
            if hp["noise"] > 0:                 # augmentation, in standardised units
                xb = xb + hp["noise"] * torch.randn(xb.shape, generator=g)
            opt.zero_grad()
            nn.functional.smooth_l1_loss(m(xb), yt[s], beta=0.5).backward()
            opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            v = float(nn.functional.l1_loss(m(Xv) * ys + ym, yv))
        if v < best - 1e-4:
            best, bad = v, 0
            state = {k: t.clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if state is not None:
        m.load_state_dict(state)
    m.eval()
    return m, ym, ys


def predict(members, Z):
    """Ensemble mean, in PERCENT of rod mass."""
    Zt = torch.tensor(Z, dtype=torch.float32)
    with torch.no_grad():
        return np.mean([(m(Zt) * ys + ym).numpy() for m, ym, ys in members], axis=0)


def split_by_cfg(c, frac, seed):
    u = np.unique(c)
    r = np.random.default_rng(seed).permutation(len(u))
    held = set(u[r[:max(2, int(frac * len(u)))]])
    return np.array([x in held for x in c])


def standardise(Xtr, Xot):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    return (Xtr - mu) / sd, (Xot - mu) / sd, mu, sd


def ridge_ref(Ztr, ytr, Zte, lam=3.0):
    A = np.c_[Ztr, np.ones(len(Ztr))]
    c = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
    return np.c_[Zte, np.ones(len(Zte))] @ c


# ------------------------------------------------ stage 1: hyperparameter search
SPACE = dict(
    hidden=[(32,), (64,), (32, 32), (64, 64), (128, 64), (64, 64, 64), (128, 128)],
    lr=[3e-4, 1e-3, 3e-3],
    wd=[1e-5, 1e-4, 1e-3, 1e-2],
    dropout=[0.0, 0.1, 0.2, 0.3],
    noise=[0.0, 0.05, 0.10, 0.20, 0.30],
    batch=[32, 64, 128],
)

best_hp = dict(PINNED)
if TRIALS > 0:
    # SELECTION SET: configurations held out AND payloads {15, 35} held out, together.
    # Both must be unseen or the search optimises the wrong kind of generalisation.
    sel_va_cfg = split_by_cfg(C, 0.20, seed=11)
    sel_tr = is_tr_m & ~sel_va_cfg & ~np.isin(Y, sorted(SEL_PCTS))
    sel_va = is_tr_m & sel_va_cfg & np.isin(Y, sorted(SEL_PCTS))
    Ztr, Zva, _, _ = standardise(X[sel_tr], X[sel_va])
    ytr, yva = Y[sel_tr], Y[sel_va]
    # inner early-stopping split, carved out of the search's own training rows
    es = split_by_cfg(C[sel_tr], 0.15, seed=12)
    print(f"\nsearch: fit {int(sel_tr.sum())} rows, select on {int(sel_va.sum())} rows "
          f"(payloads {sorted(SEL_PCTS)} in unseen poses)")
    ref = ridge_ref(Ztr[~es], ytr[~es], Zva)
    print(f"ridge reference on this selection set: {np.abs(ref - yva).mean():.3f} pp\n")

    rng = np.random.default_rng(3)
    results = []
    t0 = time.time()
    for t in range(TRIALS):
        hp = (dict(PINNED) if t == 0 else
              {k: v[rng.integers(len(v))] for k, v in SPACE.items()})
        hp["hidden"] = tuple(hp["hidden"])
        try:
            mems = [fit(Ztr[~es], ytr[~es], Ztr[es], ytr[es], hp, seed=s, epochs=400,
                        patience=50) for s in range(2)]
            e = float(np.abs(predict(mems, Zva) - yva).mean())
        except Exception as exc:                      # a bad LR can produce non-finite loss
            e = np.inf
            print(f"  trial {t:2d} FAILED: {exc}")
        improved = all(e < r[0] for r in results)
        results.append((e, hp))
        if improved:
            print(f"  trial {t:2d}  {e:6.3f} pp  <- best   {hp}", flush=True)
        elif t % 5 == 0:
            print(f"  trial {t:2d}  {e:6.3f} pp", flush=True)
    results.sort(key=lambda r: r[0])
    best_hp = results[0][1]
    print(f"\nsearch done in {time.time()-t0:.0f}s over {TRIALS} trials")
    print(f"BEST {results[0][0]:.3f} pp  {best_hp}")
    print("runner-up configurations:")
    for e, hp in results[1:4]:
        print(f"  {e:6.3f} pp  {hp}")
else:
    print(f"\nsearch skipped (TRIALS=0), using pinned config: {best_hp}")

# --------------------------- stage 2: honest score on the held-out payloads
print(f"\nouter evaluation: {OUTER_FOLDS} folds by configuration, "
      f"ensemble of {ENSEMBLE}, scored on payloads {sorted(TEST_PCTS)}")
perm = np.random.default_rng(0).permutation(len(CFGS))
fold_of = {c: perm[i] % OUTER_FOLDS for i, c in enumerate(CFGS)}
FOLD = np.array([fold_of[c] for c in C])

P_nn = np.full(len(Y), np.nan)
P_rg = np.full(len(Y), np.nan)
t0 = time.time()
for f in range(OUTER_FOLDS):
    held = FOLD == f
    tr, te = (~held) & is_tr_m, held & is_te_m
    if not te.any():
        continue
    Ztr, Zte, _, _ = standardise(X[tr], X[te])
    ytr = Y[tr]
    # each ensemble member gets a DIFFERENT inner split as well as a different seed, so the
    # members disagree for two reasons rather than one -- a more diverse, stronger ensemble
    mems = []
    for s in range(ENSEMBLE):
        es = split_by_cfg(C[tr], 0.15, seed=100 + s)
        mems.append(fit(Ztr[~es], ytr[~es], Ztr[es], ytr[es], best_hp, seed=s))
    P_nn[te] = predict(mems, Zte)
    P_rg[te] = ridge_ref(Ztr, ytr, Zte)
    print(f"  fold {f+1}/{OUTER_FOLDS}  NN {np.abs(P_nn[te]-Y[te]).mean():.3f} pp   "
          f"ridge {np.abs(P_rg[te]-Y[te]).mean():.3f} pp", flush=True)

m = is_te_m & np.isfinite(P_nn)
e_nn, e_rg = P_nn[m] - Y[m], P_rg[m] - Y[m]
print(f"\n{'':22s} {'MAE pp':>8} {'MSE pp2':>9} {'MAE kg':>9}")
print(f"{'neural net (ensemble)':22s} {np.abs(e_nn).mean():8.3f} {(e_nn**2).mean():9.3f} "
      f"{np.abs(e_nn).mean()*PP2KG:9.4f}")
print(f"{'ridge (deployed)':22s} {np.abs(e_rg).mean():8.3f} {(e_rg**2).mean():9.3f} "
      f"{np.abs(e_rg).mean()*PP2KG:9.4f}")
d = (np.abs(e_nn).mean() - np.abs(e_rg).mean()) / np.abs(e_rg).mean() * 100
print(f"\nnetwork is {d:+.1f}% vs ridge  "
      f"({'better' if d < -2 else 'worse' if d > 2 else 'no real difference'})")
print(f"\n{'payload':>8} {'=kg':>7} {'NN MAE':>8} {'ridge MAE':>10}")
for t in sorted(TEST_PCTS):
    k2 = m & (Y == t)
    print(f"{t:7.0f}% {t*PP2KG:6.3f} {np.abs(P_nn[k2]-Y[k2]).mean():8.3f} "
          f"{np.abs(P_rg[k2]-Y[k2]).mean():10.3f}")
print(f"\nouter evaluation took {time.time()-t0:.0f}s")

# Save the OUT-OF-FOLD predictions so the plots are drawn from the same cross-validated
# numbers reported above, rather than from a re-fit that would quietly be optimistic.
PRED_OUT = os.environ.get("PRED_OUT", "data/estimator_nn_preds.npz")
np.savez(PRED_OUT, P_nn=P_nn, P_rg=P_rg, Y=Y, C=C, is_te_m=is_te_m, is_tr_m=is_tr_m,
         rod_mass=np.float64(ROD), hp=np.array(json.dumps(
             {**best_hp, "hidden": list(best_hp["hidden"])})))
print(f"saved out-of-fold predictions -> {PRED_OUT}")

# ------------------------- stage 3: the deployed ensemble, fitted on everything
print(f"\nfitting the DEPLOYED ensemble on all {len(X)} rows "
      f"(the numbers above describe this configuration, not these exact weights)")
mu, sd = X.mean(0), X.std(0) + 1e-12
Z = (X - mu) / sd
members = []
for s in range(ENSEMBLE):
    es = split_by_cfg(C, 0.15, seed=200 + s)
    mdl, ym, ys = fit(Z[~es], Y[~es], Z[es], Y[es], best_hp, seed=s)
    members.append({"state": mdl.state_dict(), "y_mu": ym, "y_sd": ys})
torch.save({
    "members": members, "hp": {**best_hp, "hidden": list(best_hp["hidden"])},
    "mu": mu, "sd": sd, "d_in": D_IN,
    "feature_names": list(ef.FEATURE_NAMES), "n_samples": int(len(X)),
    "rod_mass": ROD, "meta": META,
    "held_out_mae_pp": float(np.abs(e_nn).mean()),
    "held_out_mse_pp2": float((e_nn ** 2).mean()),
}, OUT)
print(f"saved -> {OUT}")
print("  output is PERCENT OF ROD MASS; divide by 100 for the observation entry")
print("  runtime MUST use estimator_features.extract() and this mu/sd, average ALL members,")
print("  and rescale each member by its own y_mu/y_sd, or it is fed something it never saw")
print(f"\nhyperparameters: {json.dumps({**best_hp, 'hidden': list(best_hp['hidden'])})}")

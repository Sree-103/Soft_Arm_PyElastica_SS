"""Feature extraction for the payload estimator -- SHARED by training and runtime.

THIS FILE EXISTS TO PREVENT ONE SPECIFIC BUG. The estimator is fitted offline on recorded
probes and then used online inside the environment. If those two paths computed features
even slightly differently -- a different window, a different guard, a different order --
the model would be fed something it was never trained on and would fail silently, with no
error and no obvious symptom. So both paths import THIS function and nothing else.

ARM-SIDE INPUTS ONLY. Everything here is something the ROBOT can sense about ITSELF --
its own deformation and motion. In particular the acceleration used is the TIP's, obtained
by differencing the tip velocity, NOT the payload's.

That distinction is not cosmetic. Using the payload's own acceleration looks attractive
because when the joint is rigid the two coincide, and it gave 2-5% error on loaded probes.
But it fails completely when nothing is attached: the payload sits anchored, its
acceleration is zero, ||a - g|| collapses to plain gravity, and the rod's flailing force
divided by 9.81 reports ~7.9 kg for an EMPTY arm. It is also information a real robot
cannot have -- no machine senses the acceleration of an object it is not holding -- so a
model leaning on it would never transfer off the simulator.

With the TIP's acceleration the physics still holds while loaded (the rigid joint makes tip
and payload move together) and degrades gracefully when empty: the internal force is then
only accelerating the rod's own tip mass, so the quotient tends to that constant and the
estimate falls to ~0 once it is subtracted.

INPUTS are exactly what one calibration probe produces, all raw:
    tip_p   (T,3)   rod tip position each step
    tip_v   (T,3)   rod tip velocity  -> differenced here for tip acceleration
    tip_dir (T,3,3) rod tip directors -> differenced here for tip angular velocity
    force   (T,3)   rod internal force at the tip
    kappa   (T,N)   curvature magnitude per element
    rod_ke  (T,)    rod kinetic energy
    shape0  (5,)    arm shape at probe start: reach, mean bend, peak bend, tip height,
                    tip speed

WHY EACH GROUP IS HERE, measured rather than assumed (cross-validated error on 60
configurations, linear model):
    response only ............... 14.5 pp
    + arm shape ................. 7.6 pp    pose disambiguates the response
    + force and acceleration .... 4.8 pp    the terms Newton's law actually uses
    + the F/(a-g) ratio ......... 3.6 pp    the physics quotient, which a linear model
                                            cannot form on its own from F and a
Dropping any group makes it worse; the ratio matters because m = ||F|| / ||a - g||, and no
linear combination of F and a can express a division.
"""
import numpy as np

GRAVITY = np.array([0.0, -9.80665, 0.0])
# Guard on the denominator of the physics ratio. ||a - g|| is small when the payload is in
# near-free-fall, and dividing by it there produces huge, meaningless values that would
# dominate the mean. 1.0 m/s^2 keeps only well-conditioned samples; if fewer than
# MIN_RATIO_SAMPLES survive, the ratio features fall back to 0 and the model must rely on
# the rest, which it can (4.8 pp without them).
RATIO_DENOM_MIN = 1.0
MIN_RATIO_SAMPLES = 8

FEATURE_NAMES = [
    # --- response to the routine ---
    "disp_max", "disp_rms", "disp_final", "tipspeed_max", "tipspeed_rms",
    "spec_peak_bin", "spec_peak_mag",
    # --- physics terms (Newton's law), all from the TIP ---
    "force_mean", "force_max", "force_rms", "accdev_mean", "accdev_max",
    # --- rotation of the tip: the payload sits on a lever arm, so angular motion enters
    #     the force the tip must transmit and cannot be ignored ---
    "omega_mean", "omega_max", "alpha_mean", "alpha_max",
    # --- the quotient itself ---
    "ratio_median", "ratio_mean", "ratio_std",
    # --- deformation ---
    "kappa_mean", "kappa_max", "kappa_delta", "ke_mean", "ke_max",
    # --- pose at probe start ---
    "shape_reach", "shape_bend_mean", "shape_bend_max", "shape_tip_y", "shape_tip_speed",
]


def extract(tip_p, tip_v, tip_dir, force, kappa, rod_ke, shape0, dt):
    """One probe -> one fixed-length feature vector. Deterministic, no fitted state."""
    tip_p = np.asarray(tip_p, dtype=np.float64)
    tip_v = np.asarray(tip_v, dtype=np.float64)
    tip_dir = np.asarray(tip_dir, dtype=np.float64)
    force = np.asarray(force, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    rod_ke = np.asarray(rod_ke, dtype=np.float64).ravel()
    shape0 = np.asarray(shape0, dtype=np.float64).ravel()

    # displacement is measured FROM THE PROBE'S FIRST FRAME, not from the origin, so the
    # feature describes the response to the routine rather than where the arm happens to be
    d = np.linalg.norm(tip_p - tip_p[0], axis=1)
    v = np.linalg.norm(tip_v, axis=1)
    fmag = np.linalg.norm(force, axis=1)
    # TIP acceleration by finite difference of the tip velocity (arm-side only).
    a_tip = np.gradient(tip_v, dt, axis=0)
    adev = np.linalg.norm(a_tip - GRAVITY, axis=1)        # the ||a - g|| of F = m(a - g)
    # tip angular velocity from the director frames: omega_hat = dR/dt . R^T, read off the
    # skew part. The payload hangs on a lever arm, so tip rotation contributes to the force
    # the tip must transmit; without this the quotient is biased whenever the tip spins.
    dR = np.gradient(tip_dir, dt, axis=0)
    W = np.einsum("tij,tkj->tik", dR, tip_dir)
    omega = np.stack([W[:, 2, 1] - W[:, 1, 2],
                      W[:, 0, 2] - W[:, 2, 0],
                      W[:, 1, 0] - W[:, 0, 1]], axis=1) * 0.5
    omag = np.linalg.norm(omega, axis=1)
    amag = np.linalg.norm(np.gradient(omega, dt, axis=0), axis=1)

    spec = np.abs(np.fft.rfft(d - d.mean()))
    ok = adev > RATIO_DENOM_MIN
    if ok.sum() >= MIN_RATIO_SAMPLES:
        r = fmag[ok] / adev[ok]
        r_med, r_mean, r_std = np.median(r), r.mean(), r.std()
    else:
        r_med = r_mean = r_std = 0.0

    return np.array([
        d.max(), np.sqrt((d ** 2).mean()), d[-1], v.max(), np.sqrt((v ** 2).mean()),
        float(np.argmax(spec)), float(spec.max()),
        fmag.mean(), fmag.max(), np.sqrt((fmag ** 2).mean()), adev.mean(), adev.max(),
        omag.mean(), omag.max(), amag.mean(), amag.max(),
        r_med, r_mean, r_std,
        kappa.mean(), kappa.max(), float(kappa[-1].mean() - kappa[0].mean()),
        rod_ke.mean(), rod_ke.max(),
        shape0[0], shape0[1], shape0[2], shape0[3], shape0[4],
    ], dtype=np.float64)


def extract_batch(ds, idx=None):
    """Feature matrix from a loaded calibration .npz."""
    n = len(ds["mass_pct"])
    idx = range(n) if idx is None else idx
    dt = float(ds["dt"])
    return np.array([
        extract(ds["tip_p"][i], ds["tip_v"][i], ds["tip_dir"][i], ds["force"][i],
                ds["kappa"][i], ds["rod_ke"][i], ds["shape0"][i], dt) for i in idx
    ])

"""Neural estimators shared by all experiments: variant definitions, input features, predictors,
and rebuilding a trained model from its saved weights.

    from neural import VARIANTS, features, make_predictor, load_predictor
    predict = make_predictor("residual", net, lm, miss)      # after training
    est = predict(Y, snr_db)                                  # (n, |miss|) complex, withheld subcarriers

`snr_db` is the pilot SNR the receiver ASSUMES. SNR-aware variants get it as an input feature, and
the residual variant also uses it in its LMMSE base, so a noise-power mismatch reaches both in the
same way as it reaches the LMMSE baseline. The blind variant has the same input size but its SNR
input is always zero.
"""
import numpy as np

from mlp_numpy import MLP

# name: (hidden layer sizes, SNR input, learns a correction to the LMMSE estimate)
VARIANTS = {
    "mlp": ([128, 128], True, False),
    "mlp_relweight": ([128, 128], True, False),
    "residual": ([128, 128], True, True),
    "mlp_wide": ([256, 256, 256], True, False),
    "mlp_blind": ([128, 128], False, False),
}


def features(Y, snr_db, snr_input=True):
    """[Re y, Im y, snr/30]; the last column is zero for a blind model."""
    s = (np.asarray(snr_db, float) / 30.0) if snr_input else np.zeros(len(Y))
    return np.concatenate([Y.real, Y.imag, s[:, None]], axis=1)


def make_predictor(variant, net, lm, miss):
    """Closure (Y, snr_db) -> estimate of the withheld subcarriers; lm = LMMSE on training statistics."""
    _, snr_input, residual = VARIANTS[variant]
    M = len(miss)

    def predict(Y, s_vec):
        s_vec = np.asarray(s_vec, float)
        out = net.predict(features(Y, s_vec, snr_input))
        r = out[:, :M] + 1j * out[:, M:]
        if residual:
            r = r + lm.apply_per_sample(Y, 10 ** (-s_vec / 10))[:, miss]
        return r
    return predict


def load_predictor(path, lm, miss):
    """Rebuild the predictor saved by experiments/02 (MLP.save with variant / pilots metadata)."""
    net, meta = MLP.from_file(path)
    if list(meta["pilots"]) != [int(x) for x in lm.p]:
        raise ValueError(f"{path}: pilots {meta['pilots']} do not match the LMMSE pilots {list(lm.p)}")
    if list(meta["withheld"]) != [int(x) for x in miss]:
        raise ValueError(f"{path}: withheld subcarriers do not match")
    return make_predictor(meta["variant"], net, lm, miss), meta

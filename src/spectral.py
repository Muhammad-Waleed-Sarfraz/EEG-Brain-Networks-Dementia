import numpy as np

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}


def band_power_per_channel(epochs):
    """Relative band power per channel, shape (n_bands, n_channels)."""
    spec = epochs.compute_psd(method="welch", fmin=1, fmax=45, verbose=False)
    psd, freqs = spec.get_data(return_freqs=True)    # epochs x channels x freqs
    psd = psd.mean(axis=0)                            # average over epochs
    total = psd.sum(axis=1)                           # total power per channel, 1-45 Hz
    rel = []
    for name, (lo, hi) in BANDS.items():
        upper = (freqs <= hi) if name == "gamma" else (freqs < hi)
        mask = (freqs >= lo) & upper
        rel.append(psd[:, mask].sum(axis=1) / total)
    return np.array(rel)


def spectral_row(sub_id, epochs):
    """One summary row and one per-channel row for a subject."""
    rel = band_power_per_channel(epochs)
    row = {"participant_id": sub_id}
    chan_row = {"participant_id": sub_id}
    for (name, _), vals in zip(BANDS.items(), rel):
        row[f"relpow_{name}"] = vals.mean()
        for ch, v in zip(epochs.ch_names, vals):
            chan_row[f"{name}_{ch}"] = v
    row["theta_alpha_ratio"] = row["relpow_theta"] / row["relpow_alpha"]
    return row, chan_row

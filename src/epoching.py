import glob
import mne


def load_clean_epochs(sub_id, n_keep=60, reject_uv=300):
    """Load one subject, cut 4-s epochs, drop those above reject_uv or crossing a boundary."""
    path = glob.glob(f"data/ds004504/derivatives/{sub_id}/eeg/*.set")[0]
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    n_boundary = sum("boundary" in d.lower() for d in raw.annotations.description)
    raw.annotations.description[:] = [
        "BAD_boundary" if "boundary" in d.lower() else d for d in raw.annotations.description]
    epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True, verbose=False)
    n_total = len(epochs)
    epochs.drop_bad(reject=dict(eeg=reject_uv * 1e-6), verbose=False)
    n_clean = len(epochs)
    n_dropped_boundary = sum("BAD_boundary" in x for x in epochs.drop_log)
    epochs = epochs[:n_keep]
    log = dict(participant_id=sub_id, n_boundary_marks=n_boundary, n_total=n_total,
               n_clean=n_clean, n_dropped_boundary=n_dropped_boundary,
               n_kept=len(epochs), reject_uv=reject_uv)
    return epochs, log

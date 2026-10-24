import numpy as np
from mne_connectivity import spectral_connectivity_epochs

BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
FMIN = (1.25, 4, 8, 13, 30)   # delta starts at 1.25 Hz: 4-s epochs hold < 5 cycles at 1 Hz
FMAX = (4, 8, 13, 30, 45)


def wpli_matrices(epochs):
    """Return an array (5 bands, 19, 19) of symmetric wPLI matrices."""
    con = spectral_connectivity_epochs(epochs, method="wpli", mode="multitaper",
                                       fmin=FMIN, fmax=FMAX, faverage=True, verbose=False)
    m = con.get_data(output="dense")
    m = np.moveaxis(m, -1, 0)
    m = m + np.transpose(m, (0, 2, 1))
    for b in range(m.shape[0]):
        np.fill_diagonal(m[b], 0)
    return m
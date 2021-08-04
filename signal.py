import numpy as np
import pandas as pd
from warnings import warn
import scipy.signal as sig
import scipy.misc

from .misc import as_vector, find_closest
from .stats import standardize


class Signal:

    def __init__(self, data, fs, sig_type, sig_len, base_time, units):
        
        self.data = data
        self.fs = fs
        self.sig_len = sig_len
        self.base_time = base_time
        self.sig_type = sig_type
        self.units = units


    def signal_binarize(self, method="threshold", threshold="auto"):
        """Binarize a continuous signal.

        Convert a continuous signal into zeros and ones depending on a given threshold.

        Parameters
        ----------
        method : str
            The algorithm used to discriminate between the two states. Can be one of 'mixture' (default) or
            'threshold'. If 'mixture', will use a Gaussian Mixture Model to categorize between the two states.
            If 'threshold', will consider as activated all points which value is superior to the threshold.
        threshold : float
            If `method` is 'mixture', then it corresponds to the minimum probability required to be considered
            as activated (if 'auto', then 0.5). If `method` is 'threshold', then it corresponds to the minimum
            amplitude to detect as onset. If "auto", takes the value between the max and the min.
        """

        #===================================================================
        #                            METHODS
        #===================================================================

        def _signal_binarize_threshold(signal, threshold="auto"):
            if threshold == "auto":
                threshold = np.mean([np.max(signal), np.min(signal)])

            binary = np.zeros(len(signal))
            binary[signal > threshold] = 1
            return binary


        def _signal_binarize_mixture(signal, threshold="auto"):

            import sklearn.mixture

            if threshold == "auto":
                threshold = 0.5

            # fit a Gaussian Mixture Model with two components
            clf = sklearn.mixture.GaussianMixture(n_components=2, random_state=333)
            clf = clf.fit(signal.reshape(-1, 1))

            # Get predicted probabilities
            probability = clf.predict_proba(signal.reshape(-1, 1))[:, np.argmax(clf.means_[:, 0])]

            binary = np.zeros(len(signal))
            binary[probability >= threshold] = 1
            return binary


        #=================================================================
        #                           EXECUTION
        #=================================================================
        method = method.lower()  # remove capitalised letters
        if method == "threshold":
            binary = _signal_binarize_threshold(self.data, threshold=threshold)
        elif method == "mixture":
            binary = _signal_binarize_mixture(self.data, threshold=threshold)
        else:
            raise ValueError("SIGNAL ERROR: signal_binarize() - 'method' should be one of 'threshold' or 'mixture'.")
        return binary



    def signal_filter(self, lowcut=None, highcut=None, method="butterworth", order=2, window_size="default", powerline=50):
        """Filter a signal using 'butterworth', 'fir' or 'savgol' filters.

        Apply a lowpass (if 'highcut' frequency is provided), highpass (if 'lowcut' frequency is provided)
        or bandpass (if both are provided) filter to the signal.

        Parameters
        ----------
        lowcut : float
            Lower cutoff frequency in Hz. The default is None.
        highcut : float
            Upper cutoff frequency in Hz. The default is None.
        method : str
            Can be one of 'butterworth', 'fir', 'bessel' or 'savgol'. Note that for Butterworth, the function
            uses the SOS method from `scipy.signal.sosfiltfilt`, recommended for general purpose filtering.
            One can also specify "butterworth_ba' for a more traditional and legacy method (often implemented
            in other software).
        order : int
            Only used if method is 'butterworth' or 'savgol'. Order of the filter (default is 2).
        window_size : int
            Only used if method is 'savgol'. The length of the filter window (i.e. the number of coefficients).
            Must be an odd integer. If 'default', will be set to the sampling rate divided by 10
            (101 if the sampling rate is 1000 Hz).
        powerline : int
            Only used if method is 'powerline'. The powerline frequency (normally 50 Hz or 60 Hz).

        """

        #======================================================================
        #                              METHODS
        #======================================================================


        # Savitzky-Golay (savgol)
        def _signal_filter_savgol(signal, sampling_rate=1000, order=2, window_size="default"):
            """Filter a signal using the Savitzky-Golay method.

            Default window size is chosen based on `Sadeghi, M., & Behnia, F. (2018). Optimum window length of
            Savitzky-Golay filters with arbitrary order. arXiv preprint arXiv:1808.10489.
            <https://arxiv.org/ftp/arxiv/papers/1808/1808.10489.pdf>`_.

            """
            window_size = _signal_filter_windowsize(window_size=window_size, sampling_rate=sampling_rate)
            if window_size % 2 == 0:
                window_size += 1  # Make sure it's odd

            filtered = sig.savgol_filter(signal, window_length=int(window_size), polyorder=order)
            return filtered



        # FIR
        def _signal_filter_fir(signal, sampling_rate=1000, lowcut=None, highcut=None, window_size="default"):
            """Filter a signal using a FIR filter."""
            try:
                import mne
            except ImportError:
                raise ImportError(
                    "EPK error: signal_filter(): the 'mne' module is required for this method to run. ",
                    "Please install it first (`pip install mne`).",
                )

            if isinstance(window_size, str):
                window_size = "auto"

            filtered = mne.filter.filter_data(
                signal,
                sfreq=sampling_rate,
                l_freq=lowcut,
                h_freq=highcut,
                method="fir",
                fir_window="hamming",
                filter_length=window_size,
                l_trans_bandwidth="auto",
                h_trans_bandwidth="auto",
                phase="zero-double",
                fir_design="firwin",
                pad="reflect_limited",
                verbose=False,
            )
            return filtered




        # Butterworth
        def _signal_filter_butterworth(signal, sampling_rate=1000, lowcut=None, highcut=None, order=5):
            """Filter a signal using IIR Butterworth SOS method."""
            freqs, filter_type = _signal_filter_sanitize(lowcut=lowcut, highcut=highcut, sampling_rate=sampling_rate)

            sos = sig.butter(order, freqs, btype=filter_type, output="sos", fs=sampling_rate)
            filtered = sig.sosfiltfilt(sos, signal)
            return filtered


        def _signal_filter_butterworth_ba(signal, sampling_rate=1000, lowcut=None, highcut=None, order=5):
            """Filter a signal using IIR Butterworth B/A method."""
            # Get coefficients
            freqs, filter_type = _signal_filter_sanitize(lowcut=lowcut, highcut=highcut, sampling_rate=sampling_rate)

            b, a = sig.butter(order, freqs, btype=filter_type, output="ba", fs=sampling_rate)
            try:
                filtered = sig.filtfilt(b, a, signal, method="gust")
            except ValueError:
                filtered = sig.filtfilt(b, a, signal, method="pad")

            return filtered



        # Bessel
        def _signal_filter_bessel(signal, sampling_rate=1000, lowcut=None, highcut=None, order=5):
            freqs, filter_type = _signal_filter_sanitize(lowcut=lowcut, highcut=highcut, sampling_rate=sampling_rate)

            sos = sig.bessel(order, freqs, btype=filter_type, output="sos", fs=sampling_rate)
            filtered = sig.sosfiltfilt(sos, signal)
            return filtered




        # Powerline
        def _signal_filter_powerline(signal, sampling_rate, powerline=50):
            """Filter out 50 Hz powerline noise by smoothing the signal with a moving average kernel with the width of one
            period of 50Hz."""

            if sampling_rate >= 100:
                b = np.ones(int(sampling_rate / powerline))
            else:
                b = np.ones(2)
            a = [len(b)]
            y = sig.filtfilt(b, a, signal, method="pad")
            return y


        # =============================================================================
        #                                 UTILITIES
        # =============================================================================
        def _signal_filter_sanitize(lowcut=None, highcut=None, sampling_rate=1000, normalize=False):

            # Sanity checks
            if isinstance(highcut, int):
                if sampling_rate <= 2 * highcut:
                    warn(
                        "The sampling rate is too low. Sampling rate"
                        " must exceed the Nyquist rate to avoid aliasing problem."
                        f" In this analysis, the sampling rate has to be higher than {2 * highcut} Hz")

            # Replace 0 by none
            if lowcut is not None and lowcut == 0:
                lowcut = None
            if highcut is not None and highcut == 0:
                highcut = None

            # Format
            if lowcut is not None and highcut is not None:
                if lowcut > highcut:
                    filter_type = "bandstop"
                else:
                    filter_type = "bandpass"
                freqs = [lowcut, highcut]
            elif lowcut is not None:
                freqs = [lowcut]
                filter_type = "highpass"
            elif highcut is not None:
                freqs = [highcut]
                filter_type = "lowpass"

            # Normalize frequency to Nyquist Frequency (Fs/2).
            # However, no need to normalize if `fs` argument is provided to the scipy filter
            if normalize is True:
                freqs = np.array(freqs) / (sampling_rate / 2)

            return freqs, filter_type


        def _signal_filter_windowsize(window_size="default", sampling_rate=1000):
            if isinstance(window_size, str):
                window_size = int(np.round(sampling_rate / 3))
                if (window_size % 2) == 0:
                    window_size + 1  # pylint: disable=W0104
            return window_size



        # =============================================================================
        #                                EXECUTION
        # =============================================================================

        method = method.lower()


        if method in ["sg", "savgol", "savitzky-golay"]:
            filtered = _signal_filter_savgol(self.data, self.fs, order, window_size=window_size)
        elif method in ["powerline"]:
                filtered = _signal_filter_powerline(self.data, self.fs, powerline)
        else:

            # Sanity checks
            if lowcut is None and highcut is None:
                raise ValueError(
                    "EPK error: signal_filter(): you need to specify a 'lowcut' or a 'highcut'."
                )

            if method in ["butter", "butterworth"]:
                filtered = _signal_filter_butterworth(self.data, self.fs, lowcut, highcut, order)
            elif method in ["butter_ba", "butterworth_ba"]:
                filtered = _signal_filter_butterworth_ba(self.data, self.fs, lowcut, highcut, order)
            elif method in ["bessel"]:
                filtered = _signal_filter_bessel(self.data, self.fs, lowcut, highcut, order)
            elif method in ["fir"]:
                filtered = _signal_filter_fir(self.data, self.fs, lowcut, highcut, window_size=window_size)
            else:
                raise ValueError(
                    "SIGNAL ERROR: signal_filter() - 'method' should be",
                    " one of 'butterworth', 'butterworth_ba', 'bessel',",
                    " 'savgol' or 'fir'."
                )
        return filtered



    def signal_findpeaks(self, height_min=None, height_max=None, relative_height_min=None, relative_height_max=None, relative_mean=True, relative_median=False, relative_max=False ):
        """Find peaks in a signal.

        Locate peaks (local maxima) in a signal and their related characteristics, such as height (prominence),
        width and distance with other peaks.

        Parameters
        ----------
        height_min : float
            The minimum height (i.e., amplitude in terms of absolute values). For example,``height_min=20``
            will remove all peaks which height is smaller or equal to 20 (in the provided signal's values).
        height_max : float
            The maximum height (i.e., amplitude in terms of absolute values).
        relative_height_min : float
            The minimum height (i.e., amplitude) relative to the sample (see below). For example,
            ``relative_height_min=-2.96`` will remove all peaks which height lies below 2.96 standard
            deviations from the mean of the heights.
        relative_height_max : float
            The maximum height (i.e., amplitude) relative to the sample (see below).
        relative_mean : bool
            If a relative threshold is specified, how should it be computed (i.e., relative to what?).
            ``relative_mean=True`` will use Z-scores.
        relative_median : bool
            If a relative threshold is specified, how should it be computed (i.e., relative to what?).
            Relative to median uses a more robust form of standardization (see ``standardize()``).
        relative_max : bool
            If a relative threshold is specified, how should it be computed (i.e., relative to what?).
            Reelative to max will consider the maximum height as the reference.

        Returns
        ----------
        dict
            Returns a dict itself containing 5 arrays:
            - 'Peaks' contains the peaks indices (as relative to the given signal). For instance, the
            value 3 means that the third data point of the signal is a peak.
            - 'Distance' contains, for each peak, the closest distance with another peak. Note that these
            values will be recomputed after filtering to match the selected peaks.
            - 'Height' contains the prominence of each peak. See `scipy.signal.peak_prominences()`.
            - 'Width' contains the width of each peak. See `scipy.signal.peak_widths()`.
            - 'Onset' contains the onset, start (or left trough), of each peak.
            - 'Offset' contains the offset, end (or right trough), of each peak.

        
        """

        
        # =============================================================================
        #                                  METHODS
        # =============================================================================

        def _signal_findpeaks_keep(
            info, what="Height", below=None, above=None, relative_mean=False, relative_median=False, relative_max=False
        ):

            if below is None and above is None:
                return info

            keep = np.full(len(info["Peaks"]), True)

            if relative_max is True:
                what = info[what] / np.max(info[what])
            elif relative_median is True:
                what = standardize(info[what], robust=True)
            elif relative_mean is True:
                what = standardize(info[what])
            else:
                what = info[what]

            if below is not None:
                keep[what > below] = False
            if above is not None:
                keep[what < above] = False

            info = _signal_findpeaks_filter(info, keep)
            return info


        def _signal_findpeaks_filter(info, keep):
            for key in info.keys():
                info[key] = info[key][keep]

            return info


        # =============================================================================
        #                                 UTILITIES
        # =============================================================================


        def _signal_findpeaks_distances(peaks):

            if len(peaks) <= 2:
                distances = np.full(len(peaks), np.nan)
            else:
                distances_next = np.concatenate([[np.nan], np.abs(np.diff(peaks))])
                distances_prev = np.concatenate([np.abs(np.diff(peaks[::-1])), [np.nan]])
                distances = np.array([np.nanmin(i) for i in list(zip(distances_next, distances_prev))])

            return distances


        def _signal_findpeaks_findbase(peaks, signal, what="onset"):
            if what == "onset":
                direction = "smaller"
            else:
                direction = "greater"

            troughs, _ = scipy.signal.find_peaks(-1 * signal)

            bases = find_closest(peaks, troughs, direction=direction, strictly=True)
            bases = as_vector(bases)

            return bases


        def _signal_findpeaks_scipy(signal):
            peaks, _ = scipy.signal.find_peaks(signal)

            # Get info
            distances = _signal_findpeaks_distances(peaks)
            heights, _, __ = scipy.signal.peak_prominences(signal, peaks)
            widths, _, __, ___ = scipy.signal.peak_widths(signal, peaks, rel_height=0.5)

            # Prepare output
            info = {"Peaks": peaks, "Distance": distances, "Height": heights, "Width": widths}

            return info

            





        # =============================================================================
        #                                 EXECUTION
        # =============================================================================

        info = _signal_findpeaks_scipy(self.data)

        # Absolute
        info = _signal_findpeaks_keep(
            info,
            what="Height",
            below=height_max,
            above=height_min,
            relative_mean=False,
            relative_median=False,
            relative_max=False,
        )

        # Relative
        info = _signal_findpeaks_keep(
            info,
            what="Height",
            below=relative_height_max,
            above=relative_height_min,
            relative_mean=relative_mean,
            relative_median=relative_median,
            relative_max=relative_max,
        )

        # Filter
        info["Distance"] = _signal_findpeaks_distances(info["Peaks"])
        info["Onsets"] = _signal_findpeaks_findbase(info["Peaks"], self.data, what="onset")
        info["Offsets"] = _signal_findpeaks_findbase(info["Peaks"], self.data, what="offset")

        return info


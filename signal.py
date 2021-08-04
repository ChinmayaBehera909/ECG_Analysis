import numpy as np
import pandas as pd
from warnings import warn
import scipy.signal as sig
import scipy.misc
import scipy.interpolate
import datetime as dt
import scipy.ndimage
from .stats import fit_loess

from .misc import as_vector, find_closest
from .stats import standardize


class Signal:

    data = None
    fs = 0.0
    sig_type = 'sig'
    sig_len = 0
    base_time = dt.datetime.now()
    units= 'mv'

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



class Peaks(Signal):

    peaks = None
    signal = None

    def __init__(self, peaks):
        self.signal = Signal()
        self.peaks = peaks

    
    def _signal_formatpeaks_sanitize(peaks, key="Peaks"): 
            # Attempt to retrieve column.
            if isinstance(peaks, tuple):
                if isinstance(peaks[0], (dict, pd.DataFrame)):
                    peaks = peaks[0]
                elif isinstance(peaks[1], dict):
                    peaks = peaks[1]
                else:
                    peaks = peaks[0]

            if isinstance(peaks, pd.DataFrame):
                col = [col for col in peaks.columns if key in col]
                if len(col) == 0:
                    raise TypeError(
                        "EPK error: _signal_formatpeaks(): wrong type of input ",
                        "provided. Please provide indices of peaks.",
                    )
                peaks_signal = peaks[col[0]].values
                peaks = np.where(peaks_signal == 1)[0]

            if isinstance(peaks, dict):
                col = [col for col in list(peaks.keys()) if key in col]
                if len(col) == 0:
                    raise TypeError(
                        "EPK error: _signal_formatpeaks(): wrong type of input ",
                        "provided. Please provide indices of peaks.",
                    )
                peaks = peaks[col[0]]

            # Retrieve length.
            try:  # Detect if single peak
                len(peaks)
            except TypeError:
                peaks = np.array([peaks])

            return peaks
        

    def signal_fixpeaks(self, iterative=True, interval_min=None, interval_max=None, relative_interval_min=None, relative_interval_max=None, robust=False, method="Kubios"):
        """Correct erroneous peak placements.

        Identify and correct erroneous peak placements based on outliers in peak-to-peak differences (period).

        Parameters
        ----------
        peaks : list or array or DataFrame or Series or dict
            The samples at which the peaks occur. If an array is passed in, it is assumed that it was obtained
            with `signal_findpeaks()`. If a DataFrame is passed in, it is assumed to be obtained with `ecg_findpeaks()`
            or `ppg_findpeaks()` and to be of the same length as the input signal.
        sampling_rate : int
            The sampling frequency of the signal that contains the peaks (in Hz, i.e., samples/second).
        iterative : bool
            Whether or not to apply the artifact correction repeatedly (results in superior artifact correction).
        interval_min : float
            The minimum interval between the peaks.
        interval_max : float
            The maximum interval between the peaks.
        relative_interval_min : float
            The minimum interval between the peaks as relative to the sample (expressed in
            standard deviation from the mean).
        relative_interval_max : float
            The maximum interval between the peaks as relative to the sample (expressed in
            standard deviation from the mean).
        robust : bool
            Use a robust method of standardization (see `standardize()`) for the relative thresholds.
        method : str
            Either "Kubios" or "Neurokit". "Kubios" uses the artifact detection and correction described
            in Lipponen, J. A., & Tarvainen, M. P. (2019). Note that "Kubios" is only meant for peaks in
            ECG or PPG. "neurokit" can be used with peaks in ECG, PPG, or respiratory data.

        Returns
        -------
        peaks_clean : array
            The corrected peak locations.
        artifacts : dict
            Only if method="Kubios". A dictionary containing the indices of artifacts, accessible with the
            keys "ectopic", "missed", "extra", and "longshort".

        
        """
        # =============================================================================
        #                                    METHODS
        # =============================================================================

        peaks = self.peaks
        sampling_rate = self.signal.fs

        # neurokit
        def _signal_fixpeaks_neurokit(peaks, sampling_rate, interval_min=None, interval_max=None, relative_interval_min=None,
            relative_interval_max=None,
            robust=False ):
            
            peaks_clean = _remove_small(peaks, sampling_rate, interval_min, relative_interval_min, robust)
            peaks_clean = _interpolate_big(peaks, sampling_rate, interval_max, relative_interval_max, robust)

            return peaks_clean


        # kubios
        def _signal_fixpeaks_kubios(peaks, sampling_rate=1000, iterative=True):
            """kubios method."""

            # Get corrected peaks and normal-to-normal intervals.
            artifacts, subspaces = _find_artifacts(peaks, sampling_rate=sampling_rate)
            peaks_clean = _correct_artifacts(artifacts, peaks)

            if iterative:

                # Iteratively apply the artifact correction until the number of artifact
                # reaches an equilibrium (i.e., the number of artifacts does not change
                # anymore from one iteration to the next).
                n_artifacts_previous = np.inf
                n_artifacts_current = sum([len(i) for i in artifacts.values()])

                previous_diff = 0

                while n_artifacts_current - n_artifacts_previous != previous_diff:

                    previous_diff = n_artifacts_previous - n_artifacts_current

                    artifacts, subspaces = _find_artifacts(peaks_clean, sampling_rate=sampling_rate)
                    peaks_clean = _correct_artifacts(artifacts, peaks_clean)

                    n_artifacts_previous = n_artifacts_current
                    n_artifacts_current = sum([len(i) for i in artifacts.values()])

            return artifacts, peaks_clean


        def _find_artifacts(peaks, c1=0.13, c2=0.17, alpha=5.2, window_width=91, medfilt_order=11, sampling_rate=1000):

            # Compute period series (make sure it has same numer of elements as peaks);
            # peaks are in samples, convert to seconds.
            rr = np.ediff1d(peaks, to_begin=0) / sampling_rate
            # For subsequent analysis it is important that the first element has
            # a value in a realistic range (e.g., for median filtering).
            rr[0] = np.mean(rr[1:])

            # Artifact identification #

            # Compute dRRs: time series of differences of consecutive periods (dRRs).
            drrs = np.ediff1d(rr, to_begin=0)
            drrs[0] = np.mean(drrs[1:])
            # Normalize by threshold.
            th1 = _compute_threshold(drrs, alpha, window_width)
            drrs /= th1

            # ignore division by 0 warning
            np.seterr(divide="ignore", invalid="ignore")

            # Cast dRRs to subspace s12.
            # Pad drrs with one element.
            padding = 2
            drrs_pad = np.pad(drrs, padding, "reflect")

            s12 = np.zeros(drrs.size)
            for d in np.arange(padding, padding + drrs.size):

                if drrs_pad[d] > 0:
                    s12[d - padding] = np.max([drrs_pad[d - 1], drrs_pad[d + 1]])
                elif drrs_pad[d] < 0:
                    s12[d - padding] = np.min([drrs_pad[d - 1], drrs_pad[d + 1]])

            # Cast dRRs to subspace s22.
            s22 = np.zeros(drrs.size)
            for d in np.arange(padding, padding + drrs.size):

                if drrs_pad[d] >= 0:
                    s22[d - padding] = np.min([drrs_pad[d + 1], drrs_pad[d + 2]])
                elif drrs_pad[d] < 0:
                    s22[d - padding] = np.max([drrs_pad[d + 1], drrs_pad[d + 2]])

            # Compute mRRs: time series of deviation of RRs from median.
            df = pd.DataFrame({"signal": rr})
            medrr = df.rolling(medfilt_order, center=True, min_periods=1).median().signal.values
            mrrs = rr - medrr
            mrrs[mrrs < 0] = mrrs[mrrs < 0] * 2
            # Normalize by threshold.
            th2 = _compute_threshold(mrrs, alpha, window_width)
            mrrs /= th2


            # Artifact classes.
            extra_idcs = []
            missed_idcs = []
            ectopic_idcs = []
            longshort_idcs = []

            i = 0
            while i < rr.size - 2:  # The flow control is implemented based on Figure 1

                if np.abs(drrs[i]) <= 1:  # Figure 1
                    i += 1
                    continue
                eq1 = np.logical_and(drrs[i] > 1, s12[i] < (-c1 * drrs[i] - c2))  # pylint: disable=E1111
                eq2 = np.logical_and(drrs[i] < -1, s12[i] > (-c1 * drrs[i] + c2))  # pylint: disable=E1111

                if np.any([eq1, eq2]):
                    # If any of the two equations is true.
                    ectopic_idcs.append(i)
                    i += 1
                    continue
                # If none of the two equations is true.
                if ~np.any([np.abs(drrs[i]) > 1, np.abs(mrrs[i]) > 3]):  # Figure 1
                    i += 1
                    continue
                longshort_candidates = [i]
                # Check if the following beat also needs to be evaluated.
                if np.abs(drrs[i + 1]) < np.abs(drrs[i + 2]):
                    longshort_candidates.append(i + 1)

                for j in longshort_candidates:
                    # Long beat.
                    eq3 = np.logical_and(drrs[j] > 1, s22[j] < -1)  # pylint: disable=E1111
                    # Long or short.
                    eq4 = np.abs(mrrs[j]) > 3  # Figure 1
                    # Short beat.
                    eq5 = np.logical_and(drrs[j] < -1, s22[j] > 1)  # pylint: disable=E1111

                    if ~np.any([eq3, eq4, eq5]):
                        # If none of the three equations is true: normal beat.
                        i += 1
                        continue
                    # If any of the three equations is true: check for missing or extra
                    # peaks.

                    # Missing.
                    eq6 = np.abs(rr[j] / 2 - medrr[j]) < th2[j]  # Figure 1
                    # Extra.
                    eq7 = np.abs(rr[j] + rr[j + 1] - medrr[j]) < th2[j]  # Figure 1

                    # Check if extra.
                    if np.all([eq5, eq7]):
                        extra_idcs.append(j)
                        i += 1
                        continue
                    # Check if missing.
                    if np.all([eq3, eq6]):
                        missed_idcs.append(j)
                        i += 1
                        continue
                    # If neither classified as extra or missing, classify as "long or
                    # short".
                    longshort_idcs.append(j)
                    i += 1

            # Prepare output
            artifacts = {"ectopic": ectopic_idcs, "missed": missed_idcs, "extra": extra_idcs, "longshort": longshort_idcs}

            subspaces = {"rr": rr, "drrs": drrs, "mrrs": mrrs, "s12": s12, "s22": s22, "c1": c1, "c2": c2}

            return artifacts, subspaces


        def _compute_threshold(signal, alpha, window_width):

            df = pd.DataFrame({"signal": np.abs(signal)})
            q1 = df.rolling(window_width, center=True, min_periods=1).quantile(0.25).signal.values
            q3 = df.rolling(window_width, center=True, min_periods=1).quantile(0.75).signal.values
            th = alpha * ((q3 - q1) / 2)

            return th


        def _correct_artifacts(artifacts, peaks):

            # Artifact correction
            #--------------------
            # The integrity of indices must be maintained if peaks are inserted or
            # deleted: for each deleted beat, decrease indices following that beat in
            # all other index lists by 1. Likewise, for each added beat, increment the
            # indices following that beat in all other lists by 1.
            extra_idcs = artifacts["extra"]
            missed_idcs = artifacts["missed"]
            ectopic_idcs = artifacts["ectopic"]
            longshort_idcs = artifacts["longshort"]

            # Delete extra peaks.
            if extra_idcs:
                peaks = _correct_extra(extra_idcs, peaks)
                # Update remaining indices.
                missed_idcs = _update_indices(extra_idcs, missed_idcs, -1)
                ectopic_idcs = _update_indices(extra_idcs, ectopic_idcs, -1)
                longshort_idcs = _update_indices(extra_idcs, longshort_idcs, -1)

            # Add missing peaks.
            if missed_idcs:
                peaks = _correct_missed(missed_idcs, peaks)
                # Update remaining indices.
                ectopic_idcs = _update_indices(missed_idcs, ectopic_idcs, 1)
                longshort_idcs = _update_indices(missed_idcs, longshort_idcs, 1)

            if ectopic_idcs:
                peaks = _correct_misaligned(ectopic_idcs, peaks)

            if longshort_idcs:
                peaks = _correct_misaligned(longshort_idcs, peaks)

            return peaks


        def _correct_extra(extra_idcs, peaks):

            corrected_peaks = peaks.copy()
            corrected_peaks = np.delete(corrected_peaks, extra_idcs)

            return corrected_peaks


        def _correct_missed(missed_idcs, peaks):

            corrected_peaks = peaks.copy()
            missed_idcs = np.array(missed_idcs)
            # Calculate the position(s) of new beat(s). Make sure to not generate
            # negative indices. prev_peaks and next_peaks must have the same
            # number of elements.
            valid_idcs = np.logical_and(missed_idcs > 1, missed_idcs < len(corrected_peaks))  # pylint: disable=E1111
            missed_idcs = missed_idcs[valid_idcs]
            prev_peaks = corrected_peaks[[i - 1 for i in missed_idcs]]
            next_peaks = corrected_peaks[missed_idcs]
            added_peaks = prev_peaks + (next_peaks - prev_peaks) / 2
            # Add the new peaks before the missed indices (see numpy docs).
            corrected_peaks = np.insert(corrected_peaks, missed_idcs, added_peaks)

            return corrected_peaks


        def _correct_misaligned(misaligned_idcs, peaks):

            corrected_peaks = peaks.copy()
            misaligned_idcs = np.array(misaligned_idcs)
            # Make sure to not generate negative indices, or indices that exceed
            # the total number of peaks. prev_peaks and next_peaks must have the
            # same number of elements.
            valid_idcs = np.logical_and(
                misaligned_idcs > 1, misaligned_idcs < len(corrected_peaks) - 1  # pylint: disable=E1111
            )
            misaligned_idcs = misaligned_idcs[valid_idcs]
            prev_peaks = corrected_peaks[[i - 1 for i in misaligned_idcs]]
            next_peaks = corrected_peaks[[i + 1 for i in misaligned_idcs]]

            half_ibi = (next_peaks - prev_peaks) / 2
            peaks_interp = prev_peaks + half_ibi
            # Shift the R-peaks from the old to the new position.
            corrected_peaks = np.delete(corrected_peaks, misaligned_idcs)
            corrected_peaks = np.concatenate((corrected_peaks, peaks_interp)).astype(int)
            corrected_peaks.sort(kind="mergesort")

            return corrected_peaks


        def _update_indices(source_idcs, update_idcs, update):
            """For every element s in source_idcs, change every element u in update_idcs according to update, if u is larger
            than s."""
            if not update_idcs:
                return update_idcs

            for s in source_idcs:
                update_idcs = [u + update if u > s else u for u in update_idcs]

            return update_idcs

        # =============================================================================
        #                                UTILITIES
        # =============================================================================
        def _remove_small(peaks, sampling_rate=1000, interval_min=None, relative_interval_min=None, robust=False):
            if interval_min is None and relative_interval_min is None:
                return peaks

            if interval_min is not None:
                interval = self.signal_period(peaks, sampling_rate=sampling_rate, desired_length=None)
                peaks = peaks[interval > interval_min]

            if relative_interval_min is not None:
                interval = self.signal_period(peaks, sampling_rate=sampling_rate, desired_length=None)
                peaks = peaks[standardize(interval, robust=robust) > relative_interval_min]

            return peaks


        def _interpolate_big(peaks, sampling_rate=1000, interval_max=None, relative_interval_max=None, robust=False):
            if interval_max is None and relative_interval_max is None:
                return peaks

            continue_loop = True
            while continue_loop is True:
                if interval_max is not None:
                    interval = self.signal_period(peaks, sampling_rate=sampling_rate, desired_length=None)
                    peaks, continue_loop = _interpolate_missing(peaks, interval, interval_max, sampling_rate)

                if relative_interval_max is not None:
                    interval = self.signal_period(peaks, sampling_rate=sampling_rate, desired_length=None)
                    interval = standardize(interval, robust=robust)
                    peaks, continue_loop = _interpolate_missing(peaks, interval, interval_max, sampling_rate)

            return peaks


        def _interpolate_missing(peaks, interval, interval_max, sampling_rate):
            outliers = interval > interval_max
            outliers_loc = np.where(outliers)[0]
            if np.sum(outliers) == 0:
                return peaks, False

            # Delete large interval and replace by two unknown intervals
            interval[outliers] = np.nan
            interval = np.insert(interval, outliers_loc, np.nan)
            #    new_peaks_location = np.where(np.isnan(interval))[0]

            # Interpolate values
            interval = pd.Series(interval).interpolate().values
            peaks_corrected = _period_to_location(interval, sampling_rate, first_location=peaks[0])
            peaks = np.insert(peaks, outliers_loc, peaks_corrected[outliers_loc + np.arange(len(outliers_loc))])
            return peaks, True


        def _period_to_location(period, sampling_rate=1000, first_location=0):
            location = np.cumsum(period * sampling_rate)
            location = location - (location[0] - first_location)
            return location.astype(np.int)


        # =============================================================================
        #                                EXECUTION
        # =============================================================================

        # Format input
        peaks = self._signal_formatpeaks_sanitize(peaks)

        # If method Kubios
        if method.lower() == "kubios":

            return _signal_fixpeaks_kubios(peaks, sampling_rate=sampling_rate, iterative=iterative)

        # Else method is NeuroKit
        return _signal_fixpeaks_neurokit(
            peaks,
            sampling_rate=sampling_rate,
            interval_min=interval_min,
            interval_max=interval_max,
            relative_interval_min=relative_interval_min,
            relative_interval_max=relative_interval_max,
            robust=robust,
        )


    def signal_formatpeaks(self, info, desired_length, peak_indices=None):
        """Transforms an peak-info dict to a signal of given length."""

        # =============================================================================
        #                                 UTILITIES
        # =============================================================================


        def _signal_from_indices(indices, desired_length=None, value=1):
            """Generates array of 0 and given values at given indices.

            Used in *_findpeaks to transform vectors of peak indices to signal.

            """
            signal = np.zeros(desired_length, dtype=np.int)

            if isinstance(indices, list) and (not indices):    # skip empty lists
                return signal
            if isinstance(indices, np.ndarray) and (indices.size == 0):    # skip empty arrays
                return signal

            # Force indices as int
            if isinstance(indices[0], np.float):
                indices = indices[~np.isnan(indices)].astype(np.int)

            if isinstance(value, (int, float)):
                signal[indices] = value
            else:
                if len(value) != len(indices):
                    raise ValueError(
                        "EPK error: _signal_from_indices(): The number of values "
                        "is different from the number of indices."
                    )
                signal[indices] = value
            return signal

        
        # =============================================================================
        #                                 EXECUTION
        # =============================================================================

        if peak_indices is None:
            peak_indices = [key for key in info.keys() if "Peaks" in key]

        signals = {}
        for feature, values in info.items():
            if any(x in str(feature) for x in ["Peak", "Onset", "Offset", "Trough", "Recovery"]):
                signals[feature] = _signal_from_indices(values, desired_length, 1)
            else:
                signals[feature] = _signal_from_indices(peak_indices, desired_length, values)
        signals = pd.DataFrame(signals)
        return signals


    def signal_period(self, peaks, sampling_rate=1000, desired_length=None, interpolation_method="monotone_cubic"):
        """Calculate signal period from a series of peaks.

        Parameters
        ----------
        peaks : Union[list, np.array, pd.DataFrame, pd.Series, dict]
            The samples at which the peaks occur. If an array is passed in, it is assumed that it was obtained
            with `signal_findpeaks()`. If a DataFrame is passed in, it is assumed it is of the same length as
            the input signal in which occurrences of R-peaks are marked as "1", with such containers obtained
            with e.g., ecg_findpeaks() or rsp_findpeaks().
        sampling_rate : int
            The sampling frequency of the signal that contains peaks (in Hz, i.e., samples/second).
            Defaults to 1000.
        desired_length : int
            If left at the default None, the returned period will have the same number of elements as peaks.
            If set to a value larger than the sample at which the last peak occurs in the signal (i.e., peaks[-1]),
            the returned period will be interpolated between peaks over `desired_length` samples. To interpolate
            the period over the entire duration of the signal, set desired_length to the number of samples in the
            signal. Cannot be smaller than or equal to the sample at which the last peak occurs in the signal.
            Defaults to None.
        interpolation_method : str
            Method used to interpolate the rate between peaks. See `signal_interpolate()`. 'monotone_cubic' is chosen
            as the default interpolation method since it ensures monotone interpolation between data points
            (i.e., it prevents physiologically implausible "overshoots" or "undershoots" in the y-direction).
            In contrast, the widely used cubic spline interpolation does not ensure monotonicity.
        Returns
        -------
        array
            A vector containing the period.

        """
        peaks = self._signal_formatpeaks_sanitize(peaks)

        # Sanity checks.
        if np.size(peaks) <= 3:
            warn(
                "Too few peaks detected to compute the rate. Returning empty vector.")
            return np.full(desired_length, np.nan)

        if isinstance(desired_length, (int, float)):
            if desired_length <= peaks[-1]:
                raise ValueError("EPK error: desired_length must be None or larger than the index of the last peak.")

        # Calculate period in sec, based on peak to peak difference and make sure
        # that rate has the same number of elements as peaks (important for
        # interpolation later) by prepending the mean of all periods.
        period = np.ediff1d(peaks, to_begin=0) / sampling_rate
        period[0] = np.mean(period[1:])

        # Interpolate all statistics to desired length.
        if desired_length is not None:
            period = self.signal_interpolate(peaks, period, x_new=np.arange(desired_length), method=interpolation_method)

        return period


    def signal_rate(self, peaks, sampling_rate=1000, desired_length=None, interpolation_method="quadratic"):
        """Calculate signal rate from a series of peaks.

        This function can also be called either via ``ecg_rate()``, ```ppg_rate()`` or ``rsp_rate()``
        (aliases provided for consistency).

        Parameters
        ----------
        peaks : Union[list, np.array, pd.DataFrame, pd.Series, dict]
            The samples at which the peaks occur. If an array is passed in, it is assumed that it was obtained
            with `signal_findpeaks()`. If a DataFrame is passed in, it is assumed it is of the same length
            as the input signal in which occurrences of R-peaks are marked as "1", with such containers
            obtained with e.g., ecg_findpeaks() or rsp_findpeaks().
        sampling_rate : int
            The sampling frequency of the signal that contains peaks (in Hz, i.e., samples/second). Defaults to 1000.
        desired_length : int
            If left at the default None, the returned rated will have the same number of elements as peaks.
            If set to a value larger than the sample at which the last peak occurs in the signal (i.e., peaks[-1]),
            the returned rate will be interpolated between peaks over `desired_length` samples. To interpolate
            the rate over the entire duration of the signal, set desired_length to the number of samples in the
            signal. Cannot be smaller than or equal to the sample at which the last peak occurs in the signal.
            Defaults to None.
        interpolation_method : str
            Method used to interpolate the rate between peaks. See `signal_interpolate()`. 'monotone_cubic' is chosen
            as the default interpolation method since it ensures monotone interpolation between data points
            (i.e., it prevents physiologically implausible "overshoots" or "undershoots" in the y-direction).
            In contrast, the widely used cubic spline interpolation does not ensure monotonicity.

        Returns
        -------
        array
            A vector containing the rate.

        """
        period = self.signal_period(peaks, sampling_rate, desired_length, interpolation_method)
        rate = 60 / period

        return rate



class signalTools:

    def signal_interpolate(self, x_values, y_values, x_new=None, method="quadratic"):
        """Interpolate a signal.

        Interpolate a signal using different methods.

        Parameters
        ----------
        x_values : Union[list, np.array, pd.Series]
            The samples corresponding to the values to be interpolated.
        y_values : Union[list, np.array, pd.Series]
            The values to be interpolated.
        x_new : Union[list, np.array, pd.Series] or int
            The samples at which to interpolate the y_values. Samples before the first value in x_values
            or after the last value in x_values will be extrapolated.
            If an integer is passed, nex_x will be considered as the desired length of the interpolated
            signal between the first and the last values of x_values. No extrapolation will be done for values
            before or after the first and the last valus of x_values.
        method : str
            Method of interpolation. Can be 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
            'previous', 'next' or 'monotone_cubic'.  'zero', 'slinear', 'quadratic' and 'cubic' refer to
            a spline interpolation of zeroth, first, second or third order; 'previous' and 'next' simply
            return the previous or next value of the point) or as an integer specifying the order of the
            spline interpolator to use.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html
            for details on the 'monotone_cubic' method.

        Returns
        -------
        array
            Vector of interpolated samples.

        """
        # Sanity checks
        if len(x_values) != len(y_values):
            raise ValueError("SIGNAL ERROR: signal_interpolate() - x_values and y_values must be of the same length.")

        if isinstance(x_new, int):
            if len(x_values) == x_new:
                return y_values
        else:
            if len(x_values) == len(x_new):
                return y_values

        monotone_cubic = isinstance(method, str) and method.lower() == "monotone_cubic"  # bool

        if monotone_cubic:
            interpolation_function = scipy.interpolate.PchipInterpolator(x_values, y_values, extrapolate=True)
        else:
            interpolation_function = scipy.interpolate.interp1d(
                x_values, y_values, kind=method, bounds_error=False, fill_value=([y_values[0]], [y_values[-1]])
            )

        if isinstance(x_new, int):
            x_new = np.linspace(x_values[0], x_values[-1], x_new)

        interpolated = interpolation_function(x_new)

        if monotone_cubic:
            # Swap out the cubic extrapolation of out-of-bounds segments generated by
            # scipy.interpolate.PchipInterpolator for constant extrapolation akin to the behavior of
            # scipy.interpolate.interp1d with fill_value=([y_values[0]], [y_values[-1]].
            interpolated[: x_values[0]] = interpolated[x_values[0]]
            interpolated[x_values[-1] :] = interpolated[x_values[-1]]

        return interpolated

    def signal_resample(self, signal, desired_length=None, sampling_rate=None, desired_sampling_rate=None, method="interpolation" ):
        """Resample a continuous signal to a different length or sampling rate.

        Up- or down-sample a signal. The user can specify either a desired length for the vector, or input
        the original sampling rate and the desired sampling rate.
        
        Parameters
        ----------
        signal :  Union[list, np.array, pd.Series]
            The signal (i.e., a time series) in the form of a vector of values.
        desired_length : int
            The desired length of the signal.
        sampling_rate : int
            The original sampling frequency (in Hz, i.e., samples/second).
        desired_sampling_rate : int
            The desired (output) sampling frequency (in Hz, i.e., samples/second).
        method : str
            Can be 'interpolation' (see `scipy.ndimage.zoom()`), 'numpy' for numpy's interpolation
            (see `numpy.interp()`),'pandas' for Pandas' time series resampling, 'poly' (see `scipy.signal.resample_poly()`)
            or 'FFT' (see `scipy.signal.resample()`) for the Fourier method. FFT is the most accurate
            (if the signal is periodic), but becomes exponentially slower as the signal length increases.
            In contrast, 'interpolation' is the fastest, followed by 'numpy', 'poly' and 'pandas'.

        Returns
        -------
        array
            Vector containing resampled signal values.

        """

        # =============================================================================
        #                                   METHODS
        # =============================================================================


        def _resample_numpy(signal, desired_length):
            resampled_signal = np.interp(
                np.linspace(0.0, 1.0, desired_length, endpoint=False),  # where to interpolate
                np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
                signal,  # known data points
            )
            return resampled_signal


        def _resample_interpolation(signal, desired_length):
            resampled_signal = scipy.ndimage.zoom(signal, desired_length / len(signal))
            return resampled_signal


        def _resample_fft(signal, desired_length):
            resampled_signal = sig.resample(signal, desired_length)
            return resampled_signal


        def _resample_poly(signal, desired_length):
            resampled_signal = sig.resample_poly(signal, desired_length, len(signal))
            return resampled_signal


        def _resample_pandas(signal, desired_length):
            # Convert to Time Series
            index = pd.date_range("20131212", freq="L", periods=len(signal))
            resampled_signal = pd.Series(signal, index=index)

            # Create resampling factor
            resampling_factor = str(np.round(1 / (desired_length / len(signal)), 6)) + "L"

            # Resample
            resampled_signal = resampled_signal.resample(resampling_factor).bfill().values

            # Sanitize
            resampled_signal = _resample_sanitize(resampled_signal, desired_length)

            return resampled_signal


        # =============================================================================
        #                                  UTILITIES
        # =============================================================================


        def _resample_sanitize(resampled_signal, desired_length):
            # Adjust extremities
            diff = len(resampled_signal) - desired_length
            if diff < 0:
                resampled_signal = np.concatenate([resampled_signal, np.full(np.abs(diff), resampled_signal[-1])])
            elif diff > 0:
                resampled_signal = resampled_signal[0:desired_length]
            return resampled_signal


        # =============================================================================
        #                                  EXECUTION
        # =============================================================================

        if desired_length is None:
            desired_length = int(np.round(len(signal) * desired_sampling_rate / sampling_rate))
            #print('Desired length:',desired_length)

        # Sanity checks
        if len(signal) == desired_length:
            return signal

        # Resample
        if method.lower() == "fft":
            resampled = _resample_fft(signal, desired_length)
        elif method.lower() == "poly":
            resampled = _resample_poly(signal, desired_length)
        elif method.lower() == "numpy":
            resampled = _resample_numpy(signal, desired_length)
        elif method.lower() == "pandas":
            resampled = _resample_pandas(signal, desired_length)
        else:
            resampled = _resample_interpolation(signal, desired_length)

        return resampled

    def signal_sanitize(self, signal):
        """Reset indexing for Pandas Series

        Parameters
        ----------
        signal : Series
            The indexed input signal (pandas set_index())

        Returns
        -------
        Series
            The default indexed signal

        """

        # Series check for non-default index
        if type(signal) is pd.Series and type(signal.index) != pd.RangeIndex:
            return signal.reset_index(drop=True)

        return signal

    def signal_smooth(self, signal, method="convolution", kernel="boxzen", size=10, alpha=0.1):
        """Signal smoothing.

        Signal smoothing can be achieved using either the convolution of a filter kernel with the input
        signal to compute the smoothed signal (Smith, 1997) or a LOESS regression.

        Parameters
        ----------
        signal : Union[list, np.array, pd.Series]
            The signal (i.e., a time series) in the form of a vector of values.
        method : str
            Can be one of 'convolution' (default) or 'loess'.
        kernel : Union[str, np.array]
            Only used if `method` is 'convolution'. Type of kernel to use; if array, use directly as the
            kernel. Can be one of 'median', 'boxzen', 'boxcar', 'triang', 'blackman', 'hamming', 'hann',
            'bartlett', 'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall', 'barthann', 'kaiser'
            (needs beta), 'gaussian' (needs std), 'general_gaussian' (needs power, width), 'slepian' (needs width)
            or 'chebwin' (needs attenuation).
        size : int
            Only used if `method` is 'convolution'. Size of the kernel; ignored if kernel is an array.
        alpha : float
            Only used if `method` is 'loess'. The parameter which controls the degree of smoothing.

        Returns
        -------
        array
            Smoothed signal.


    
        """

            
        # =============================================================================
        #                                  UTILITIES
        # =============================================================================
        def _signal_smoothing_median(signal, size=5):

            # Enforce odd kernel size.
            if size % 2 == 0:
                size += 1

            smoothed = sig.medfilt(signal, kernel_size=int(size))
            return smoothed


        def _signal_smoothing(signal, kernel="boxcar", size=5):

            # Get window.
            size = int(size)
            window = sig.get_window(kernel, size)
            w = window / window.sum()

            # Extend signal edges to avoid boundary effects.
            x = np.concatenate((signal[0] * np.ones(size), signal, signal[-1] * np.ones(size)))

            # Compute moving average.
            smoothed = np.convolve(w, x, mode="same")
            smoothed = smoothed[size:-size]
            return smoothed



        # =============================================================================
        #                                  EXECUTION
        # =============================================================================

        if isinstance(signal, pd.Series):
            signal = signal.values

        length = len(signal)

        if isinstance(kernel, str) is False:
            raise TypeError("EPK error: signal_smooth(): 'kernel' should be a string.")

        # Check length.
        if size > length or size < 1:
            raise TypeError("EPK error: signal_smooth(): 'size' should be between 1 and length of the signal.")

        method = method.lower()

        # LOESS
        if method in ["loess", "lowess"]:
            smoothed = fit_loess(signal, alpha=alpha)

        # Convolution
        else:
            if kernel == "boxzen":
                # hybrid method
                # 1st pass - boxcar kernel
                x = _signal_smoothing(signal, kernel="boxcar", size=size)

                # 2nd pass - parzen kernel
                smoothed = _signal_smoothing(x, kernel="parzen", size=size)

            elif kernel == "median":
                smoothed = _signal_smoothing_median(signal, size)

            else:
                smoothed = _signal_smoothing(signal, kernel=kernel, size=size)

        return smoothed

    def signal_zerocrossings(self, signal, direction="both"):
        """Locate the indices where the signal crosses zero.

        Note that when the signal crosses zero between two points, the first index is returned.

        Parameters
        ----------
        signal : Union[list, np.array, pd.Series]
            The signal (i.e., a time series) in the form of a vector of values.
        direction : str
            Direction in which the signal crosses zero, can be "positive", "negative" or "both" (default).

        Returns
        -------
        array
            Vector containing the indices of zero crossings.

        """
        df = np.diff(np.sign(signal))
        if direction in ["positive", "up"]:
            zerocrossings = np.where(df > 0)[0]
        elif direction in ["negative", "down"]:
            zerocrossings = np.where(df < 0)[0]
        else:
            zerocrossings = np.nonzero(np.abs(df) > 0)[0]

        return zerocrossings


    ### handle all the non required functions to make them globally available
    ### make sure that the signal class methods include on the primary signal functions
    ### look into inheritance for adding a signal.peaks class and its attributes
from .signal import Signal
import numpy as np
import scipy.signal
from .misc import as_vector
import pandas as pd
from scipy.ndimage import gaussian_filter1d


#from .signal import signal_filter

from .stats import standardize



class ecg:

    signal = None
    peaks = None
    ecg_info ={}
    quality = None
    ecg_templates = None
    rate = None

    def __init__(self, data, fs, sig_type, base_time, units, c_method, p_method):

        self.signal = Signal(data = data, fs = fs, sig_type = sig_type, base_time = base_time, units = units)

        self.quality, self.peaks, self.ecg_info, self.ecg_templates, self.rate = self.ecg_process(c_method=None, p_method=None)


    def ecg_clean(self, method="nk"):
        """Clean an ECG signal.

        Prepare a raw ECG signal for R-peak detection with the specified method.

        Parameters
        ----------
        ecg_signal : Union[list, np.array, pd.Series]
            The raw ECG channel.
        sampling_rate : int
            The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
            Defaults to 1000.

        Returns
        -------
        array
            Vector containing the cleaned ECG signal.
        """


        
        # =============================================================================
        #                                  METHODS
        # =============================================================================

        # Nk
        def _ecg_clean_nk(self):

            # Remove slow drift and dc offset with highpass Butterworth.
            self.signal.data = self.signal.signal_filter(self.signal, lowcut=0.5, method="butterworth", order=5)

            self.signal.data = self.signal.signal_filter(self.signal, method="powerline", powerline=50)


        # Biosppy
        def _ecg_clean_biosppy(ecg_signal, sampling_rate=1000):
            """Adapted from https://github.com/PIA-
            Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py#L69."""

            order = int(0.3 * sampling_rate)
            if order % 2 == 0:
                order += 1  # Enforce odd number

            # -> filter_signal()
            frequency = [3, 45]

            #   -> get_filter()
            #     -> _norm_freq()
            frequency = 2 * np.array(frequency) / sampling_rate  # Normalize frequency to Nyquist Frequency (Fs/2).

            #     -> get coeffs
            a = np.array([1])
            b = scipy.signal.firwin(numtaps=order, cutoff=frequency, pass_zero=False)

            # _filter_signal()
            filtered = scipy.signal.filtfilt(b, a, ecg_signal)

            return filtered


        # Pan & Tompkins (1985)
        def _ecg_clean_pantompkins(ecg_signal, sampling_rate=1000):
            """Adapted from https://github.com/PIA-
            Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py#L69."""

            f1 = 5 / sampling_rate
            f2 = 15 / sampling_rate
            order = 1

            b, a = scipy.signal.butter(order, [f1 * 2, f2 * 2], btype="bandpass")

            return scipy.signal.lfilter(b, a, ecg_signal)  # Return filtered


        # Elgendi et al. (2010)
        def _ecg_clean_elgendi(ecg_signal, sampling_rate=1000):
            """From https://github.com/berndporr/py-ecg-detectors/

            - Elgendi, Mohamed & Jonkman, Mirjam & De Boer, Friso. (2010). Frequency Bands Effects on QRS
            Detection. The 3rd International Conference on Bio-inspired Systems and Signal Processing
            (BIOSIGNALS2010). 428-431.

            """

            f1 = 8 / sampling_rate
            f2 = 20 / sampling_rate

            b, a = scipy.signal.butter(2, [f1 * 2, f2 * 2], btype="bandpass")

            return scipy.signal.lfilter(b, a, ecg_signal)  # Return filtered


        # Hamilton (2002)
        def _ecg_clean_hamilton(ecg_signal, sampling_rate=1000):
            """Adapted from https://github.com/PIA-
            Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py#L69."""

            f1 = 8 / sampling_rate
            f2 = 16 / sampling_rate

            b, a = scipy.signal.butter(1, [f1 * 2, f2 * 2], btype="bandpass")

            return scipy.signal.lfilter(b, a, ecg_signal)  # Return filtered


        # Engzee Modified (2012)
        def _ecg_clean_engzee(ecg_signal, sampling_rate=1000):
            """From https://github.com/berndporr/py-ecg-detectors/

            - C. Zeelenberg, A single scan algorithm for QRS detection and feature extraction, IEEE Comp.
            in Cardiology, vol. 6, pp. 37-42, 1979.

            - A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred, "Real Time Electrocardiogram Segmentation
            for Finger Based ECG Biometrics", BIOSIGNALS 2012, pp. 49-54, 2012.

            """

            f1 = 48 / sampling_rate
            f2 = 52 / sampling_rate
            b, a = scipy.signal.butter(4, [f1 * 2, f2 * 2], btype="bandstop")
            return scipy.signal.lfilter(b, a, ecg_signal)  # Return filtered


        # =============================================================================
        #                                 EXECUTION
        # =============================================================================
            
        ecg_signal = as_vector(self.signal.data)

        method = method.lower()  # remove capitalised letters
        if method in ["nk", "nk2"]:
            _ecg_clean_nk(self)
        elif method in ["biosppy", "gamboa2008"]:
            clean = _ecg_clean_biosppy(self.signal.data,  self.signal.fs)
        elif method in ["pantompkins", "pantompkins1985"]:
            clean = _ecg_clean_pantompkins(self.signal.data,  self.signal.fs)
        elif method in ["hamilton", "hamilton2002"]:
            clean = _ecg_clean_hamilton(self.signal.data,  self.signal.fs)
        elif method in ["elgendi", "elgendi2010"]:
            clean = -(_ecg_clean_elgendi(self.signal.data,  self.signal.fs))
        elif method in ["engzee", "engzee2012", "engzeemod", "engzeemod2012"]:
            clean = _ecg_clean_engzee(self.signal.data,  self.signal.fs)
        elif method in [
            "christov",
            "christov2004",
            "ssf",
            "slopesumfunction",
            "zong",
            "zong2003",
            "kalidas2017",
            "swt",
            "kalidas",
            "kalidastamil",
            "kalidastamil2017",
        ]:
            clean = self.signal.data

        else:
            raise ValueError(
                "ECG METHOD ERROR: ecg_clean() - 'method' should be "
                "one of 'nk', 'biosppy', 'pantompkins1985',"
                " 'hamilton2002', 'elgendi2010', 'engzeemod2012'."
            )

        # Replace the original signal with cleaned signal
        #self.signal.data = clean
        return clean


    def ecg_delineate(self, method="dwt", check=False ):
        """Delineate QRS complex.

        Function to delineate the QRS complex.

        - **Cardiac Cycle**: A typical ECG heartbeat consists of a P wave, a QRS complex and a T wave.
        The P wave represents the wave of depolarization that spreads from the SA-node throughout the atria.
        The QRS complex reflects the rapid depolarization of the right and left ventricles. Since the
        ventricles are the largest part of the heart, in terms of mass, the QRS complex usually has a much
        larger amplitude than the P-wave. The T wave represents the ventricular repolarization of the
        ventricles.On rare occasions, a U wave can be seen following the T wave. The U wave is believed
        to be related to the last remnants of ventricular repolarization.

        Parameters
        ----------
        ecg_cleaned : Union[list, np.array, pd.Series]
            The cleaned ECG channel as returned by `ecg_clean()`.
        rpeaks : Union[list, np.array, pd.Series]
            The samples at which R-peaks occur. Accessible with the key "ECG_R_Peaks" in the info dictionary
            returned by `ecg_findpeaks()`.
        sampling_rate : int
            The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
            Defaults to 500.
        method : str
            Can be one of 'peak' (default) for a peak-based method, 'cwt' for continuous wavelet transform
            or 'dwt' for discrete wavelet transform.
        check : bool
            Defaults to False.

        Returns
        -------
        waves : dict
            A dictionary containing additional information.
            For derivative method, the dictionary contains the samples at which P-peaks, Q-peaks, S-peaks,
            T-peaks, P-onsets and T-offsets occur, accessible with the key "ECG_P_Peaks", "ECG_Q_Peaks",
            "ECG_S_Peaks", "ECG_T_Peaks", "ECG_P_Onsets", "ECG_T_Offsets" respectively.

            For wavelet methods, the dictionary contains the samples at which P-peaks, T-peaks, P-onsets,
            P-offsets, T-onsets, T-offsets, QRS-onsets and QRS-offsets occur, accessible with the key
            "ECG_P_Peaks", "ECG_T_Peaks", "ECG_P_Onsets", "ECG_P_Offsets", "ECG_T_Onsets", "ECG_T_Offsets",
            "ECG_R_Onsets", "ECG_R_Offsets" respectively.

        signals : DataFrame
            A DataFrame of same length as the input signal in which occurences of
            peaks, onsets and offsets marked as "1" in a list of zeros.

        """

        from .signal import signal_zerocrossings
        # =============================================================================
        #                                METHODS
        # =============================================================================

        # WAVELET METHOD (DWT)
        def _dwt_resample_points(peaks, sampling_rate, desired_sampling_rate):
            """Resample given points to a different sampling rate."""
            peaks_resample = np.array(peaks) * desired_sampling_rate / sampling_rate
            peaks_resample = [np.nan if np.isnan(x) else int(x) for x in peaks_resample.tolist()]
            return peaks_resample


        def _dwt_ecg_delineator(ecg, rpeaks, sampling_rate, analysis_sampling_rate=2000):
            """Delinate ecg signal using discrete wavelet transforms.

            Parameters
            ----------
            ecg : Union[list, np.array, pd.Series]
                The cleaned ECG channel as returned by `ecg_clean()`.
            rpeaks : Union[list, np.array, pd.Series]
                The samples at which R-peaks occur. Accessible with the key "ECG_R_Peaks" in the info dictionary
                returned by `ecg_findpeaks()`.
            sampling_rate : int
                The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
            analysis_sampling_rate : int
                The sampling frequency for analysis (in Hz, i.e., samples/second).

            Returns
            --------
            dict
                Dictionary of the points.

            """
            ecg = self.signal.signal_resample(ecg, sampling_rate=sampling_rate, desired_sampling_rate=analysis_sampling_rate)
            dwtmatr = _dwt_compute_multiscales(ecg, 9)

            rpeaks_resampled = _dwt_resample_points(rpeaks, sampling_rate, analysis_sampling_rate)

            tpeaks, ppeaks = _dwt_delineate_tp_peaks(ecg, rpeaks_resampled, dwtmatr, sampling_rate=analysis_sampling_rate)
            qrs_onsets, qrs_offsets = _dwt_delineate_qrs_bounds(
                rpeaks_resampled, dwtmatr, ppeaks, tpeaks, sampling_rate=analysis_sampling_rate
            )
            ponsets, poffsets = _dwt_delineate_tp_onsets_offsets(ppeaks, dwtmatr, sampling_rate=analysis_sampling_rate)
            tonsets, toffsets = _dwt_delineate_tp_onsets_offsets(
                tpeaks, dwtmatr, sampling_rate=analysis_sampling_rate, onset_weight=0.6, duration=0.6
            )

            return dict(
                ECG_T_Peaks=_dwt_resample_points(tpeaks, analysis_sampling_rate, desired_sampling_rate=sampling_rate),
                ECG_T_Onsets=_dwt_resample_points(tonsets, analysis_sampling_rate, desired_sampling_rate=sampling_rate),
                ECG_T_Offsets=_dwt_resample_points(toffsets, analysis_sampling_rate, desired_sampling_rate=sampling_rate),
                ECG_P_Peaks=_dwt_resample_points(ppeaks, analysis_sampling_rate, desired_sampling_rate=sampling_rate),
                ECG_P_Onsets=_dwt_resample_points(ponsets, analysis_sampling_rate, desired_sampling_rate=sampling_rate),
                ECG_P_Offsets=_dwt_resample_points(poffsets, analysis_sampling_rate, desired_sampling_rate=sampling_rate),
                ECG_R_Onsets=_dwt_resample_points(qrs_onsets, analysis_sampling_rate, desired_sampling_rate=sampling_rate),
                ECG_R_Offsets=_dwt_resample_points(qrs_offsets, analysis_sampling_rate, desired_sampling_rate=sampling_rate),
            )


        def _dwt_compensate_degree(sampling_rate):
            return int(np.log2(sampling_rate / 250))


        def _dwt_delineate_tp_peaks(
            ecg,
            rpeaks,
            dwtmatr,
            sampling_rate=250,
            qrs_width=0.13,
            p2r_duration=0.2,
            rt_duration=0.25,
            degree_tpeak=3,
            degree_ppeak=2,
            epsilon_T_weight=0.25,
            epsilon_P_weight=0.02,
        ):

            srch_bndry = int(0.5 * qrs_width * sampling_rate)
            degree_add = _dwt_compensate_degree(sampling_rate)

            tpeaks = []
            for rpeak_ in rpeaks:
                if np.isnan(rpeak_):
                    tpeaks.append(np.nan)
                    continue
                # search for T peaks from R peaks
                srch_idx_start = rpeak_ + srch_bndry
                srch_idx_end = rpeak_ + 2 * int(rt_duration * sampling_rate)
                dwt_local = dwtmatr[degree_tpeak + degree_add, srch_idx_start:srch_idx_end]
                height = epsilon_T_weight * np.sqrt(np.mean(np.square(dwt_local)))

                if len(dwt_local) == 0:
                    tpeaks.append(np.nan)
                    continue

                ecg_local = ecg[srch_idx_start:srch_idx_end]
                peaks, __ = scipy.signal.find_peaks(np.abs(dwt_local), height=height)
                peaks = list(filter(lambda p: np.abs(dwt_local[p]) > 0.025 * max(dwt_local), peaks))  # pylint: disable=W0640
                if dwt_local[0] > 0:  # just append
                    peaks = [0] + peaks

                # detect morphology
                candidate_peaks = []
                candidate_peaks_scores = []
                for idx_peak, idx_peak_nxt in zip(peaks[:-1], peaks[1:]):
                    correct_sign = dwt_local[idx_peak] > 0 and dwt_local[idx_peak_nxt] < 0  # pylint: disable=R1716
                    if correct_sign:
                        idx_zero = signal_zerocrossings(dwt_local[idx_peak:idx_peak_nxt])[0] + idx_peak
                        # This is the score assigned to each peak. The peak with the highest score will be
                        # selected.
                        score = ecg_local[idx_zero] - (float(idx_zero) / sampling_rate - (rt_duration - 0.5 * qrs_width))
                        candidate_peaks.append(idx_zero)
                        candidate_peaks_scores.append(score)

                if not candidate_peaks:
                    tpeaks.append(np.nan)
                    continue

                tpeaks.append(candidate_peaks[np.argmax(candidate_peaks_scores)] + srch_idx_start)

            ppeaks = []
            for rpeak in rpeaks:
                if np.isnan(rpeak):
                    ppeaks.append(np.nan)
                    continue

                # search for P peaks from Rpeaks
                srch_idx_start = rpeak - 2 * int(p2r_duration * sampling_rate)
                srch_idx_end = rpeak - srch_bndry
                dwt_local = dwtmatr[degree_ppeak + degree_add, srch_idx_start:srch_idx_end]
                height = epsilon_P_weight * np.sqrt(np.mean(np.square(dwt_local)))

                if len(dwt_local) == 0:
                    ppeaks.append(np.nan)
                    continue

                ecg_local = ecg[srch_idx_start:srch_idx_end]
                peaks, __ = scipy.signal.find_peaks(np.abs(dwt_local), height=height)
                peaks = list(filter(lambda p: np.abs(dwt_local[p]) > 0.025 * max(dwt_local), peaks))
                if dwt_local[0] > 0:  # just append
                    peaks = [0] + peaks

                # detect morphology
                candidate_peaks = []
                candidate_peaks_scores = []
                for idx_peak, idx_peak_nxt in zip(peaks[:-1], peaks[1:]):
                    correct_sign = dwt_local[idx_peak] > 0 and dwt_local[idx_peak_nxt] < 0  # pylint: disable=R1716
                    if correct_sign:
                        idx_zero = signal_zerocrossings(dwt_local[idx_peak:idx_peak_nxt])[0] + idx_peak
                        # This is the score assigned to each peak. The peak with the highest score will be
                        # selected.
                        score = ecg_local[idx_zero] - abs(
                            float(idx_zero) / sampling_rate - p2r_duration
                        )  # Minus p2r because of the srch_idx_start
                        candidate_peaks.append(idx_zero)
                        candidate_peaks_scores.append(score)

                if not candidate_peaks:
                    ppeaks.append(np.nan)
                    continue

                ppeaks.append(candidate_peaks[np.argmax(candidate_peaks_scores)] + srch_idx_start)

            return tpeaks, ppeaks


        def _dwt_delineate_tp_onsets_offsets(
            peaks,
            dwtmatr,
            sampling_rate=250,
            duration=0.3,
            duration_offset=0.3,
            onset_weight=0.4,
            offset_weight=0.4,
            degree_onset=2,
            degree_offset=2,
        ):
            degree = _dwt_compensate_degree(sampling_rate)
            onsets = []
            offsets = []
            for i in range(len(peaks)):  # pylint: disable=C0200
                # look for onsets
                srch_idx_start = peaks[i] - int(duration * sampling_rate)
                srch_idx_end = peaks[i]
                if srch_idx_start is np.nan or srch_idx_end is np.nan:
                    onsets.append(np.nan)
                    continue
                dwt_local = dwtmatr[degree_onset + degree, srch_idx_start:srch_idx_end]
                onset_slope_peaks, __ = scipy.signal.find_peaks(dwt_local)
                if len(onset_slope_peaks) == 0:
                    onsets.append(np.nan)
                    continue
                epsilon_onset = onset_weight * dwt_local[onset_slope_peaks[-1]]
                if not (dwt_local[: onset_slope_peaks[-1]] < epsilon_onset).any():
                    onsets.append(np.nan)
                    continue
                candidate_onsets = np.where(dwt_local[: onset_slope_peaks[-1]] < epsilon_onset)[0]
                onsets.append(candidate_onsets[-1] + srch_idx_start)


            for i in range(len(peaks)):  # pylint: disable=C0200
                # look for offset
                srch_idx_start = peaks[i]
                srch_idx_end = peaks[i] + int(duration_offset * sampling_rate)
                if srch_idx_start is np.nan or srch_idx_end is np.nan:
                    offsets.append(np.nan)
                    continue
                dwt_local = dwtmatr[degree_offset + degree, srch_idx_start:srch_idx_end]
                offset_slope_peaks, __ = scipy.signal.find_peaks(-dwt_local)
                if len(offset_slope_peaks) == 0:
                    offsets.append(np.nan)
                    continue
                epsilon_offset = -offset_weight * dwt_local[offset_slope_peaks[0]]
                if not (-dwt_local[onset_slope_peaks[0] :] < epsilon_offset).any():
                    offsets.append(np.nan)
                    continue
                candidate_offsets = np.where(-dwt_local[offset_slope_peaks[0] :] < epsilon_offset)[0] + offset_slope_peaks[0]
                try:
                    offsets.append(candidate_offsets[0] + srch_idx_start)
                except:
                    pass

            return onsets, offsets


        def _dwt_delineate_qrs_bounds(rpeaks, dwtmatr, ppeaks, tpeaks, sampling_rate=250):
            degree = int(np.log2(sampling_rate / 250))
            onsets = []
            for i in range(len(rpeaks)):  # pylint: disable=C0200
                # look for onsets
                srch_idx_start = ppeaks[i]
                srch_idx_end = rpeaks[i]
                if srch_idx_start is np.nan or srch_idx_end is np.nan:
                    onsets.append(np.nan)
                    continue
                dwt_local = dwtmatr[2 + degree, srch_idx_start:srch_idx_end]
                onset_slope_peaks, __ = scipy.signal.find_peaks(-dwt_local)
                if len(onset_slope_peaks) == 0:
                    onsets.append(np.nan)
                    continue
                epsilon_onset = 0.5 * -dwt_local[onset_slope_peaks[-1]]
                if not (-dwt_local[: onset_slope_peaks[-1]] < epsilon_onset).any():
                    onsets.append(np.nan)
                    continue
                candidate_onsets = np.where(-dwt_local[: onset_slope_peaks[-1]] < epsilon_onset)[0]
                onsets.append(candidate_onsets[-1] + srch_idx_start)


            offsets = []
            for i in range(len(rpeaks)):  # pylint: disable=C0200
                # look for offsets
                srch_idx_start = rpeaks[i]
                srch_idx_end = tpeaks[i]
                if srch_idx_start is np.nan or srch_idx_end is np.nan:
                    offsets.append(np.nan)
                    continue
                dwt_local = dwtmatr[2 + degree, srch_idx_start:srch_idx_end]
                onset_slope_peaks, __ = scipy.signal.find_peaks(dwt_local)
                if len(onset_slope_peaks) == 0:
                    offsets.append(np.nan)
                    continue
                epsilon_offset = 0.5 * dwt_local[onset_slope_peaks[0]]
                if not (dwt_local[onset_slope_peaks[0] :] < epsilon_offset).any():
                    offsets.append(np.nan)
                    continue
                candidate_offsets = np.where(dwt_local[onset_slope_peaks[0] :] < epsilon_offset)[0] + onset_slope_peaks[0]
                offsets.append(candidate_offsets[0] + srch_idx_start)


            return onsets, offsets


        def _dwt_compute_multiscales(ecg: np.ndarray, max_degree):
            """Return multiscales wavelet transforms."""

            def _apply_H_filter(signal_i, power=0):
                zeros = np.zeros(2 ** power - 1)
                timedelay = 2 ** power
                banks = np.r_[
                    1.0 / 8, zeros, 3.0 / 8, zeros, 3.0 / 8, zeros, 1.0 / 8,
                ]
                signal_f = scipy.signal.convolve(signal_i, banks, mode="full")
                signal_f[:-timedelay] = signal_f[timedelay:]  # timeshift: 2 steps
                return signal_f

            def _apply_G_filter(signal_i, power=0):
                zeros = np.zeros(2 ** power - 1)
                timedelay = 2 ** power
                banks = np.r_[2, zeros, -2]
                signal_f = scipy.signal.convolve(signal_i, banks, mode="full")
                signal_f[:-timedelay] = signal_f[timedelay:]  # timeshift: 1 step
                return signal_f

            dwtmatr = []
            intermediate_ret = np.array(ecg)
            for deg in range(max_degree):
                S_deg = _apply_G_filter(intermediate_ret, power=deg)
                T_deg = _apply_H_filter(intermediate_ret, power=deg)
                dwtmatr.append(S_deg)
                intermediate_ret = np.array(T_deg)
            dwtmatr = [arr[: len(ecg)] for arr in dwtmatr]  # rescale transforms to the same length
            return np.array(dwtmatr)


        # WAVELET METHOD (CWT)
        def _ecg_delineator_cwt(ecg, rpeaks=None, sampling_rate=1000):

            # P-Peaks and T-Peaks
            tpeaks, ppeaks = _peaks_delineator(ecg, rpeaks, sampling_rate=sampling_rate)

            # qrs onsets and offsets
            qrs_onsets, qrs_offsets = _onset_offset_delineator(ecg, rpeaks, peak_type="rpeaks", sampling_rate=sampling_rate)

            # ppeaks onsets and offsets
            p_onsets, p_offsets = _onset_offset_delineator(ecg, ppeaks, peak_type="ppeaks", sampling_rate=sampling_rate)

            # tpeaks onsets and offsets
            t_onsets, t_offsets = _onset_offset_delineator(ecg, tpeaks, peak_type="tpeaks", sampling_rate=sampling_rate)

            # Return info dictionary
            return {
                "ECG_P_Peaks": ppeaks,
                "ECG_T_Peaks": tpeaks,
                "ECG_R_Onsets": qrs_onsets,
                "ECG_R_Offsets": qrs_offsets,
                "ECG_P_Onsets": p_onsets,
                "ECG_P_Offsets": p_offsets,
                "ECG_T_Onsets": t_onsets,
                "ECG_T_Offsets": t_offsets,
            }


        # Internals
        # ---------------------


        def _onset_offset_delineator(ecg, peaks, peak_type="rpeaks", sampling_rate=1000):
            # Try loading pywt
            try:
                import pywt
            except ImportError:
                raise ImportError(
                    "Library error: ecg_delineator(): the 'PyWavelets' module is required for this method to run. ",
                    "Please install it first (`pip install PyWavelets`).",
                )
            # first derivative of the Gaissian signal
            scales = np.array([1, 2, 4, 8, 16])
            cwtmatr, __ = pywt.cwt(ecg, scales, "gaus1", sampling_period=1.0 / sampling_rate)

            half_wave_width = int(0.1 * sampling_rate)  # NEED TO CHECK
            onsets = []
            offsets = []
            for index_peak in peaks:
                # find onset
                if np.isnan(index_peak):
                    onsets.append(np.nan)
                    offsets.append(np.nan)
                    continue
                if peak_type == "rpeaks":
                    search_window = cwtmatr[2, index_peak - half_wave_width : index_peak]
                    prominence = 0.20 * max(search_window)
                    height = 0.0
                    wt_peaks, wt_peaks_data = scipy.signal.find_peaks(search_window, height=height, prominence=prominence)

                elif peak_type in ["tpeaks", "ppeaks"]:
                    search_window = -cwtmatr[4, index_peak - half_wave_width : index_peak]

                    prominence = 0.10 * max(search_window)
                    height = 0.0
                    wt_peaks, wt_peaks_data = scipy.signal.find_peaks(search_window, height=height, prominence=prominence)

                if len(wt_peaks) == 0:
                    # print("Fail to find onset at index: %d", index_peak)
                    onsets.append(np.nan)
                else:
                    # The last peak is nfirst in (Martinez, 2004)
                    nfirst = wt_peaks[-1] + index_peak - half_wave_width
                    if peak_type == "rpeaks":
                        if wt_peaks_data["peak_heights"][-1] > 0:
                            epsilon_onset = 0.05 * wt_peaks_data["peak_heights"][-1]
                    elif peak_type == "ppeaks":
                        epsilon_onset = 0.50 * wt_peaks_data["peak_heights"][-1]
                    elif peak_type == "tpeaks":
                        epsilon_onset = 0.25 * wt_peaks_data["peak_heights"][-1]
                    leftbase = wt_peaks_data["left_bases"][-1] + index_peak - half_wave_width
                    if peak_type == "rpeaks":
                        candidate_onsets = np.where(cwtmatr[2, nfirst - 100 : nfirst] < epsilon_onset)[0] + nfirst - 100
                    elif peak_type in ["tpeaks", "ppeaks"]:
                        candidate_onsets = np.where(-cwtmatr[4, nfirst - 100 : nfirst] < epsilon_onset)[0] + nfirst - 100

                    candidate_onsets = candidate_onsets.tolist() + [leftbase]
                    if len(candidate_onsets) == 0:
                        onsets.append(np.nan)
                    else:
                        onsets.append(max(candidate_onsets))

                # find offset
                if peak_type == "rpeaks":
                    search_window = -cwtmatr[2, index_peak : index_peak + half_wave_width]
                    prominence = 0.50 * max(search_window)
                    wt_peaks, wt_peaks_data = scipy.signal.find_peaks(search_window, height=height, prominence=prominence)

                elif peak_type in ["tpeaks", "ppeaks"]:
                    search_window = cwtmatr[4, index_peak : index_peak + half_wave_width]
                    prominence = 0.10 * max(search_window)
                    wt_peaks, wt_peaks_data = scipy.signal.find_peaks(search_window, height=height, prominence=prominence)

                if len(wt_peaks) == 0:
                    # print("Fail to find offsets at index: %d", index_peak)
                    offsets.append(np.nan)
                else:
                    nlast = wt_peaks[0] + index_peak
                    if peak_type == "rpeaks":
                        if wt_peaks_data["peak_heights"][0] > 0:
                            epsilon_offset = 0.125 * wt_peaks_data["peak_heights"][0]
                    elif peak_type == "ppeaks":
                        epsilon_offset = 0.9 * wt_peaks_data["peak_heights"][0]
                    elif peak_type == "tpeaks":
                        epsilon_offset = 0.4 * wt_peaks_data["peak_heights"][0]
                    rightbase = wt_peaks_data["right_bases"][0] + index_peak
                    if peak_type == "rpeaks":
                        candidate_offsets = np.where((-cwtmatr[2, nlast : nlast + 100]) < epsilon_offset)[0] + nlast
                    elif peak_type in ["tpeaks", "ppeaks"]:
                        candidate_offsets = np.where((cwtmatr[4, nlast : nlast + 100]) < epsilon_offset)[0] + nlast

                    candidate_offsets = candidate_offsets.tolist() + [rightbase]
                    if len(candidate_offsets) == 0:
                        offsets.append(np.nan)
                    else:
                        offsets.append(min(candidate_offsets))

            onsets = np.array(onsets, dtype="object")
            offsets = np.array(offsets, dtype="object")
            return onsets, offsets


        def _peaks_delineator(ecg, rpeaks, sampling_rate=1000):
            # Try loading pywt
            try:
                import pywt
            except ImportError:
                raise ImportError(
                    "Library error: ecg_delineator(): the 'PyWavelets' module is required for this method to run. ",
                    "Please install it first (`pip install PyWavelets`).",
                )
            # first derivative of the Gaissian signal
            scales = np.array([1, 2, 4, 8, 16])
            cwtmatr, __ = pywt.cwt(ecg, scales, "gaus1", sampling_period=1.0 / sampling_rate)

            qrs_duration = 0.1

            search_boundary = int(0.9 * qrs_duration * sampling_rate / 2)
            significant_peaks_groups = []
            for i in range(len(rpeaks) - 1):
                # search for T peaks and P peaks from R peaks
                start = rpeaks[i] + search_boundary
                end = rpeaks[i + 1] - search_boundary
                search_window = cwtmatr[4, start:end]
                height = 0.25 * np.sqrt(np.mean(np.square(search_window)))
                peaks_tp, heights_tp = scipy.signal.find_peaks(np.abs(search_window), height=height)
                peaks_tp = peaks_tp + rpeaks[i] + search_boundary
                # set threshold for heights of peaks to find significant peaks in wavelet
                threshold = 0.125 * max(search_window)
                significant_peaks_tp = []
                significant_peaks_tp = [peaks_tp[j] for j in range(len(peaks_tp)) if heights_tp["peak_heights"][j] > threshold]

                significant_peaks_groups.append(_find_tppeaks(ecg, significant_peaks_tp, sampling_rate=sampling_rate))

            tpeaks, ppeaks = zip(*[(g[0], g[-1]) for g in significant_peaks_groups])

            tpeaks = np.array(tpeaks, dtype="object")
            ppeaks = np.array(ppeaks, dtype="object")
            return tpeaks, ppeaks


        def _find_tppeaks(ecg, keep_tp, sampling_rate=1000):
            # Try loading pywt
            try:
                import pywt
            except ImportError:
                raise ImportError(
                    "Library error: ecg_delineator(): the 'PyWavelets' module is required for this method to run. ",
                    "Please install it first (`pip install PyWavelets`).",
                )
            # first derivative of the Gaissian signal
            scales = np.array([1, 2, 4, 8, 16])
            cwtmatr, __ = pywt.cwt(ecg, scales, "gaus1", sampling_period=1.0 / sampling_rate)
            max_search_duration = 0.05
            tppeaks = []
            for index_cur, index_next in zip(keep_tp[:-1], keep_tp[1:]):
                # limit 1
                correct_sign = cwtmatr[4, :][index_cur] < 0 and cwtmatr[4, :][index_next] > 0  # pylint: disable=R1716
                #    near = (index_next - index_cur) < max_wv_peak_dist #limit 2
                #    if near and correct_sign:
                if correct_sign:
                    index_zero_cr = signal_zerocrossings(cwtmatr[4, :][index_cur:index_next])[0] + index_cur
                    nb_idx = int(max_search_duration * sampling_rate)
                    index_max = np.argmax(ecg[index_zero_cr - nb_idx : index_zero_cr + nb_idx]) + (index_zero_cr - nb_idx)
                    tppeaks.append(index_max)
            if len(tppeaks) == 0:
                tppeaks = [np.nan]
            return tppeaks

        # Internal

        def _ecg_delineate_check(waves, rpeaks):
            """This function replaces the delineated features with np.nan if its standardized distance from R-peaks is more than
            3."""
            df = pd.DataFrame.from_dict(waves)
            features_columns = df.columns

            df = pd.concat([df, pd.DataFrame({"ECG_R_Peaks": rpeaks})], axis=1)

            # loop through all columns to calculate the z distance
            for column in features_columns:  # pylint: disable=W0612
                df = _calculate_abs_z(df, features_columns)

            # Replace with nan if distance > 3
            for col in features_columns:
                for i in range(len(df)):
                    if df["Dist_R_" + col][i] > 3:
                        df[col][i] = np.nan

            # Return df without distance columns
            df = df[features_columns]
            waves = df.to_dict("list")
            return waves


        def _calculate_abs_z(df, columns):
            """This function helps to calculate the absolute standardized distance between R-peaks and other delineated waves
            features by `ecg_delineate()`"""
            for column in columns:
                df["Dist_R_" + column] = np.abs(standardize(df[column].sub(df["ECG_R_Peaks"], axis=0)))
            return df


        # =============================================================================
        #                                  EXECUTION
        # =============================================================================        

        method = method.lower()  # remove capitalised letters
        if method in ["cwt", "continuous wavelet transform"]:
            waves = _ecg_delineator_cwt(self.signal.data, rpeaks=self.peaks, sampling_rate=self.signal.fs)
        elif method in ["dwt", "discrete wavelet transform"]:
            waves = _dwt_ecg_delineator(self.signal.data, self.peaks, sampling_rate=self.signal.fs)

        else:
            raise ValueError("Method error: ecg_delineate(): 'method' should be one of 'cwt' or 'dwt'.")

        # Remove NaN in Peaks, Onsets, and Offsets
        waves_noNA = waves.copy()
        for feature in waves_noNA.keys():
            waves_noNA[feature] = [int(x) for x in waves_noNA[feature] if ~np.isnan(x)]

        instant_peaks = self.signal.signal_formatpeaks(waves_noNA, desired_length=len(self.signal.data))
        signals = instant_peaks

        if check is True:
            waves = _ecg_delineate_check(waves, self.peaks)

        return signals, waves


    def ecg_findpeaks(ecg_cleaned, sampling_rate=1000, method="nk"):
        """Find R-peaks in an ECG signal.

        Low-level function used by `ecg_peaks()` to identify R-peaks in an ECG signal using a different set of algorithms. See `ecg_peaks()` for details.

        Parameters
        ----------
        ecg_cleaned : list, array or Series
            The cleaned ECG channel as returned by `ecg_clean()`.
        sampling_rate : int
            The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
            Defaults to 1000.
        method : string
            The algorithm to be used for R-peak detection. Can be one of 'nk' (default),
            'pamtompkins1985', 'hamilton2002', 'christov2004', 'gamboa2008', 'elgendi2010', 'engzeemod2012', 'kalidas2017' or 'martinez2003'.

        Returns
        -------
        info : dict
            A dictionary containing additional information, in this case the
            samples at which R-peaks occur, accessible with the key "ECG_R_Peaks".
        """



            

        # =============================================================================
        #                                    METHODS
        # =============================================================================

        from .signal import signal_smooth, signal_zerocrossings
        # NK
        def _ecg_findpeaks_nk(signal, sampling_rate=1000, smoothwindow=.1, avgwindow=.75,
                                    gradthreshweight=1.5, minlenweight=0.4, mindelay=0.3):
            """
            All tune-able parameters are specified as keyword arguments. The `signal`
            must be the highpass-filtered raw ECG with a lowcut of .5 Hz.
            """

            # Compute the ECG's gradient as well as the gradient threshold. Run with
            grad = np.gradient(signal)
            absgrad = np.abs(grad)
            smooth_kernel = int(np.rint(smoothwindow * sampling_rate))
            avg_kernel = int(np.rint(avgwindow * sampling_rate))
            smoothgrad = signal_smooth(absgrad, kernel="boxcar", size=smooth_kernel)
            avggrad = signal_smooth(smoothgrad, kernel="boxcar", size=avg_kernel)
            gradthreshold = gradthreshweight * avggrad
            mindelay = int(np.rint(sampling_rate * mindelay))

            # Identify start and end of QRS complexes.
            qrs = smoothgrad > gradthreshold
            beg_qrs = np.where(np.logical_and(np.logical_not(qrs[0:-1]), qrs[1:]))[0]
            end_qrs = np.where(np.logical_and(qrs[0:-1], np.logical_not(qrs[1:])))[0]
            # Throw out QRS-ends that precede first QRS-start.
            end_qrs = end_qrs[end_qrs > beg_qrs[0]]

            # Identify R-peaks within QRS (ignore QRS that are too short).
            num_qrs = min(beg_qrs.size, end_qrs.size)
            min_len = np.mean(end_qrs[:num_qrs] - beg_qrs[:num_qrs]) * minlenweight
            peaks = [0]

            for i in range(num_qrs):

                beg = beg_qrs[i]
                end = end_qrs[i]
                len_qrs = end - beg

                if len_qrs < min_len:
                    continue

                # Find local maxima and their prominence within QRS.
                data = signal[beg:end]
                locmax, props = scipy.signal.find_peaks(data, prominence=(None, None))

                if locmax.size > 0:
                    # Identify most prominent local maximum.
                    peak = beg + locmax[np.argmax(props["prominences"])]
                    # Enforce minimum delay between peaks.
                    if peak - peaks[-1] > mindelay:
                        peaks.append(peak)

            peaks.pop(0)

            peaks = np.asarray(peaks).astype(int)  # Convert to int
            return peaks




        # Pan & Tompkins (1985)
        def _ecg_findpeaks_pantompkins(signal, sampling_rate=1000):
            """
            From https://github.com/berndporr/py-ecg-detectors/

            - Jiapu Pan and Willis J. Tompkins. A Real-Time QRS Detection Algorithm.
            In: IEEE Transactions on Biomedical Engineering BME-32.3 (1985), pp. 230–236.
            """
            diff = np.diff(signal)

            squared = diff * diff
            #a=0.12
            N = int(0.04 * sampling_rate)
            mwa = _ecg_findpeaks_MWA(squared, N)
            #np.save('test_mwa.npy',mwa)
            mwa[:int(0.2 * sampling_rate)] = 0
            #np.save('test_mwa1.npy',mwa)

            mwa_peaks = _ecg_findpeaks_peakdetect(mwa, sampling_rate)

            mwa_peaks = np.array(mwa_peaks, dtype='int')
            return mwa_peaks



        # Hamilton (2002)
        def _ecg_findpeaks_hamilton(signal, sampling_rate=1000):
            """
            From https://github.com/berndporr/py-ecg-detectors/

            - Hamilton, Open Source ECG Analysis Software Documentation, E.P.Limited, 2002.
            """
            diff = abs(np.diff(signal))

            b = np.ones(int(0.08 * sampling_rate))
            b = b/int(0.08 * sampling_rate)
            a = [1]

            ma = scipy.signal.lfilter(b, a, diff)

            ma[0:len(b) * 2] = 0

            n_pks = []
            n_pks_ave = 0.0
            s_pks = []
            s_pks_ave = 0.0
            QRS = [0]
            RR = []
            RR_ave = 0.0

            th = 0.0

            i = 0
            idx = []
            peaks = []

            for i in range(len(ma)):

                if i > 0 and i < len(ma) - 1:
                    if ma[i-1] < ma[i] and ma[i + 1] < ma[i]:
                        peak = i
                        peaks.append(i)

                        if ma[peak] > th and (peak-QRS[-1]) > 0.3 * sampling_rate:
                            QRS.append(peak)
                            idx.append(i)
                            s_pks.append(ma[peak])
                            if len(n_pks) > 8:
                                s_pks.pop(0)
                            s_pks_ave = np.mean(s_pks)

                            if RR_ave != 0.0:
                                if QRS[-1]-QRS[-2] > 1.5 * RR_ave:
                                    missed_peaks = peaks[idx[-2] + 1:idx[-1]]
                                    for missed_peak in missed_peaks:
                                        if missed_peak - peaks[idx[-2]] > int(0.360 * sampling_rate) and ma[missed_peak] > 0.5 * th:
                                            QRS.append(missed_peak)
                                            QRS.sort()
                                            break

                            if len(QRS) > 2:
                                RR.append(QRS[-1]-QRS[-2])
                                if len(RR) > 8:
                                    RR.pop(0)
                                RR_ave = int(np.mean(RR))

                        else:
                            n_pks.append(ma[peak])
                            if len(n_pks) > 8:
                                n_pks.pop(0)
                            n_pks_ave = np.mean(n_pks)

                        th = n_pks_ave + 0.45 * (s_pks_ave-n_pks_ave)

                        i += 1

            QRS.pop(0)

            QRS = np.array(QRS, dtype='int')
            return QRS


        # Slope Sum Function (SSF) - Zong et al. (2003)
        def _ecg_findpeaks_ssf(signal, sampling_rate=1000, threshold=20, before=0.03, after=0.01):
            """
            From https://github.com/PIA-Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py#L448

            - W. Zong, T. Heldt, G.B. Moody, and R.G. Mark. An open-source algorithm to detect onset of arterial blood pressure pulses. In Computers in
        Cardiology, 2003, pages 259–262, 2003.
            """
            # TODO: Doesn't really seems to work

            # convert to samples
            winB = int(before * sampling_rate)
            winA = int(after * sampling_rate)

            Rset = set()
            length = len(signal)

            # diff
            dx = np.diff(signal)
            dx[dx >= 0] = 0
            dx = dx ** 2

            # detection
            idx, = np.nonzero(dx > threshold)
            idx0 = np.hstack(([0], idx))
            didx = np.diff(idx0)

            # search
            sidx = idx[didx > 1]
            for item in sidx:
                a = item - winB
                if a < 0:
                    a = 0
                b = item + winA
                if b > length:
                    continue

                r = np.argmax(signal[a:b]) + a
                Rset.add(r)

            # output
            rpeaks = list(Rset)
            rpeaks.sort()
            rpeaks = np.array(rpeaks, dtype='int')
            return rpeaks



        # Christov (2004)
        def _ecg_findpeaks_christov(signal, sampling_rate=1000):
            """
            From https://github.com/berndporr/py-ecg-detectors/

            - Ivaylo I. Christov, Real time electrocardiogram QRS detection using combined adaptive threshold, BioMedical Engineering OnLine 2004, vol. 3:28, 2004.
            """
            total_taps = 0

            b = np.ones(int(0.02 * sampling_rate))
            b = b/int(0.02 * sampling_rate)
            total_taps += len(b)
            a = [1]

            MA1 = scipy.signal.lfilter(b, a, signal)

            b = np.ones(int(0.028 * sampling_rate))
            b = b/int(0.028 * sampling_rate)
            total_taps += len(b)
            a = [1]

            MA2 = scipy.signal.lfilter(b, a, MA1)

            Y = []
            for i in range(1, len(MA2)-1):

                diff = abs(MA2[i + 1]-MA2[i-1])

                Y.append(diff)

            b = np.ones(int(0.040 * sampling_rate))
            b = b/int(0.040 * sampling_rate)
            total_taps += len(b)
            a = [1]

            MA3 = scipy.signal.lfilter(b, a, Y)

            MA3[0:total_taps] = 0

            ms50 = int(0.05 * sampling_rate)
            ms200 = int(0.2 * sampling_rate)
            ms1200 = int(1.2 * sampling_rate)
            ms350 = int(0.35 * sampling_rate)

            M = 0
            newM5 = 0
            M_list = []
            MM = []
            M_slope = np.linspace(1.0, 0.6, ms1200-ms200)
            F = 0
            F_list = []
            R = 0
            RR = []
            Rm = 0
            R_list = []

            MFR = 0
            MFR_list = []

            QRS = []

            for i in range(len(MA3)):

                # M
                if i < 5 * sampling_rate:
                    M = 0.6 * np.max(MA3[:i + 1])
                    MM.append(M)
                    if len(MM) > 5:
                        MM.pop(0)

                elif QRS and i < QRS[-1] + ms200:
                    newM5 = 0.6 * np.max(MA3[QRS[-1]:i])
                    if newM5 > 1.5 * MM[-1]:
                        newM5 = 1.1 * MM[-1]

                elif QRS and i == QRS[-1] + ms200:
                    if newM5 == 0:
                        newM5 = MM[-1]
                    MM.append(newM5)
                    if len(MM) > 5:
                        MM.pop(0)
                    M = np.mean(MM)

                elif QRS and i > QRS[-1] + ms200 and i < QRS[-1] + ms1200:

                    M = np.mean(MM) * M_slope[i-(QRS[-1] + ms200)]

                elif QRS and i > QRS[-1] + ms1200:
                    M = 0.6 * np.mean(MM)

                # F
                if i > ms350:
                    F_section = MA3[i-ms350:i]
                    max_latest = np.max(F_section[-ms50:])
                    max_earliest = np.max(F_section[:ms50])
                    F = F + ((max_latest-max_earliest)/150.0)

                # R
                if QRS and i < QRS[-1] + int((2.0/3.0 * Rm)):

                    R = 0

                elif QRS and i > QRS[-1] + int((2.0/3.0 * Rm)) and i < QRS[-1] + Rm:

                    dec = (M-np.mean(MM))/1.4
                    R = 0 + dec

                MFR = M + F + R
                M_list.append(M)
                F_list.append(F)
                R_list.append(R)
                MFR_list.append(MFR)

                if not QRS and MA3[i] > MFR:
                    QRS.append(i)

                elif QRS and i > QRS[-1] + ms200 and MA3[i] > MFR:
                    QRS.append(i)
                    if len(QRS) > 2:
                        RR.append(QRS[-1] - QRS[-2])
                        if len(RR) > 5:
                            RR.pop(0)
                        Rm = int(np.mean(RR))

            QRS.pop(0)
            QRS = np.array(QRS, dtype='int')
            return QRS



        # Gamboa (2008)
        def _ecg_findpeaks_gamboa(signal, sampling_rate=1000, tol=0.002):
            """
            From https://github.com/PIA-Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py#L834

            - Gamboa, H. (2008). Multi-modal behavioral biometrics based on hci and electrophysiology. PhD ThesisUniversidade.
            """

            # convert to samples
            v_100ms = int(0.1 * sampling_rate)
            v_300ms = int(0.3 * sampling_rate)
            hist, edges = np.histogram(signal, 100, density=True)

            TH = 0.01
            F = np.cumsum(hist)

            v0 = edges[np.nonzero(F > TH)[0][0]]
            v1 = edges[np.nonzero(F < (1 - TH))[0][-1]]

            nrm = max([abs(v0), abs(v1)])
            norm_signal = signal / float(nrm)

            d2 = np.diff(norm_signal, 2)

            b = np.nonzero((np.diff(np.sign(np.diff(-d2)))) == -2)[0] + 2
            b = np.intersect1d(b, np.nonzero(-d2 > tol)[0])

            if len(b) < 3:
                rpeaks = []
            else:
                b = b.astype('float')
                rpeaks = []
                previous = b[0]
                for i in b[1:]:
                    if i - previous > v_300ms:
                        previous = i
                        rpeaks.append(np.argmax(signal[int(i):int(i + v_100ms)]) + i)

            rpeaks = sorted(list(set(rpeaks)))
            rpeaks = np.array(rpeaks, dtype='int')
            return rpeaks







        # Engzee Modified (2012)
        def _ecg_findpeaks_engzee(signal, sampling_rate=1000):
            """
            From https://github.com/berndporr/py-ecg-detectors/

            - C. Zeelenberg, A single scan algorithm for QRS detection and feature extraction, IEEE Comp. in Cardiology, vol. 6, pp. 37-42, 1979
            - A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred, "Real Time Electrocardiogram Segmentation for Finger Based ECG Biometrics", BIOSIGNALS 2012, pp. 49-54, 2012.
            """
            engzee_fake_delay = 0

            diff = np.zeros(len(signal))
            for i in range(4, len(diff)):
                diff[i] = signal[i]-signal[i-4]

            ci = [1, 4, 6, 4, 1]
            low_pass = scipy.signal.lfilter(ci, 1, diff)

            low_pass[:int(0.2 * sampling_rate)] = 0

            ms200 = int(0.2 * sampling_rate)
            ms1200 = int(1.2 * sampling_rate)
            ms160 = int(0.16 * sampling_rate)
            neg_threshold = int(0.01 * sampling_rate)

            M = 0
            M_list = []
            neg_m = []
            MM = []
            M_slope = np.linspace(1.0, 0.6, ms1200-ms200)

            QRS = []
            r_peaks = []

            counter = 0

            thi_list = []
            thi = False
            thf_list = []
            thf = False

            for i in range(len(low_pass)):

                # M
                if i < 5 * sampling_rate:
                    M = 0.6 * np.max(low_pass[:i + 1])
                    MM.append(M)
                    if len(MM) > 5:
                        MM.pop(0)

                elif QRS and i < QRS[-1] + ms200:

                    newM5 = 0.6 * np.max(low_pass[QRS[-1]:i])

                    if newM5 > 1.5 * MM[-1]:
                        newM5 = 1.1 * MM[-1]

                elif QRS and i == QRS[-1] + ms200:
                    MM.append(newM5)
                    if len(MM) > 5:
                        MM.pop(0)
                    M = np.mean(MM)

                elif QRS and i > QRS[-1] + ms200 and i < QRS[-1] + ms1200:

                    M = np.mean(MM) * M_slope[i-(QRS[-1] + ms200)]

                elif QRS and i > QRS[-1] + ms1200:
                    M = 0.6 * np.mean(MM)

                M_list.append(M)
                neg_m.append(-M)

                if not QRS and low_pass[i] > M:
                    QRS.append(i)
                    thi_list.append(i)
                    thi = True

                elif QRS and i > QRS[-1] + ms200 and low_pass[i] > M:
                    QRS.append(i)
                    thi_list.append(i)
                    thi = True

                if thi and i < thi_list[-1] + ms160:
                    if low_pass[i] < -M and low_pass[i-1] > -M:
                        # thf_list.append(i)
                        thf = True

                    if thf and low_pass[i] < -M:
                        thf_list.append(i)
                        counter += 1

                    elif low_pass[i] > -M and thf:
                        counter = 0
                        thi = False
                        thf = False

                elif thi and i > thi_list[-1] + ms160:
                    counter = 0
                    thi = False
                    thf = False

                if counter > neg_threshold:
                    unfiltered_section = signal[thi_list[-1] - int(0.01 * sampling_rate):i]
                    r_peaks.append(engzee_fake_delay + np.argmax(unfiltered_section) + thi_list[-1] - int(0.01 * sampling_rate))
                    counter = 0
                    thi = False
                    thf = False

            r_peaks = np.array(r_peaks, dtype='int')
            return r_peaks




        # Stationary Wavelet Transform  (SWT) - Kalidas and Tamil (2017)
        def _ecg_findpeaks_kalidas(signal, sampling_rate=1000):
            """
            From https://github.com/berndporr/py-ecg-detectors/

            - Vignesh Kalidas and Lakshman Tamil (2017). Real-time QRS detector using Stationary Wavelet Transform for Automated ECG Analysis. In: 2017 IEEE 17th International Conference on Bioinformatics and Bioengineering (BIBE). Uses the Pan and Tompkins thresolding.
            """
            # Try loading pywt
            try:
                import pywt
            except ImportError:
                raise ImportError("Libaray error: ecg_findpeaks(): the 'PyWavelets' "
                                "module is required for this method to run. ",
                                "Please install it first (`pip install PyWavelets`).")

            swt_level = 3
            padding = -1
            for i in range(1000):
                if (len(signal) + i) % 2 ** swt_level == 0:
                    padding = i
                    break

            if padding > 0:
                signal = np.pad(signal, (0, padding), 'edge')
            elif padding == -1:
                print("Padding greater than 1000 required\n")

            swt_ecg = pywt.swt(signal, 'db3', level=swt_level)
            swt_ecg = np.array(swt_ecg)
            swt_ecg = swt_ecg[0, 1, :]

            squared = swt_ecg * swt_ecg

            f1 = 0.01/sampling_rate
            f2 = 10/sampling_rate

            b, a = scipy.signal.butter(3, [f1 * 2, f2 * 2], btype='bandpass')
            filtered_squared = scipy.signal.lfilter(b, a, squared)

            filt_peaks = _ecg_findpeaks_peakdetect(filtered_squared, sampling_rate)

            filt_peaks = np.array(filt_peaks, dtype='int')
            return filt_peaks





        # Elgendi et al. (2010)
        def _ecg_findpeaks_elgendi(signal, sampling_rate=1000):
            """
            From https://github.com/berndporr/py-ecg-detectors/

            - Elgendi, Mohamed & Jonkman, Mirjam & De Boer, Friso. (2010). Frequency Bands Effects on QRS Detection. The 3rd International Conference on Bio-inspired Systems and Signal Processing (BIOSIGNALS2010). 428-431.
            """

            window1 = int(0.12 * sampling_rate)
            mwa_qrs = _ecg_findpeaks_MWA(abs(signal), window1)

            window2 = int(0.6 * sampling_rate)
            mwa_beat = _ecg_findpeaks_MWA(abs(signal), window2)

            blocks = np.zeros(len(signal))
            block_height = np.max(signal)

            for i in range(len(mwa_qrs)):
                if mwa_qrs[i] > mwa_beat[i]:
                    blocks[i] = block_height
                else:
                    blocks[i] = 0

            QRS = []

            for i in range(1, len(blocks)):
                if blocks[i-1] == 0 and blocks[i] == block_height:
                    start = i

                elif blocks[i-1] == block_height and blocks[i] == 0:
                    end = i-1

                    if end-start > int(0.08 * sampling_rate):
                        detection = np.argmax(signal[start:end + 1]) + start
                        if QRS:
                            if detection-QRS[-1] > int(0.3 * sampling_rate):
                                QRS.append(detection)
                        else:
                            QRS.append(detection)

            QRS = np.array(QRS, dtype='int')
            return QRS


        # Continuous Wavelet Transform (CWT) - Martinez et al. (2003)
        def _ecg_findpeaks_WT(signal, sampling_rate=1000):
            # Try loading pywt
            try:
                import pywt
            except ImportError:
                raise ImportError("Library error: ecg_delineator(): the 'PyWavelets' "
                                "module is required for this method to run. ",
                                "Please install it first (`pip install PyWavelets`).")
            # first derivative of the Gaissian signal
            scales = np.array([1, 2, 4, 8, 16])
            cwtmatr, freqs = pywt.cwt(signal, scales, 'gaus1', sampling_period=1.0/sampling_rate)

            # For wt of scale 2^4
            signal_4 = cwtmatr[4, :]
            epsilon_4 = np.sqrt(np.mean(np.square(signal_4)))
            peaks_4, _ = scipy.signal.find_peaks(np.abs(signal_4), height=epsilon_4)

            # For wt of scale 2^3
            signal_3 = cwtmatr[3, :]
            epsilon_3 = np.sqrt(np.mean(np.square(signal_3)))
            peaks_3, _ = scipy.signal.find_peaks(np.abs(signal_3), height=epsilon_3)
            # Keep only peaks_3 that are nearest to peaks_4
            peaks_3_keep = np.zeros_like(peaks_4)
            for i in range(len(peaks_4)):
                peaks_distance = abs(peaks_4[i] - peaks_3)
                peaks_3_keep[i] = peaks_3[np.argmin(peaks_distance)]

            # For wt of scale 2^2
            signal_2 = cwtmatr[2, :]
            epsilon_2 = np.sqrt(np.mean(np.square(signal_2)))
            peaks_2, _ = scipy.signal.find_peaks(np.abs(signal_2), height=epsilon_2)
            # Keep only peaks_2 that are nearest to peaks_3
            peaks_2_keep = np.zeros_like(peaks_4)
            for i in range(len(peaks_4)):
                peaks_distance = abs(peaks_3_keep[i] - peaks_2)
                peaks_2_keep[i] = peaks_2[np.argmin(peaks_distance)]

            # For wt of scale 2^1
            signal_1 = cwtmatr[1, :]
            epsilon_1 = np.sqrt(np.mean(np.square(signal_1)))
            peaks_1, _ = scipy.signal.find_peaks(np.abs(signal_1), height=epsilon_1)
            # Keep only peaks_1 that are nearest to peaks_2
            peaks_1_keep = np.zeros_like(peaks_4)
            for i in range(len(peaks_4)):
                peaks_distance = abs(peaks_2_keep[i] - peaks_1)
                peaks_1_keep[i] = peaks_1[np.argmin(peaks_distance)]

            # Find R peaks
            max_R_peak_dist = int(0.1 * sampling_rate)
            rpeaks = []
            for index_cur, index_next in zip(peaks_1_keep[:-1], peaks_1_keep[1:]):
                correct_sign = signal_1[index_cur] < 0 and signal_1[index_next] > 0  # limit 1
                near = (index_next - index_cur) < max_R_peak_dist  # limit 2
                if near and correct_sign:
                    rpeaks.append(signal_zerocrossings(
                            signal_1[index_cur:index_next])[0] + index_cur)

            rpeaks = np.array(rpeaks, dtype='int')
            return rpeaks

        # =============================================================================
        #                                UTILITIES
        # =============================================================================

        def _ecg_findpeaks_MWA(signal, window_size):
            """
            From https://github.com/berndporr/py-ecg-detectors/
            """
            mwa = np.zeros(len(signal))
            for i in range(len(signal)):
                if i < window_size:
                    section = signal[0:i]
                else:
                    section = signal[i-window_size:i]

                if i != 0:
                    mwa[i] = np.mean(section)
                else:
                    mwa[i] = signal[i]

            '''temp = np.zeros(int(window_size/2))
            for i in range(len(mwa)-int(window_size/2), len(mwa)):
                np.delete(mwa,i)
                
            mwa = np.concatenate([temp,mwa], axis=0)'''
            return mwa






        def _ecg_findpeaks_peakdetect(detection, sampling_rate=1000):
            """
            From https://github.com/berndporr/py-ecg-detectors/
            """
            min_distance = int(0.25 * sampling_rate)

            signal_peaks = [0]
            noise_peaks = []

            SPKI = 0.0
            NPKI = 0.0

            threshold_I1 = 0.0
            threshold_I2 = 0.0

            RR_missed = 0
            index = 0
            indexes = []

            missed_peaks = []
            peaks = []

            for i in range(len(detection)):

                if i > 0 and i < len(detection) - 1:
                    if detection[i-1] < detection[i] and detection[i + 1] < detection[i]:
                        peak = i
                        peaks.append(i)

                        if detection[peak] > threshold_I1 and (peak - signal_peaks[-1]) > 0.3 * sampling_rate:

                            signal_peaks.append(peak)
                            indexes.append(index)
                            SPKI = 0.125 * detection[signal_peaks[-1]] + 0.875 * SPKI
                            if RR_missed != 0:
                                if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                                    missed_section_peaks = peaks[indexes[-2] + 1:indexes[-1]]
                                    missed_section_peaks2 = []
                                    for missed_peak in missed_section_peaks:
                                        if missed_peak - signal_peaks[-2] > min_distance and signal_peaks[-1] - missed_peak > min_distance and detection[missed_peak] > threshold_I2:
                                            missed_section_peaks2.append(missed_peak)

                                    if len(missed_section_peaks2) > 0:
                                        missed_peak = missed_section_peaks2[np.argmax(detection[missed_section_peaks2])]
                                        missed_peaks.append(missed_peak)
                                        signal_peaks.append(signal_peaks[-1])
                                        signal_peaks[-2] = missed_peak

                        else:
                            noise_peaks.append(peak)
                            NPKI = 0.125 * detection[noise_peaks[-1]] + 0.875 * NPKI

                        threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
                        threshold_I2 = 0.5 * threshold_I1

                        if len(signal_peaks) > 8:
                            RR = np.diff(signal_peaks[-9:])
                            RR_ave = int(np.mean(RR))
                            RR_missed = int(1.66 * RR_ave)

                        index = index + 1

            signal_peaks.pop(0)

            return signal_peaks



        method = method.lower()  # remove capitalised letters
        # Run peak detection algorithm
        if method in ["nk", "nk2"]:
            rpeaks = _ecg_findpeaks_nk(ecg_cleaned, sampling_rate)
        elif method in ["pantompkins", "pantompkins1985"]:
            rpeaks = _ecg_findpeaks_pantompkins(ecg_cleaned, sampling_rate)
        elif method in ["gamboa2008", "gamboa"]:
            rpeaks = _ecg_findpeaks_gamboa(ecg_cleaned, sampling_rate)
        elif method in ["ssf", "slopesumfunction", "zong", "zong2003"]:
            rpeaks = _ecg_findpeaks_ssf(ecg_cleaned, sampling_rate)
        elif method in ["hamilton", "hamilton2002"]:
            rpeaks = _ecg_findpeaks_hamilton(ecg_cleaned, sampling_rate)
        elif method in ["christov", "christov2004"]:
            rpeaks = _ecg_findpeaks_christov(ecg_cleaned, sampling_rate)
        elif method in ["engzee", "engzee2012", "engzeemod", "engzeemod2012"]:
            rpeaks = _ecg_findpeaks_engzee(ecg_cleaned, sampling_rate)
        elif method in ["elgendi", "elgendi2010"]:
            rpeaks = _ecg_findpeaks_elgendi(ecg_cleaned, sampling_rate)
        elif method in ["kalidas2017", "swt", "kalidas", "kalidastamil", "kalidastamil2017"]:
            rpeaks = _ecg_findpeaks_kalidas(ecg_cleaned, sampling_rate)
        elif method in ["martinez2003", "martinez"]:
            rpeaks = _ecg_findpeaks_WT(ecg_cleaned, sampling_rate)
        else:
            raise ValueError("Method error: ecg_findpeaks(): 'method' should be "
                            "one of 'nk' or 'pamtompkins'.")


        # Prepare output.
        info = {"ECG_R_Peaks": rpeaks}

        return info


    def ecg_peaks(self, correct_artifacts=False, method='nk'):
        """Find R-peaks in an ECG signal.

        Find R-peaks in an ECG signal using the specified method.

        Parameters
        ----------
        ecg_cleaned : Union[list, np.array, pd.Series]
            The cleaned ECG channel as returned by `ecg_clean()`.
        sampling_rate : int
            The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
            Defaults to 1000.
        correct_artifacts : bool
            Whether or not to identify artifacts as defined by Jukka A. Lipponen & Mika P. Tarvainen (2019):
            A robust algorithm for heart rate variability time series artefact correction using novel beat
            classification, Journal of Medical Engineering & Technology, DOI: 10.1080/03091902.2019.1640306.

        Returns
        -------
        signals : DataFrame
            A DataFrame of same length as the input signal in which occurences of R-peaks marked as "1"
            in a list of zeros with the same length as `ecg_cleaned`. Accessible with the keys "ECG_R_Peaks".
        info : dict
            A dictionary containing additional information, in this case the samples at which R-peaks occur,
            accessible with the key "ECG_R_Peaks".

        

        """

        def Check_repeat(x):
            _size = len(x)
            repeated = []
            nrep=[]
            for i in range(_size):
                k = i + 1
                for j in range(k, _size):
                    if x[i] == x[j] and x[i] not in repeated:
                        repeated.append(x[i])
                    
            for i in range(_size):
                if x[i] in repeated:
                    continue
                else:
                    nrep.append(x[i])
            return repeated, nrep

        rpeaks = self.ecg_findpeaks(self.signal.data, sampling_rate=self.signal.fs, method=method)

        if correct_artifacts:
            _, rpeaks = self.signal.signal_fixpeaks(rpeaks, sampling_rate=self.signal.fs, iterative=True, method="Kubios")
            
            #Removing repeated peaks
            r, xv = Check_repeat(rpeaks)
            rpeaks = xv
            
        return rpeaks


    def ecg_process(self, c_method=None, p_method=None, gaussian=None):


        """Process an ECG signal.

        Convenience function that automatically processes an ECG signal.

        Parameters
        ----------
        ecg_signal : Union[list, np.array, pd.Series]
            The raw ECG channel.
        sampling_rate : int
            The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
            Defaults to 1000.

        Returns
        -------
        signals : DataFrame
            A DataFrame of the same length as the `ecg_signal` containing the following columns:

            - *"ECG_Raw"*: the raw signal.

            - *"ECG_Clean"*: the cleaned signal.

            - *"ECG_R_Peaks"*: the R-peaks marked as "1" in a list of zeros.

            - *"ECG_Rate"*: heart rate interpolated between R-peaks.

            - *"ECG_P_Peaks"*: the P-peaks marked as "1" in a list of zeros

            - *"ECG_Q_Peaks"*: the Q-peaks marked as "1" in a list of zeros .

            - *"ECG_S_Peaks"*: the S-peaks marked as "1" in a list of zeros.

            - *"ECG_T_Peaks"*: the T-peaks marked as "1" in a list of zeros.

            - *"ECG_P_Onsets"*: the P-onsets marked as "1" in a list of zeros.

            - *"ECG_P_Offsets"*: the P-offsets marked as "1" in a list of zeros
                                (only when method in `ecg_delineate` is wavelet).

            - *"ECG_T_Onsets"*: the T-onsets marked as "1" in a list of zeros
                                (only when method in `ecg_delineate` is wavelet).

            - *"ECG_T_Offsets"*: the T-offsets marked as "1" in a list of zeros.

            - *"ECG_R_Onsets"*: the R-onsets marked as "1" in a list of zeros
                                (only when method in `ecg_delineate` is wavelet).

            - *"ECG_R_Offsets"*: the R-offsets marked as "1" in a list of zeros
                                (only when method in `ecg_delineate` is wavelet).

            - *"ECG_Phase_Atrial"*: cardiac phase, marked by "1" for systole
            and "0" for diastole.

            - *"ECG_Phase_Ventricular"*: cardiac phase, marked by "1" for systole and "0" for diastole.

            - *"ECG_Atrial_PhaseCompletion"*: cardiac phase (atrial) completion, expressed in percentage
            (from 0 to 1), representing the stage of the current cardiac phase.

            - *"ECG_Ventricular_PhaseCompletion"*: cardiac phase (ventricular) completion, expressed in
            percentage (from 0 to 1), representing the stage of the current cardiac phase.
        info : dict
            A dictionary containing the samples at which the R-peaks occur, accessible with the key
            "ECG_Peaks".

        """
  

        # Sanitize input
        ecg_signal = self.signal.signal_sanitize(self.signal.data)

        ecg_cleaned = self.ecg_clean(ecg_signal, sampling_rate=self.signal.fs, method=c_method)

        if gaussian != None:
            ecg_cleaned = gaussian_filter1d(ecg_cleaned, gaussian)

        #ecg_cleaned = scale_factor2(2,1,ecg_cleaned)
        # R-peaks 
        peaks = self.ecg_peaks(
            ecg_cleaned=ecg_cleaned, sampling_rate=self.signal.fs, correct_artifacts=True, method=p_method,
        )


        rate = self.signal.signal_rate(peaks, sampling_rate=self.signal.fs, desired_length=len(ecg_cleaned))

        quality, beats = self.ecg_quality(ecg_cleaned, rpeaks=peaks, sampling_rate=self.signal.fs)

        custom_beats = {}
        
        for i in range(beats.shape[1]):
            custom_beats[str(i)] = beats[:,i]
        

        # Additional info of the ecg signal
        _, delineate_info = self.ecg_delineate(
            ecg_cleaned=ecg_cleaned, rpeaks=peaks, sampling_rate=self.signal.fs, method='dwt')
        
        return quality, peaks, delineate_info, custom_beats, rate


    def ecg_quality(self):
        """Quality of ECG Signal.

        Compute a continuous index of quality of the ECG signal, by interpolating the distance
        of each QRS segment from the average QRS segment present in the data. This index is
        therefore relative, and 1 corresponds to heartbeats that are the closest to the average
        sample and 0 corresponds to the most distance heartbeat, from that average sample.

        Returns
        -------
        array
            Vector containing the quality index ranging from 0 to 1.

        
        """

        def epochs_to_df(epochs):
            """Convert epochs to a DataFrame.

            Parameters
            ----------
            epochs : dict
                A dict containing one DataFrame per event/trial. Usually obtained via `epochs_create()`.


            Returns
            ----------
            DataFrame
                A DataFrame containing all epochs identifiable by the 'Label' column, which time axis
                is stored in the 'Time' column.

            """
            data = pd.concat(epochs)
            data["Time"] = data.index.get_level_values(1).values
            data = data.reset_index(drop=True)

            return data

        from .stats import distance, rescale
        # Sanitize inputs
        if self.peaks is None:
            rpeaks = self.ecg_peaks(self.signal.data, sampling_rate=self.signal.fs)

        # Get heartbeats
        heartbeats, beats = self.ecg_segment(self.signal.data, rpeaks, self.signal.fs)
        data = epochs_to_df(heartbeats).pivot(index="Label", columns="Time", values="Signal")
        data.index = data.index.astype(int)
        data = data.sort_index()

        # Filter Nans
        missing = data.T.isnull().sum().values
        nonmissing = np.where(missing == 0)[0]

        data = data.iloc[nonmissing, :]

        # Compute distance
        dist = distance(data, method="mean")
        dist = rescale(np.abs(dist), to=[0, 1])
        dist = np.abs(dist - 1)  # So that 1 is top quality

        # Replace missing by 0
        quality = np.zeros(len(heartbeats))
        quality[nonmissing] = dist

        # Interpolate
        quality = self.signal.signal_interpolate(rpeaks, quality, x_new=np.arange(len(self.signal.data)), method="quadratic")
                        
        return quality, beats


    def ecg_segment(self):
        """Segment an ECG signal into single heartbeats.

        Parameters
        ----------
        ecg_cleaned : Union[list, np.array, pd.Series]
            The cleaned ECG channel as returned by `ecg_clean()`.
        rpeaks : dict
            The samples at which the R-peaks occur. Dict returned by
            `ecg_peaks()`. Defaults to None.
        sampling_rate : int
            The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
            Defaults to 1000.
        Returns
        -------
        dict
            A dict containing DataFrames for all segmented heartbeats.

        """

        from .signal import signal_rate
        from .stats import listify
        

        def _events_find_label(events, event_labels=None, event_conditions=None, function_name="events_find"):
            # Get n events
            n = len(events["onset"])

            # Labels
            if event_labels is None:
                event_labels = (np.arange(n) + 1).astype(np.str)

            if len(list(set(event_labels))) != n:
                raise ValueError(
                    "EPK error: "
                    + function_name
                    + "(): oops, it seems like the `event_labels` that you provided "
                    + "are not unique (all different). Please provide "
                    + str(n)
                    + " distinct labels."
                )

            if len(event_labels) != n:
                raise ValueError(
                    "EPK error: "
                    + function_name
                    + "(): oops, it seems like you provided "
                    + str(n)
                    + " `event_labels`, but "
                    + str(n)
                    + " events got detected :(. Check your event names or the event signal!"
                )

            events["label"] = event_labels

            # Condition
            if event_conditions is not None:
                if len(event_conditions) != n:
                    raise ValueError(
                        "EPK error: "
                        + function_name
                        + "(): oops, it seems like you provided "
                        + str(n)
                        + " `event_conditions`, but "
                        + str(n)
                        + " events got detected :(. Check your event conditions or the event signal!"
                    )
                events["condition"] = event_conditions
            return events


        def epochs_create( data, events=None,  sampling_rate=1000, epochs_start=0, epochs_end=1, event_labels=None, event_conditions=None, baseline_correction=False, ):
            """Epoching a dataframe.

            Parameters
            ----------
            data : DataFrame
                A DataFrame containing the different signal(s) as different columns.
                If a vector of values is passed, it will be transformed in a DataFrame
                with a single 'Signal' column.
            events : list or ndarray or dict
                Events onset location. If a dict is passed (e.g., from ``events_find()``),
                will select only the 'onset' list. If an integer is passed,
                will use this number to create an evenly spaced list of events. If None,
                will chunk the signal into successive blocks of the set duration.
            sampling_rate : int
                The sampling frequency of the signal (in Hz, i.e., samples/second).
            epochs_start : int
                Epochs start relative to events_onsets (in seconds). The start can be negative to
                start epochs before a given event (to have a baseline for instance).
            epochs_end : int
                Epochs end relative to events_onsets (in seconds).
            event_labels : list
                A list containing unique event identifiers. If `None`, will use the event index number.
            event_conditions : list
                An optional list containing, for each event, for example the trial category, group or
                experimental conditions.
            baseline_correction : bool
                Defaults to False.


            Returns
            ----------
            dict
                A dict containing DataFrames for all epochs.

            """

            # Santize data input
            if isinstance(data, tuple):  # If a tuple of data and info is passed
                data = data[0]

            if isinstance(data, (list, np.ndarray, pd.Series)):
                data = pd.DataFrame({"Signal": list(data)})

            # Sanitize events input
            if events is None:
                max_duration = (np.max(epochs_end) - np.min(epochs_start)) * sampling_rate
                events = np.arange(0, len(data) - max_duration, max_duration)
            if isinstance(events, int):
                events = np.linspace(0, len(data), events + 2)[1:-1]
            if isinstance(events, dict) is False:
                events = _events_find_label({"onset": events}, event_labels=event_labels, event_conditions=event_conditions)

            event_onsets = list(events["onset"])
            event_labels = list(events["label"])
            if "condition" in events.keys():
                event_conditions = list(events["condition"])

            # Create epochs
            parameters = listify(
                onset=event_onsets, label=event_labels, condition=event_conditions, start=epochs_start, end=epochs_end
            )

            # Find the maximum numbers of samples in an epoch
            parameters["duration"] = np.array(parameters["end"]) - np.array(parameters["start"])
            epoch_max_duration = int(max((i * sampling_rate for i in parameters["duration"])))

            # Extend data by the max samples in epochs * NaN (to prevent non-complete data)
            length_buffer = epoch_max_duration
            buffer = pd.DataFrame(index=range(length_buffer), columns=data.columns)
            data = data.append(buffer, ignore_index=True, sort=False)
            data = buffer.append(data, ignore_index=True, sort=False)

            # Adjust the Onset of the events for the buffer
            parameters["onset"] = [i + length_buffer for i in parameters["onset"]]

            epochs = {}
            for i, label in enumerate(parameters["label"]):

                # Find indices
                start = parameters["onset"][i] + (parameters["start"][i] * sampling_rate)
                end = parameters["onset"][i] + (parameters["end"][i] * sampling_rate)

                # Slice dataframe
                epoch = data.iloc[int(start) : int(end)].copy()

                # Correct index
                epoch["Index"] = epoch.index.values - length_buffer
                epoch.index = np.linspace(
                    start=parameters["start"][i], stop=parameters["end"][i], num=len(epoch), endpoint=True
                )

                if baseline_correction is True:
                    baseline_end = 0 if epochs_start <= 0 else epochs_start
                    epoch = epoch - epoch.loc[:baseline_end].mean()

                # Add additional
                epoch["Label"] = parameters["label"][i]
                if parameters["condition"][i] is not None:
                    epoch["Condition"] = parameters["condition"][i]

                # Store
                epochs[label] = epoch

            return epochs

        def _ecg_segment_window( rpeaks=None, sampling_rate=1000, desired_length=None):

            heart_rate = np.mean(signal_rate(rpeaks, sampling_rate=sampling_rate, desired_length=desired_length))

            # Modulator
            m = heart_rate / 60

            # Window
            epochs_start = -0.35 / m
            epochs_end = 0.5 / m

            # Adjust for high heart rates
            if heart_rate >= 80:
                c = 0.1
                epochs_start = epochs_start - c
                epochs_end = epochs_end + c

            return epochs_start, epochs_end



        # Sanitize inputs
        if self.peaks is None:
            rpeaks = self.ecg_peaks(self.signal.data, sampling_rate=self.signal.fs, correct_artifacts=True)

        epochs_start, epochs_end = _ecg_segment_window(
            rpeaks=rpeaks, sampling_rate=self.signal.fs, desired_length=len(self.signal.data)
        )
        #print(epochs_start,epochs_end)
        heartbeats = epochs_create(
            self.signal.data, rpeaks, sampling_rate=self.signal.fs, epochs_start=epochs_start, epochs_end=epochs_end
        )


        heartbeats_plot = self.ecg_quality.epochs_to_df(heartbeats)
        heartbeats_pivoted = heartbeats_plot.pivot(index="Time", columns="Label", values="Signal")

        beats=[]
        for i in range(heartbeats_pivoted.shape[1]):
            x = heartbeats_pivoted[heartbeats_pivoted.columns[i]]
            #print(x.shape)
            beats.append(scipy.signal.resample(x, 100, t=None, axis=0, window=None, domain='time'))
            #custom_beats[str(i)] = x
        
        beats = np.array(beats)
        #print(beats.shape)

        return heartbeats,beats




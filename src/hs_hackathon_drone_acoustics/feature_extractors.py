from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Final

import librosa
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from hs_hackathon_drone_acoustics.base import AudioWaveform

logger: Final = logging.getLogger(__name__)


def convert_to_spectrogram(waveform: AudioWaveform, n_fft: int = 2048, hop_length: int = 512) -> NDArray[np.float64]:
    """Convert audio waveform to spectrogram (magnitude spectrum over time)."""
    stft = librosa.stft(waveform.data.numpy(), n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    return spectrogram


def convert_to_mel_spectrogram(waveform: AudioWaveform, n_mels: int = 128, n_fft: int = 2048, hop_length: int = 512) -> NDArray[np.float64]:
    """Convert audio waveform to mel-scaled spectrogram."""
    mel_spec = librosa.feature.melspectrogram(
        y=waveform.data.numpy(), 
        sr=waveform.sample_rate, 
        n_mels=n_mels, 
        n_fft=n_fft, 
        hop_length=hop_length
    )
    return mel_spec


def convert_to_spectrum(waveform: AudioWaveform) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert audio waveform to frequency spectrum (single snapshot)."""
    fft = np.fft.fft(waveform.data.numpy())
    magnitude = np.abs(fft)
    frequencies = np.fft.fftfreq(len(fft), 1/waveform.sample_rate)
    # Return only positive frequencies
    n_positive = len(frequencies) // 2
    return frequencies[:n_positive], magnitude[:n_positive]


def extract_amplitude_features_from_spectrogram(spectrogram: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    """Extract various amplitude features from a spectrogram."""
    return {
        "mean_amplitude_per_frequency": np.mean(spectrogram, axis=1),  # Average amplitude for each frequency bin
        "max_amplitude_per_frequency": np.max(spectrogram, axis=1),   # Peak amplitude for each frequency bin
        "mean_amplitude_per_time": np.mean(spectrogram, axis=0),      # Average amplitude for each time frame
        "max_amplitude_per_time": np.max(spectrogram, axis=0),        # Peak amplitude for each time frame
        "total_energy": np.sum(spectrogram**2),                       # Total energy in the spectrogram
        "spectral_centroid": np.sum(spectrogram * np.arange(spectrogram.shape[0])[:, np.newaxis], axis=0) / np.sum(spectrogram, axis=0)  # Frequency center of mass over time
    }


def extract_peak_frequencies_and_amplitudes(spectrogram: NDArray[np.float64], sample_rate: float, n_peaks: int = 5) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract the top N peak frequencies and their amplitudes from a spectrogram."""
    # Average across time to get overall frequency spectrum
    avg_spectrum = np.mean(spectrogram, axis=1)
    
    # Find peak indices
    peak_indices = np.argsort(avg_spectrum)[-n_peaks:][::-1]  # Top N peaks
    
    # Convert indices to frequencies
    freq_resolution = sample_rate / (2 * spectrogram.shape[0])
    peak_frequencies = peak_indices * freq_resolution
    peak_amplitudes = avg_spectrum[peak_indices]
    
    return peak_frequencies, peak_amplitudes


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, waveform: AudioWaveform) -> NDArray[np.float64]: ...


class EnergyFeatureExtractor(FeatureExtractor):
    def extract(self, waveform: AudioWaveform) -> NDArray[np.float64]:
        energy = np.linalg.vector_norm(waveform.data, ord=2)
        return np.array([energy])


class MFCCFeatureExtractor(FeatureExtractor):
    def __init__(self, n_mfcc: int = 20):
        self._n_mfcc = n_mfcc

    def extract(self, waveform: AudioWaveform) -> NDArray[np.float64]:
        mfccs = librosa.feature.mfcc(y=waveform.data.numpy(), sr=waveform.sample_rate, n_mfcc=self._n_mfcc)
        ensemble_mfcc: NDArray[np.float64] = np.mean(mfccs, axis=1)
        return ensemble_mfcc


class FeatureExtractorPipeline(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    def __init__(self, *feature_extractors: FeatureExtractor) -> None:
        super().__init__()
        self._feature_extractors = feature_extractors

    def fit(self, *args: Any, **kwargs: Any) -> FeatureExtractorPipeline:
        return self

    def transform(self, waveforms: list[AudioWaveform]) -> NDArray[np.float64]:
        if not isinstance(waveforms, list):
            raise TypeError("Input x must be a list of AudioWaveform objects.")
        all_features = []
        for waveform in waveforms:
            waveform_features = []
            for feature_extractor in self._feature_extractors:
                waveform_features.append(feature_extractor.extract(waveform))
            all_features.append(np.concatenate(waveform_features))
        return np.stack(all_features)

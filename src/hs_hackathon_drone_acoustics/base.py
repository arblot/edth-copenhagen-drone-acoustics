from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
from torch import Tensor


@dataclass(frozen=True)
class AudioWaveform:
    data: Tensor
    sample_rate: float

    @classmethod
    def load(cls, path: Path) -> AudioWaveform:
        data, samplerate = sf.read(path)
        return AudioWaveform(torch.as_tensor(data, dtype=torch.float32), samplerate)

    @property
    def duration(self) -> float:
        return self.data.shape[-1] / self.sample_rate

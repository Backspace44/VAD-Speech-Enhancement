"""Common enhancement pipeline helpers used by demo and evaluation scripts."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import numpy as np
import torch

from src.dsp.spectral_subtraction import enhance_waveform as spectral_subtraction_enhance
from src.dsp.wiener_filter import enhance_waveform as wiener_filter_enhance
from src.models import enhance_audio_with_masknet


EnhancementFn = Callable[[np.ndarray], np.ndarray]


def build_enhancement_methods(model=None, device: torch.device | None = None) -> "OrderedDict[str, EnhancementFn]":
    """Return the shared set of enhancement methods in display order."""
    methods: "OrderedDict[str, EnhancementFn]" = OrderedDict()
    methods["Noisy"] = lambda noisy_audio: noisy_audio
    methods["Spectral_Subtraction"] = spectral_subtraction_enhance
    methods["Wiener_Filter"] = wiener_filter_enhance

    if model is not None and device is not None:
        methods["MaskNet"] = lambda noisy_audio: enhance_audio_with_masknet(noisy_audio, model, device)

    return methods


def run_enhancement_methods(
    noisy_audio: np.ndarray,
    model=None,
    device: torch.device | None = None,
) -> "OrderedDict[str, np.ndarray]":
    """Run all configured enhancement methods on the same noisy waveform."""
    methods = build_enhancement_methods(model=model, device=device)
    return OrderedDict((name, method(noisy_audio)) for name, method in methods.items())

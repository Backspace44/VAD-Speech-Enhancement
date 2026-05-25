"""Shared inference utilities for MaskNet-based enhancement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src import config
from src.dsp.stft_utils import (
    apply_complex_mask,
    complex_to_channels,
    compute_stft,
    inverse_complex_stft,
    inverse_stft,
)
from src.models.mask_model import MaskNet


def load_masknet_checkpoint(checkpoint_path: str | Path, device: torch.device) -> MaskNet:
    """Load a MaskNet model from a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", {})

    model = MaskNet(
        in_channels=model_config.get("in_channels", 1),
        base_channels=model_config.get("base_channels", 32),
        use_lstm=model_config.get("use_lstm", True),
        use_attention=model_config.get("use_attention", True),
        use_residual=model_config.get("use_residual", False),
        use_depthwise=model_config.get("use_depthwise", False),
        output_channels=model_config.get("output_channels", 1),
        output_activation=model_config.get("output_activation", "sigmoid"),
        output_scale=model_config.get("output_scale", 1.0),
        mask_type=model_config.get("mask_type", "magnitude"),
        complex_mask_clip=model_config.get("complex_mask_clip", 5.0),
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        raise KeyError(
            f"Checkpoint must contain 'model_state_dict' or 'model_state', found: {list(checkpoint.keys())}"
        )

    model.to(device)
    model.eval()
    return model


def enhance_audio_with_masknet(
    noisy_audio: np.ndarray,
    model: MaskNet,
    device: torch.device,
    n_fft: int | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
) -> np.ndarray:
    """Enhance a noisy waveform with a trained MaskNet model."""
    noisy_tensor = torch.from_numpy(noisy_audio).float().to(device)
    n_fft = n_fft or config.N_FFT
    hop_length = hop_length or config.HOP_LEN
    win_length = win_length or config.FRAME_LEN

    mask_type = getattr(model, "mask_type", "magnitude")
    autocast_enabled = device.type == "cuda"

    if mask_type == "complex":
        noisy_complex = compute_stft(
            noisy_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            return_complex=True,
        )
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                noisy_complex_input = complex_to_channels(noisy_complex).unsqueeze(0)
                predicted_mask = model(noisy_complex_input)

        enhanced_complex = apply_complex_mask(noisy_complex, predicted_mask.squeeze(0))
        enhanced_audio = inverse_complex_stft(
            enhanced_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            length=len(noisy_audio),
        )
    else:
        noisy_mag, noisy_phase = compute_stft(
            noisy_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                noisy_mag_input = noisy_mag.unsqueeze(0).unsqueeze(0)
                predicted_mask = model(noisy_mag_input)

        enhanced_mag = noisy_mag * predicted_mask.squeeze(0).squeeze(0)
        enhanced_audio = inverse_stft(
            enhanced_mag,
            noisy_phase,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            length=len(noisy_audio),
        )
    return enhanced_audio.detach().cpu().numpy()

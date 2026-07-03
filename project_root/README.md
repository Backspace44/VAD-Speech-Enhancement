# Voice Activity Detection & Speech Enhancement

This project is a complete speech enhancement pipeline built for my bachelor
thesis. It combines classic DSP methods with MaskNet-style neural models, adds
Voice Activity Detection, and includes a realtime interface for comparing raw
and enhanced audio.

The main idea is simple: start from noisy speech, estimate where the useful
speech information is in the time-frequency domain, and reconstruct a cleaner
signal.

## What Is In The Repository

```text
src/
  data_prep/      dataset validation, metadata, VAD labels, augmentation
  dsp/            STFT/ISTFT, metrics, spectral subtraction, Wiener filter
  models/         MaskNet and inference helpers
  train/          training setup, loop and checkpoint handling
  eval/           evaluation, comparisons and report helpers
  tools/          CLI, healthcheck, realtime demo

docs/
  assets/         university logos and small static assets
  DELIVERY_STRUCTURE.md

demo.py           small offline demo entry point
launch_realtime_demo.cmd
requirements.txt
```

The repo does not include large/generated files. These are expected to exist
locally when needed:

```text
data/             LibriSpeech, DEMAND, VoiceBank-DEMAND
checkpoints/      trained model checkpoints
results/          metrics, plots, snapshots, recordings
logs/             runtime logs
export/           delivery/export packages
```

PDFs and LaTeX working files are also kept outside the project tree to keep the
source repository focused on code.

## Setup

Create a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

The project was developed on Windows, so the default configuration keeps data
loader workers conservative to avoid multiprocessing issues.

## Quick Checks

Before training or running the demo, check that the local data paths are valid:

```bash
python -m src.tools.project_cli healthcheck -- --dataset librispeech
```

To inspect available audio devices for the realtime demo:

```bash
python -m src.tools.project_cli realtime-demo -- --list-devices
```

## Training

The recommended training setup uses LibriSpeech mixed with DEMAND noise and
soft VAD labels:

```bash
python -m src.data_prep.generate_vad_labels --split all --label-set adaptive_soft_v1 --label-mode soft
python -m src.train.train_mask_model --experiment-preset librispeech_soft_vad_recommended
```

The complex-mask variant can be trained with:

```bash
python -m src.train.train_mask_model --experiment-preset librispeech_complex_masknet_recommended
```

Short smoke runs are useful when checking that the pipeline still works:

```bash
python -m src.train.train_mask_model --experiment-preset librispeech_fast_benchmark --max-batches 30
```

Training writes checkpoints and summaries under local ignored folders such as
`checkpoints/` and `results/`.

## Evaluation

VoiceBank-DEMAND is used as the fixed benchmark set. A typical full comparison
run looks like this:

```bash
python -m src.tools.project_cli eval-all -- --clean-dir data/voicebank_demand/clean/test --noisy-dir data/voicebank_demand/noisy/test
```

Useful comparison commands:

```bash
python -m src.tools.project_cli compare-vad -- --split test --max-files 20
python -m src.tools.project_cli compare-latest -- --clean-dir data/voicebank_demand/clean/test --noisy-dir data/voicebank_demand/noisy/test --reference baseline=checkpoints/masknet_best.pth --max-files 20
python -m src.tools.project_cli project-inventory
```

The final thesis results were interpreted using objective metrics, spectrograms
and listening tests. The most relevant metrics are PESQ, STOI, SNR, segmental
SNR and Log Spectral Distance.

## Realtime Demo

The easiest way to start the live interface on Windows is:

```bash
launch_realtime_demo.cmd
```

Or run it manually:

```bash
python -m src.tools.project_cli realtime-demo -- --method masknet --visualize
```

For recording raw and enhanced audio:

```bash
python -m src.tools.project_cli realtime-demo -- --method masknet --visualize --record
```

Recordings are written locally under `results/`. They are not committed to Git.

The UI supports:

- microphone or speech+noise input;
- MaskNet, Spectral Subtraction, Wiener Filter and bypass;
- raw/enhanced waveforms and spectrograms;
- VAD display;
- SNR, dry/wet and output gain controls;
- scenario presets for easier live demonstrations.

If MaskNet is too heavy on the current machine, the same interface can be used
with `spectral_subtraction` or `wiener`.

## Notes

The repository is meant to stay lightweight. Keep datasets, trained weights,
recordings, generated plots, PDFs and exported packages outside source control.
Only source code, small static assets and human-maintained documentation should
be committed.

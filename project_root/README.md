# Speech Enhancement - Deep Learning

Noise reduction system combining deep learning (U-Net + LSTM + Attention) with classic DSP methods.

Current recommended training setup: `stage2_recommended` = `balanced_res + AdamW + OneCycleLR + MSE`.
Current recommended LibriSpeech VAD setup: `librispeech_soft_vad_recommended`.
Training/validation uses LibriSpeech-DEMAND only. VoiceBank-DEMAND is reserved for benchmarking, checkpoint comparisons, and plots.
Official training presets: `stage2_recommended`, `librispeech_fast_benchmark`, `librispeech_soft_vad_recommended`, `librispeech_complex_masknet_recommended`.

## Quick Start

```bash
pip install -r requirements.txt

# Stage-0 health check for the training dataset
python -m src.tools.healthcheck --dataset librispeech

# Unified project CLI
python -m src.tools.project_cli healthcheck -- --dataset librispeech

# Recommended training (stage 2)
python -m src.train.train_mask_model --experiment-preset stage2_recommended --max-batches 30

# Same recommended command via unified CLI
python -m src.tools.project_cli train -- --experiment-preset stage2_recommended --max-batches 30

# Recommended LibriSpeech training with soft VAD gating
python -m src.train.train_mask_model --experiment-preset librispeech_soft_vad_recommended --max-batches 30

# Complex MaskNet training on LibriSpeech with soft VAD labels
python -m src.data_prep.generate_vad_labels --split all --label-set adaptive_soft_v1 --label-mode soft
python -m src.train.train_mask_model --experiment-preset librispeech_complex_masknet_recommended --max-batches 30

# Fast LibriSpeech benchmark preset
python -m src.train.train_mask_model --experiment-preset librispeech_fast_benchmark --max-batches 30

# Experimental tuned weighted loss from stage 2
python -m src.train.train_mask_model --loss-preset weighted_combined_tuned --max-batches 30 --epochs 1

# Stage-2 optimizer/scheduler smoke test
python -m src.train.train_mask_model --preset onecycle_test --optimizer-preset adamw --max-batches 30 --epochs 1

# Stage-2 residual architecture experiment
python -m src.train.train_mask_model --model-variant balanced_res --optimizer-preset adamw --preset onecycle_test --max-batches 30 --epochs 1

# Recommended stage-2 command
python -m src.train.train_mask_model --experiment-preset stage2_recommended --max-batches 30

# Compare two checkpoints on the same evaluation subset
python -m src.tools.project_cli compare-checkpoints -- --clean-dir data/voicebank_demand/clean/test --noisy-dir data/voicebank_demand/noisy/test --checkpoint baseline=checkpoints/baseline.pth --checkpoint stage2=checkpoints/stage2_recommended.pth --max-files 20

# Compare latest experiment directly against baseline checkpoint
python -m src.tools.project_cli compare-latest -- --clean-dir data/voicebank_demand/clean/test --noisy-dir data/voicebank_demand/noisy/test --reference baseline=checkpoints/masknet_best.pth --max-files 20

# Aggregate final training summaries
python -m src.tools.project_cli summarize-benchmarks -- --experiment librispeech_fast_benchmark_smoke --experiment librispeech_soft_vad_recommended_smoke

# Build a delivery-ready artifact inventory
python -m src.tools.project_cli project-inventory

# Demo
python demo.py --checkpoint checkpoints/masknet_best.pth

# Realtime microphone -> denoised speakers demo
python -m src.tools.project_cli realtime-demo -- --list-devices
python -m src.tools.project_cli realtime-demo -- --method masknet
python -m src.tools.project_cli realtime-demo -- --method masknet --visualize
python -m src.tools.project_cli realtime-demo -- --method masknet --visualize --record
python -m src.tools.project_cli realtime-demo -- --method masknet --visualize --record --fullscreen

# VAD evaluation
python -m src.eval.evaluate_vad_baseline

# Generate soft VAD confidence labels for LibriSpeech
python -m src.data_prep.generate_vad_labels --split all --label-set adaptive_soft_v1 --label-mode soft

# Compare no VAD vs hard/soft VAD target-mask behavior
python -m src.tools.project_cli compare-vad -- --split test --max-files 20
```

## Architecture

**MaskNet**: U-Net (4 levels) + Bidirectional LSTM + Channel Attention
- Variants: Tiny (200K) to XLarge (10M params)
- Recommended now: `balanced_res` with residual encoder/decoder blocks

**Complex MaskNet**: residual MaskNet variant that uses real/imaginary STFT input channels and predicts a two-channel complex ratio mask (CRM). The enhanced waveform is reconstructed from the estimated complex spectrogram instead of reusing only the noisy phase.

**Classic methods**: Spectral Subtraction, Wiener Filter, VAD (88.3% accuracy)

## Datasets

- LibriSpeech + DEMAND: training and validation for neural models, with clean/noise mixed on the fly
- VoiceBank-DEMAND: fixed clean/noisy benchmark set for evaluation, comparisons, and plots
- LibriSpeech/DEMAND preparation resamples clean speech and noise on the fly to 16 kHz when source files use a different sample rate.

## Results

**VAD**: F1=0.794, Acc=88.3% (averaged across SNR levels)

**Enhancement targets**:
- PESQ: 2.75+ (balanced) / 2.95+ (full)
- STOI: 0.91+ / 0.93+
- SNR: 13+ dB / 15+ dB

## Project Structure

```
src/
├── train/train_mask_model.py    # Training with --max-batches, --resume
├── models/mask_model.py          # U-Net architecture
├── dsp/                          # Classic methods + metrics
├── eval/                         # Evaluation scripts
└── config.py                     # Central config

demo.py                           # Quick demo
checkpoints/                      # Saved models
results/                          # Metrics and plots
```

## Delivery Hygiene

Use the inventory command to classify artifacts as `official`, `benchmark`, `smoke`, `demo`, or `legacy`:

```bash
python -m src.tools.project_cli project-inventory
```

Delivery notes and the recommended final artifacts are documented in `docs/DELIVERY_STRUCTURE.md`.

## Configuration

**Windows fix** (num_workers=0 in config.py to avoid multiprocessing errors)

**Model variants**: tiny, tiny_fast, balanced baseline, balanced_res (stage-2 recommended), large

## Training

```bash
# Recommended stage-2 preset
python -m src.train.train_mask_model --experiment-preset stage2_recommended --max-batches 30

# LibriSpeech experiment with hard adaptive VAD labels
python -m src.train.train_mask_model --dataset librispeech --use-vad-labels --vad-label-set adaptive_v1 --model-variant balanced_res --optimizer-preset adamw --preset onecycle_test --max-batches 30

# Recommended LibriSpeech experiment with soft adaptive VAD gating
python -m src.train.train_mask_model --experiment-preset librispeech_soft_vad_recommended --max-batches 30

# Complex MaskNet experiment with real/imaginary complex ratio masks
python -m src.data_prep.generate_vad_labels --split all --label-set adaptive_soft_v1 --label-mode soft
python -m src.train.train_mask_model --experiment-preset librispeech_complex_masknet_recommended --max-batches 30

# Fast LibriSpeech benchmark preset
python -m src.train.train_mask_model --experiment-preset librispeech_fast_benchmark --max-batches 30

# Resume
python -m src.train.train_mask_model --resume checkpoints/masknet_best.pth

# Full
python -m src.train.train_mask_model --epochs 100

# OneCycleLR + AdamW short experiment
python -m src.train.train_mask_model --preset onecycle_test --optimizer-preset adamw --max-batches 30 --epochs 1

# Residual MaskNet variant
python -m src.train.train_mask_model --model-variant balanced_res --optimizer-preset adamw --preset onecycle_test --max-batches 30 --epochs 1

# Throughput-oriented tiny variant for long sequences
python -m src.train.train_mask_model --model-variant tiny_fast --max-length-sec 1 --max-batches 1 --epochs 1

# Same preset via unified CLI
python -m src.tools.project_cli train -- --experiment-preset stage2_recommended --max-batches 30
```

Saves checkpoints every 50 batches, best model when val_loss improves.
Each training run now also writes:
- `checkpoints/<experiment>/training_history.json`
- `checkpoints/<experiment>/summary.json`
- `results/experiment_index.csv`

## Evaluation

```bash
python -m src.eval.evaluate_vad_baseline
python -m src.eval.compare_vad_label_sets --split test --max-files 20
python -m src.eval.evaluate_all_methods
python -m src.eval.summarize_benchmarks --experiment librispeech_fast_benchmark_smoke --experiment librispeech_soft_vad_recommended_smoke
python demo.py --checkpoint checkpoints/masknet_best.pth

# Unified CLI variants
python -m src.tools.project_cli eval-vad
python -m src.tools.project_cli compare-vad -- --split test --max-files 20
python -m src.tools.project_cli eval-all -- --clean-dir data/voicebank_demand/clean/test --noisy-dir data/voicebank_demand/noisy/test
python -m src.tools.project_cli compare-checkpoints -- --clean-dir data/voicebank_demand/clean/test --noisy-dir data/voicebank_demand/noisy/test --checkpoint baseline=checkpoints/baseline.pth --checkpoint stage2=checkpoints/stage2_recommended.pth --max-files 20
python -m src.tools.project_cli compare-latest -- --clean-dir data/voicebank_demand/clean/test --noisy-dir data/voicebank_demand/noisy/test --reference baseline=checkpoints/masknet_best.pth --max-files 20
python -m src.tools.project_cli summarize-benchmarks -- --experiment librispeech_fast_benchmark_smoke --experiment librispeech_soft_vad_recommended_smoke
python -m src.tools.project_cli demo -- --checkpoint checkpoints/masknet_best.pth
python -m src.tools.project_cli realtime-demo -- --method masknet
```

Realtime demo notes:
- default backend uses the best local `MaskNet` checkpoint if available
- use `--list-devices` first if you need to pick a specific mic/speaker pair
- add `--visualize` to open a live interface with raw microphone vs enhanced output
- the live UI now shows raw/output waveforms, raw/output spectrograms, live VAD confidence, plus a method selector and stop button
- add `--record` to save `raw_microphone.wav` and `enhanced_output.wav` under `results/audio_samples/realtime_demo_<timestamp>/`
- the UI can now save PNG snapshots and show the active checkpoint/method; use `--fullscreen` for presentations
- the UI also supports `Add Speech File`, `Add Noise File`, source switching (`microphone` / `speech+noise`), distortion toggles from `data_prep`, VAD mode selection, and denoising method selection
- the `speech+noise` source now has a live SNR slider, so you can make the mix easier or harder without restarting the demo
- the UI also includes `Dry/Wet`, `Output Gain`, and scenario presets (`neutral`, `office`, `street`, `hard`) for faster live demos
- if the neural model is too heavy on CPU, try `--method spectral_subtraction` or `--method wiener`
- headphones usually work better than laptop speakers because they reduce acoustic feedback

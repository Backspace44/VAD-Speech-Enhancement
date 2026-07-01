# Delivery Structure

This project keeps generated artifacts out of source control, but it now uses a simple convention so the final deliverables are easy to identify.

## Main Folders

`src/`
- Final source code for training, evaluation, DSP, VAD, and realtime demo.

`checkpoints/`
- Saved training runs and exported model checkpoints.
- Recommended final LibriSpeech-trained checkpoint:
  `librispeech_soft_vad_recommended_smoke/masknet_best.pth`

`results/`
- Evaluation reports, benchmark tables, VAD studies, demo recordings, and generated inventories.

`logs/`
- Runtime logs for training, evaluation, and realtime demo sessions.

## Artifact Categories

Checkpoint and results artifacts are classified into a few practical groups:

`official`
- Main artifacts that should be cited in the thesis or demo.

`benchmark`
- Named comparison runs or benchmark reports that are still useful for analysis.

`smoke`
- Short validation runs used during development.

`demo`
- Realtime demo audio, snapshots, and UI artifacts.

`legacy`
- Older historical runs kept for traceability.

## Inventory Command

Generate a machine-readable inventory with:

```bash
python -m src.tools.project_cli project-inventory
```

This writes:

- `results/project_inventory/artifact_inventory.csv`
- `results/project_inventory/artifact_inventory.json`
- `results/project_inventory/recommended_artifacts.csv`

## Recommended Final Artifacts

The current project-level references are:

1. `checkpoints/librispeech_soft_vad_recommended_smoke/masknet_best.pth`
2. `checkpoints/librispeech_fast_benchmark_smoke/masknet_best.pth`
3. `results/final_benchmark_report_v2/benchmark_best_by_dataset.csv`
4. `results/voicebank_quality_v2_vs_baseline/checkpoint_comparison.csv`
5. `results/audio_samples/`

These are the most useful artifacts to mention in the thesis, presentation, and demo workflow.

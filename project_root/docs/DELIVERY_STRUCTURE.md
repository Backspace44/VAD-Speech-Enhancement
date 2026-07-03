# Delivery Notes

The source repository is intentionally small. It contains the code and the small
assets needed to understand or rerun the project, but not the generated
artifacts.

## Kept In Git

- source code under `src/`;
- launch scripts and requirements;
- small documentation files;
- small static assets, such as the university logos.

## Kept Locally

These folders are produced during development, training or demo runs and should
stay outside source control:

```text
data/             datasets
checkpoints/      trained models
results/          metrics, plots, snapshots, recordings
logs/             runtime logs
export/           packaged/demo delivery copies
```

PDFs, LaTeX drafts and compiled presentation files are also kept outside
`project_root` after cleanup.

## Useful Local Artifacts

For the thesis presentation, the most useful local artifacts are:

- the final MaskNet checkpoint;
- benchmark CSV/JSON files from `results/`;
- saved audio examples;
- UI snapshots from the realtime demo;
- the generated presentation and speech PDFs, kept outside the repository.

To regenerate an artifact inventory locally:

```bash
python -m src.tools.project_cli project-inventory
```

The inventory output is useful for choosing what to present, but the generated
files themselves should not be committed.

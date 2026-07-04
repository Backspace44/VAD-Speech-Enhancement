# Voice Activity Detection & Speech Enhancement

Proiect de licenta pentru reducerea zgomotului din semnale vocale. Sistemul
pleaca de la metode clasice de procesare digitala a semnalului si ajunge la un
model MaskNet antrenat pentru estimarea mastilor spectrale.

Lucrarea urmareste un flux complet, nu doar un model izolat: pregatirea
datelor, VAD, STFT/ISTFT, baseline-uri DSP, antrenare MaskNet, evaluare pe
metrici obiective si o interfata real-time pentru demonstratie.

## Stare curenta

Versiunea folosita pentru prezentarea finala este bazata pe experimentul:

```text
train_balanced_res_20260614_091134
```

Checkpoint-ul final asteptat local este:

```text
checkpoints/train_balanced_res_20260614_091134/masknet_best.pth
```

Acest fisier nu este inclus in repository, deoarece checkpoint-urile sunt mari
si sunt tinute local.

## Ce contine proiectul

```text
src/
  data_prep/       validare dataset, metadata, augmentare, etichete VAD
  dsp/             STFT, ISTFT, metrici, Spectral Subtraction, Wiener Filter
  models/          MaskNet si utilitare de inferenta
  train/           configurare, antrenare, checkpointing
  eval/            evaluare, comparatii, grafice si tabele
  tools/           CLI, healthcheck, demo real-time
  utils/           functii comune pentru audio, logging si raportare

docs/
  assets/          logo-uri si fisiere mici folosite in documentatie
  DELIVERY_STRUCTURE.md

demo.py            demo offline pentru prezentare
launch_realtime_demo.cmd
requirements.txt
```

Folderele de mai jos sunt folosite local, dar nu sunt urcate pe GitHub:

```text
data/              LibriSpeech, DEMAND, VoiceBank-DEMAND
checkpoints/       modele antrenate
results/           evaluari, grafice, snapshot-uri, inregistrari
logs/              log-uri de rulare
export/            pachete de demo/copii portabile
```

## Instalare

Proiectul a fost dezvoltat pe Windows. Pentru rularea demo-ului real-time am
folosit Python 3.12, iar pentru restul codului merge si Python 3.11.

```powershell
cd project_root
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Verificarea rapida a dataset-ului:

```powershell
python -m src.tools.project_cli healthcheck -- --dataset librispeech
```

Listarea device-urilor audio pentru demo:

```powershell
py -3.12 -m src.tools.realtime_demo --list-devices
```

## Date si preprocesare

Antrenarea foloseste LibriSpeech combinat cu zgomote din DEMAND. Mixarea se
face controlat, la SNR variabil. Pentru evaluarea finala am folosit
VoiceBank-DEMAND.

Etichetele VAD pot fi generate cu:

```powershell
python -m src.data_prep.generate_vad_labels --split all --label-set adaptive_soft_v1 --label-mode soft
```

Setarile principale de semnal sunt:

```text
sample rate: 16 kHz
fereastra STFT: 25 ms
hop: 10 ms
NFFT: 512
benzi frecventa: 257
```

## Metode implementate

Baseline-urile DSP sunt:

- Spectral Subtraction
- Wiener Filter
- bypass, pentru comparatie directa cu semnalul zgomotos

Modelul principal este MaskNet in varianta `balanced_res`. Arhitectura foloseste
un U-Net rezidual, skip connections, BiLSTM si attention. In proiect exista si
variante mai mici pentru rulare rapida, plus o varianta complexa
`complex_balanced_res` pentru estimarea unei masti complexe.

## Antrenare

Presetul recomandat pentru modelul final:

```powershell
python -m src.train.train_mask_model --experiment-preset librispeech_soft_vad_recommended
```

Varianta complexa:

```powershell
python -m src.train.train_mask_model --experiment-preset librispeech_complex_masknet_recommended
```

Pentru verificari rapide:

```powershell
python -m src.train.train_mask_model --experiment-preset librispeech_fast_benchmark --max-batches 30
```

Antrenarea scrie checkpoint-uri in `checkpoints/` si log-uri in `logs/`. Acestea
raman locale.

## Evaluare

Evaluarea principala compara semnalul zgomotos, metodele clasice si MaskNet:

```powershell
python -m src.tools.project_cli eval-all -- --clean-dir data\voicebank_demand\clean\test --noisy-dir data\voicebank_demand\noisy\test --model checkpoints\train_balanced_res_20260614_091134\masknet_best.pth --output-dir results\evaluation
```

Pentru o evaluare mai scurta:

```powershell
python -m src.tools.project_cli eval-all -- --clean-dir data\voicebank_demand\clean\test --noisy-dir data\voicebank_demand\noisy\test --model checkpoints\train_balanced_res_20260614_091134\masknet_best.pth --output-dir results\evaluation_demo --max-files 20
```

Comenzi utile pentru comparatii:

```powershell
python -m src.tools.project_cli compare-vad -- --split test --max-files 20
python -m src.tools.project_cli compare-latest -- --clean-dir data\voicebank_demand\clean\test --noisy-dir data\voicebank_demand\noisy\test --reference final=checkpoints\train_balanced_res_20260614_091134\masknet_best.pth --max-files 20
python -m src.tools.project_cli project-inventory
```

## Rezultate finale

Rezultatele de mai jos sunt mediile obtinute pe evaluarea finala folosita in
lucrare.

| Metoda | PESQ | STOI | SNR (dB) | SegSNR (dB) | LSD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Noisy | 1.972 | 0.921 | 8.447 | 1.548 | 13.923 |
| Spectral Subtraction | 2.289 | 0.920 | 10.051 | 3.043 | 11.483 |
| Wiener Filter | 2.150 | 0.922 | 8.971 | 2.119 | 12.025 |
| MaskNet | 2.564 | 0.929 | 15.328 | 6.673 | 9.502 |

Imbunatatirea MaskNet fata de semnalul zgomotos:

```text
PESQ:   +0.592
STOI:   +0.008
SNR:    +6.882 dB
SegSNR: +5.125 dB
LSD:    -4.421
```

PESQ si STOI masoara calitatea perceptuala si inteligibilitatea. SNR si SegSNR
masoara raportul semnal-zgomot, iar LSD masoara distanta spectrala fata de
semnalul curat. Pentru LSD, mai mic inseamna mai bun.

## Demo offline

Demo-ul offline incarca o pereche clean/noisy, ruleaza VAD, aplica metodele de
enhancement si exporta grafice plus fisiere audio:

```powershell
python -m src.tools.project_cli demo -- --checkpoint checkpoints\train_balanced_res_20260614_091134\masknet_best.pth --test-dir data\voicebank_demand\test --output-dir demo_results
```

Rezultatele apar in `demo_results/`:

```text
demo_waveforms.png
demo_vad.png
demo_metrics.png
demo_spectrograms.png
audio/*.wav
```

## Demo real-time

Pornirea rapida pe Windows:

```powershell
launch_realtime_demo.cmd
```

Scriptul foloseste checkpoint-ul final, porneste MaskNet pe CUDA, deschide
interfata grafica, activeaza recording-ul si salveaza snapshot-uri in
`results/realtime_snapshots/`.

Output-ul audio poate fi ales din linia de comanda:

```powershell
launch_realtime_demo.cmd laptop
launch_realtime_demo.cmd speakers
launch_realtime_demo.cmd ath
launch_realtime_demo.cmd stream
launch_realtime_demo.cmd 5
```

Rulare manuala, cu aceleasi setari stabile folosite la demo:

```powershell
py -3.12 -m src.tools.realtime_demo --method masknet --checkpoint checkpoints\train_balanced_res_20260614_091134\masknet_best.pth --device cuda --visualize --record --dry-wet 1.0 --block-ms 512 --context-ms 2048 --tail-ms 64 --latency high --queue-size 256 --stats-interval 2 --display-ms 6000 --input-device parsec --output-device laptop --record-dir results\realtime_recordings --snapshot-dir results\realtime_snapshots
```

Interfata permite:

- intrare din microfon sau din fisiere speech + noise;
- comutare intre bypass, Spectral Subtraction, Wiener Filter si MaskNet;
- vizualizare waveform si spectrograma pentru raw/enhanced;
- VAD, SNR, dry/wet, gain si scenarii de test;
- snapshot PNG si inregistrare raw/enhanced.

Pentru un calculator fara placa NVIDIA, se poate folosi `--device cpu`, cu
asteptarea ca latenta sa fie mai mare.

## Observatii pentru repository

Repository-ul este pastrat intentionat mic. Nu include dataset-uri, checkpoint-uri,
rezultate generate, PDF-uri finale sau pachete exportate. Acestea se regenereaza
local sau se pastreaza separat pentru prezentare.

Codul sursa, scripturile de rulare si documentatia mica sunt partea care trebuie
versionata. Artefactele mari raman in afara Git-ului.

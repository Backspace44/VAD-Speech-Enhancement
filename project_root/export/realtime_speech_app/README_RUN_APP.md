# Realtime Speech Enhancement App

Aplicatie portabila Windows pentru demo-ul realtime cu modelul MaskNet antrenat.

## Rulare pe un laptop nou

1. Instaleaza Python 3.11 sau 3.12 de pe python.org.
2. Bifeaza `Add python.exe to PATH` in installer.
3. Ruleaza `install_windows.cmd`.
4. Porneste aplicatia cu `launch_realtime_demo.cmd`.

Pentru laptop fara placa NVIDIA sau daca apar probleme CUDA, foloseste:

```bat
launch_realtime_demo_cpu.cmd
```

## Folosire

- `Audio`: alegi microfonul si iesirea audio, apoi `Apply Audio Devices`.
- `Files`: alegi fisiere speech/noise si folosesti `Pause Files` / `Play Files`.
- `Processing`: alegi metoda de filtrare si poti face snapshot-uri.

Rezultatele de recording/snapshot sunt salvate in `results/`.

## Continut

- `src/`: codul aplicatiei.
- `checkpoints/train_balanced_res_20260614_091134/masknet_best.pth`: modelul final.
- `requirements_realtime.txt`: dependentele necesare pentru rulare.

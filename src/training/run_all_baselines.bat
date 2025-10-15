@echo off
set FEATURES=..\..\data\window_features\all_window_features.csv
set LABELS=..\..\data\labels\weak_labels.csv
set OUTDIR=..\..\runs

echo Running mode=full
python train_baselines.py --features %FEATURES% --labels %LABELS% --mode full --outdir %OUTDIR%\full

echo Running mode=clean_only
python train_baselines.py --features %FEATURES% --labels %LABELS% --mode clean_only --outdir %OUTDIR%\clean

echo Running mode=conf_weighted
python train_baselines.py --features %FEATURES% --labels %LABELS% --mode conf_weighted --outdir %OUTDIR%\confw

echo.
echo All experiments finished. Results saved in %OUTDIR%
pause

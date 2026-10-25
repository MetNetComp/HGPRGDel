# Quick-run entry points (`quickrun/`)

Run from the **repository root**.

## Verify layout (after unpacking `Data/`)

```bash
python quickrun/verify_data_layout.py --quick-start-only
python quickrun/verify_release_checkpoints.py
```

## Inference

```bash
python quickrun/quick_run_HGPRGDel.py --CBM e_coli_core
python quickrun/quick_run_baseline_GraphGdel.py
python quickrun/quick_run_baseline_DNN.py
python quickrun/quick_run_baseline_DeepGDel.py
python quickrun/quick_run_baseline_NMA.py
```

| Script | Weights |
|--------|---------|
| `quick_run_HGPRGDel.py` | `checkpoints/model_<GEM>.sav` (all three GEMs) |
| `quick_run_baseline_*.py` | `baseline/*_e_coli_core*.sav` |

Unpack `Data/` first (`Data/README.md`). Delete local `runs/` after use.

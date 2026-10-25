# HGPRGDel pretrained weights (`checkpoints/`)

Shipped **in this GitHub repository** (no external archive):

| File | Purpose |
|------|---------|
| `checkpoints/model_e_coli_core.sav` | Quick-run HGPRGDel weights for *e_coli_core* |
| `checkpoints/model_iMM904.sav` | Quick-run HGPRGDel weights for *iMM904* |
| `checkpoints/model_iML1515.sav` | Quick-run HGPRGDel weights for *iML1515* |

Used by `quickrun/quick_run_HGPRGDel.py` when `--train` is not set:

```bash
python quickrun/quick_run_HGPRGDel.py --CBM e_coli_core
python quickrun/quick_run_HGPRGDel.py --CBM iMM904
python quickrun/quick_run_HGPRGDel.py --CBM iML1515
```

Verify:

```bash
python quickrun/verify_release_checkpoints.py
```

**Baseline** weights live under `baseline/*.sav` (see [`baseline/README_CHECKPOINTS.md`](../baseline/README_CHECKPOINTS.md)).

**Not in this folder:** five-fold training dumps, ablation discs, or intermediate `.pth` files. Those belong outside the public tree (author archive only).

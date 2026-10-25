# HGPRGDel datasets (`Data/`)

Shipped in GitHub as **three ZIP archives** (file-count / size friendly).  
After clone, **unzip once** into this directory, then verify.

## Unpack (required after clone)

From the repository root:

```bash
cd Data
unzip -o e_coli_core.zip
unzip -o iMM904.zip
unzip -o iML1515.zip
cd ..
```

Or only the quick-start GEM:

```bash
cd Data && unzip -o e_coli_core.zip && cd ..
```

Each archive expands to `Data/<GEM>/` with JSON, `Gene_AA/`, `Label/`, and SMILES tables **ready to use** (no further nested unzip).

## Verify, then run

```bash
python quickrun/verify_data_layout.py --quick-start-only   # e_coli_core only
python quickrun/verify_data_layout.py                      # all three GEMs
python quickrun/verify_release_checkpoints.py
python quickrun/quick_run_HGPRGDel.py --CBM e_coli_core
```

## Archives in this folder

| Archive | Expands to |
|---------|------------|
| `e_coli_core.zip` | `Data/e_coli_core/` |
| `iMM904.zip` | `Data/iMM904/` |
| `iML1515.zip` | `Data/iML1515/` |


| GEM | Approx. reactions | Supervised targets |
|-----|-------------------|--------------------|
| *e_coli_core* | 95 | 42 |
| *iMM904* | 1,577 | 106 |
| *iML1515* | 2,712 | 409 |

Each `Data/<GEM>/` directory includes:

| Path | Description |
|------|-------------|
| `<GEM>.json` | COBRA-style model: stoichiometry, reactions, **GPR rules** |
| `Target metabolites_smile.csv` | SMILES for supervised production targets. Left as originally published apart from 22 entries (4 on *iMM904*, 18 on *iML1515*) that stored the string `c` — a failed-lookup placeholder that RDKit rejects — and are now empty. The baselines refit a character-level tokenizer on this file at run time, so its vocabulary and maximum length are baked into the shipped baseline checkpoints; blanking those 22 values leaves both unchanged, whereas replacing the file wholesale would not. |
| `All metabolites_smile.csv` | SMILES for every compound node of the hypergraph. Columns: `Metabolite`, `Smiles`, `MatchedKey`, `HasSMILES`. |
| `Gene_AA/<gene_id>.csv` | Amino acid sequences |
| `Label/<metabolite_id>.csv` | Binary deletion labels from MetNetComp |
| `SMILES_node_feature_final.csv` | Baseline graph features (*e_coli_core* only). **Legacy artifact**: 64-dimensional latents produced once by GraphGDel's SMILES-LSTM autoencoder and frozen — neither GraphGDel nor this repository regenerates them. They cover 62 of the 72 compounds and are *not* derived from the SMILES tables above; metabolites absent from the file receive a zero feature row. The GraphGDel and NMA checkpoints in `baseline/` are tied to this exact latent space, so regenerating it by any means would invalidate them. |

## Sources

Labels: [MetNetComp](https://www.genome.jp/metnetcomp/database1/indexFiles/index.html)

Metabolite structures in `Target metabolites_smile.csv` and
`All metabolites_smile.csv` are derived from
[BiGG Models](http://bigg.ucsd.edu/) (Norsigian *et al.*, *Nucleic Acids
Research* 2020) and [MetaNetX/MNXref](https://www.metanetx.org/) (Moretti *et
al.*, *Nucleic Acids Research* 2021), both licensed CC-BY 4.0. The tables are
shipped ready to use; the lookup tables and the resolution script used to build
them are not part of this release.

Cross-validation folds (seed 42): [`utils/cv_partitions.json`](../utils/cv_partitions.json)

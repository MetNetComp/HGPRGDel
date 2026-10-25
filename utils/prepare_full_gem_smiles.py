#!/usr/bin/env python3
"""Prepare per-GEM metabolite SMILES tables for HGPRGDel.

Three stages:

  resolve          (default, fully offline)
      Data/reference/* + Data/<GEM>/<GEM>.json  ->  Data/<GEM>/All metabolites_smile.csv

  build-reference  (maintainers only; needs the GraphGDel aux tree and/or network)
      Rebuilds Data/reference/ from the upstream BiGG summary tables and the
      MetaNetX chem_prop / chem_depr dumps.

  report           (offline)
      Coverage, per-tier source breakdown and optional RDKit validation.

This is a maintainer tool.  The lookup tables it needs are NOT shipped with the
release -- the repository carries the resolved `All metabolites_smile.csv` and
`Target metabolites_smile.csv` tables instead -- so `resolve` exits with an
error unless `--reference_dir` points at a directory holding them.

An earlier revision depended on an `--aux_data_root` that lived outside the
repository; when that argument was omitted the global mapping silently became
empty and coverage collapsed to whatever each GEM's own target table happened to
contain.  The guardrails below exist so that failure mode cannot recur silently:
a reference directory that is requested but absent is fatal, never skipped.

Resolution tiers, highest priority first.  Every candidate must pass the
formula gate (see `heavy_atoms`) before it is accepted; a candidate that fails
falls through to the next tier.

  T0  curated_overrides.csv     manual corrections
  T1  MetaNetX bigg_struct      authoritative BiGG structures, keyed by base id
  T2  MetaNetX chem_prop        via annotation.metanetx.chemical / bigg_xref
  T3  BiGG reference (scraped)  exact id, then base id
  T4  annotation.bigg.metabolite
  T5  old BiGG id aliases
  T6  InChIKey -> chem_prop
  T7  the GEM's own "Target metabolites_smile.csv"
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

GEMS = ("e_coli_core", "iMM904", "iML1515")

# `resolve` refuses to write anything below these values.  Coverage is an
# assertion, not a log line.  The numbers sit just under what the current
# reference set achieves (100.0 / 86.3 / 89.9), so a regression trips them while
# normal variation does not.  They are not aspirational: the remainder are
# compounds MetaNetX itself has no structure for, plus polymers and carrier
# proteins whose model formula is a generic repeat unit.
MIN_COVERAGE = {"e_coli_core": 1.00, "iMM904": 0.85, "iML1515": 0.88}

REFERENCE_FILES = (
    "bigg_struct_smiles.csv",
    "bigg_smiles_reference.csv",
    "bigg_xref.csv",
    "bigg_old_ids.csv",
    "mnx_smiles_subset.csv",
    "mnx_deprecated_subset.csv",
    "curated_overrides.csv",
)

MNX_STRUCT_URL = "https://www.metanetx.org/ftp/latest/bigg_struct.ttl.gz"
MNX_PROP_URL = "https://www.metanetx.org/ftp/latest/chem_prop.tsv"
MNX_DEPR_URL = "https://www.metanetx.org/ftp/latest/chem_depr.tsv"

# The shipped table keeps the four columns the pipeline has always had. Which
# tier supplied a structure, and whether it is a wildcard, are diagnostics: the
# `report` command recomputes them rather than shipping two columns no consumer
# reads.
OUTPUT_FIELDS = ["Metabolite", "Smiles", "MatchedKey", "HasSMILES"]

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
_SMILES_OK = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$%/\\.*:~]+$")
# R / X / Z stand for "some residue" in GEM formulas; they are not elements.
_PLACEHOLDER_RE = re.compile(r"(?<![A-Za-z])[RXZ](?![a-z])")


# --------------------------------------------------------------------------- #
# identifiers and formulas
# --------------------------------------------------------------------------- #
def metabolite_id_keys(metabolite_id: str) -> List[str]:
    """Return [full id, compartment-stripped base], lower-cased."""
    met = str(metabolite_id).strip().lower()
    keys = [met]
    if "_" in met:
        base, compartment = met.rsplit("_", 1)
        if 1 <= len(compartment) <= 2:
            keys.append(base)
    return keys


def heavy_atoms(formula: Optional[str]) -> Optional[collections.Counter]:
    """Element counts excluding hydrogen, with any trailing charge removed.

    Returns None when the formula cannot be compared meaningfully, in which case
    callers must let the candidate through.

    Hydrogen is ignored on purpose.  BiGG states metabolites in the ionic form
    used by the model (``C7H7O4``, charge -1) while structure databases usually
    store the neutral species (``C7H8O4``).  That difference is a convention,
    not an error, and must not reject an otherwise correct structure.  A
    mismatch in the heavy-atom skeleton *is* an error -- it means the two
    records describe different molecules.

    Formulas carrying an ``R``/``X``/``Z`` placeholder are not comparable at
    all: acyl-carrier proteins and charged tRNAs are written ``C11H21N2O7PRS``
    in the model but expanded explicitly in any real structure, so the counts
    legitimately differ.
    """
    if not formula:
        return None
    formula = re.sub(r"[+-]\d*$", "", str(formula).strip())
    if _PLACEHOLDER_RE.search(formula):
        return None
    counts: collections.Counter = collections.Counter()
    for element, number in _FORMULA_RE.findall(formula):
        if not element or element == "H":
            continue
        counts[element] += int(number or 1)
    return counts or None


def smiles_quality(smiles: str) -> str:
    """Classify a SMILES string without requiring RDKit."""
    s = (smiles or "").strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return "missing"
    if not _SMILES_OK.match(s):
        return "missing"
    if s.count("(") != s.count(")") or s.count("[") != s.count("]"):
        return "missing"
    # A structure made only of wildcards carries no chemistry.  Keeping it would
    # collapse every such metabolite onto one identical embedding, which is
    # worse than leaving the field empty.
    if not re.sub(r"[\*\[\]\(\)]", "", s):
        return "missing"
    # A bare lowercase aromatic atom is not a valid molecule.  GraphGDel's
    # exporter wrote "c" whenever a lookup failed, and that sentinel is still
    # present in the shipped target tables.
    if s in {"c", "n", "o", "s", "p", "b"}:
        return "missing"
    if "*" in s:
        return "wildcard"
    return "exact"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _read_csv(path: str, required: bool = True) -> List[dict]:
    if not os.path.exists(path):
        if required:
            sys.exit(
                f"ERROR: required reference table not found: {path}\n"
                f"       Run `{os.path.basename(__file__)} build-reference --help`."
            )
        return []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if required and not rows:
        sys.exit(f"ERROR: reference table loaded 0 rows: {path}")
    return rows


class Reference:
    """Every lookup table `resolve` needs, all read from Data/reference/."""

    def __init__(self, ref_dir: str, quiet: bool = False) -> None:
        self.dir = ref_dir
        if not os.path.isdir(ref_dir):
            sys.exit(f"ERROR: reference directory not found: {ref_dir}")

        self.struct: Dict[str, str] = {}
        for row in _read_csv(os.path.join(ref_dir, "bigg_struct_smiles.csv")):
            base = (row.get("bigg_base") or "").strip().lower()
            smiles = (row.get("Smiles") or "").strip()
            if base and smiles:
                self.struct.setdefault(base, smiles)

        rows = _read_csv(os.path.join(ref_dir, "bigg_smiles_reference.csv"))
        self.bigg: Dict[str, str] = {}
        for row in rows:
            bigg_id = (row.get("bigg_id") or "").strip().lower()
            smiles = (row.get("Smiles") or "").strip()
            if bigg_id and smiles:
                self.bigg.setdefault(bigg_id, smiles)
        self.bigg_base: Dict[str, str] = {}
        for bigg_id, smiles in self.bigg.items():
            for key in metabolite_id_keys(bigg_id)[1:]:
                self.bigg_base.setdefault(key, smiles)

        self.xref: Dict[str, str] = {}
        for row in _read_csv(os.path.join(ref_dir, "bigg_xref.csv")):
            bigg_id = (row.get("bigg_id") or "").strip().lower()
            mnx = (row.get("MNX_id") or "").strip()
            if bigg_id and mnx:
                self.xref.setdefault(bigg_id, mnx)

        self.alias: Dict[str, str] = {}
        for row in _read_csv(os.path.join(ref_dir, "bigg_old_ids.csv")):
            old = (row.get("old_bigg_id") or "").strip().lower()
            new = (row.get("bigg_id") or "").strip().lower()
            if old and new:
                self.alias.setdefault(old, new)

        self.mnx: Dict[str, dict] = {}
        self.by_inchikey: Dict[str, dict] = {}
        for row in _read_csv(os.path.join(ref_dir, "mnx_smiles_subset.csv")):
            mnx_id = (row.get("MNX_id") or "").strip()
            if not mnx_id:
                continue
            self.mnx.setdefault(mnx_id, row)
            key = (row.get("InChIKey") or "").strip()
            if key:
                self.by_inchikey.setdefault(key, row)

        self.deprecated: Dict[str, List[str]] = collections.defaultdict(list)
        for row in _read_csv(os.path.join(ref_dir, "mnx_deprecated_subset.csv"), required=False):
            old, new = (row.get("deprecated_ID") or "").strip(), (row.get("ID") or "").strip()
            if old and new:
                self.deprecated[old].append(new)

        self.overrides: Dict[str, str] = {}
        for row in _read_csv(os.path.join(ref_dir, "curated_overrides.csv"), required=False):
            base = (row.get("base_id") or "").strip().lower()
            smiles = (row.get("Smiles") or "").strip()
            if base and smiles:
                self.overrides[base] = smiles

        if not quiet:
            print(
                f"reference: struct={len(self.struct)} bigg={len(self.bigg)} xref={len(self.xref)} "
                f"alias={len(self.alias)} mnx={len(self.mnx)} "
                f"inchikey={len(self.by_inchikey)} overrides={len(self.overrides)}"
            )


def load_json_metabolites(path: str) -> List[dict]:
    if not os.path.exists(path):
        sys.exit(f"ERROR: GEM model not found: {path}\n       Unzip Data/<GEM>.zip first.")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle).get("metabolites", [])


def load_target_table(path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for row in _read_csv(path, required=False):
        met = row.get("Metabolite") or row.get("metabolite") or row.get("bigg_id")
        smiles = row.get("Smiles") or row.get("SMILES") or row.get("smiles")
        if not met or not smiles:
            continue
        smiles = str(smiles).strip()
        if not smiles or smiles.lower() == "nan":
            continue
        for key in metabolite_id_keys(met):
            mapping.setdefault(key, smiles)
    return mapping


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
def _mnx_candidates(mnx_id: str, ref: Reference) -> List[dict]:
    """Expand one MNX id through the deprecation table into concrete records."""
    seen, out = set(), []
    for candidate in [mnx_id, *ref.deprecated.get(mnx_id, [])]:
        row = ref.mnx.get(candidate)
        if row and candidate not in seen:
            seen.add(candidate)
            out.append(row)
    return out


def resolve_metabolite(met: dict, ref: Reference, targets: Dict[str, str]) -> Tuple[str, str, str, str]:
    """Return (smiles, matched_key, source, quality) for one GEM metabolite."""
    met_id = met.get("id", "")
    keys = metabolite_id_keys(met_id)
    annotation = met.get("annotation") or {}
    want = heavy_atoms(met.get("formula"))

    def accept(smiles: str) -> bool:
        """Formula gate: reject anything whose skeleton contradicts the model."""
        if smiles_quality(smiles) == "missing":
            return False
        formula = _rdkit_formula(smiles)
        if formula is _UNPARSEABLE:
            return False          # RDKit is present and says this is not a molecule
        if want is None or formula is None:
            return True           # nothing to compare against, or RDKit unavailable
        got = heavy_atoms(formula)
        return got is None or got == want

    # T0 -- curated overrides bypass the gate; they are the ground truth.
    for key in keys:
        if key in ref.overrides:
            smiles = ref.overrides[key]
            return smiles, key, "curated", smiles_quality(smiles)

    # T1 -- MetaNetX's own BiGG structures.  These are keyed by the
    # compartment-free base id and are the most trustworthy source available:
    # the scraped table in T3 is known to carry identifier collisions.
    for key in keys:
        smiles = ref.struct.get(key)
        if smiles and accept(smiles):
            return smiles, key, "bigg_struct", smiles_quality(smiles)

    # T2 -- MetaNetX chem_prop.
    mnx_ids: List[str] = list(annotation.get("metanetx.chemical") or [])
    for key in keys:
        if key in ref.xref and ref.xref[key] not in mnx_ids:
            mnx_ids.append(ref.xref[key])
    for mnx_id in mnx_ids:
        for row in _mnx_candidates(mnx_id, ref):
            smiles = (row.get("Smiles") or "").strip()
            if not smiles:
                continue
            got = heavy_atoms(row.get("formula"))
            if want is not None and got is not None and got != want:
                continue
            if smiles_quality(smiles) == "missing":
                continue
            return smiles, row.get("MNX_id", mnx_id), "mnx", smiles_quality(smiles)

    # T2/T3 -- the BiGG reference table.
    for source, table in (("bigg", ref.bigg), ("bigg_base", ref.bigg_base)):
        for key in keys:
            smiles = table.get(key)
            if smiles and accept(smiles):
                return smiles, key, source, smiles_quality(smiles)

    # T4 -- the official compartment-free base name from the model annotation.
    for base in annotation.get("bigg.metabolite") or []:
        smiles = ref.bigg.get(base.lower()) or ref.bigg_base.get(base.lower())
        if smiles and accept(smiles):
            return smiles, base.lower(), "annotation_bigg", smiles_quality(smiles)

    # T5 -- retired BiGG identifiers.
    for key in keys:
        target = ref.alias.get(key)
        if not target:
            continue
        smiles = ref.bigg.get(target) or ref.bigg_base.get(target)
        if smiles and accept(smiles):
            return smiles, target, "bigg_alias", smiles_quality(smiles)

    # T6 -- InChIKey.
    for inchikey in annotation.get("inchi_key") or []:
        row = ref.by_inchikey.get(inchikey.strip())
        if not row:
            continue
        smiles = (row.get("Smiles") or "").strip()
        if smiles and accept(smiles):
            return smiles, inchikey.strip(), "inchikey", smiles_quality(smiles)

    # T7 -- whatever the GEM shipped for its own supervised targets.
    for key in keys:
        smiles = targets.get(key)
        if smiles and accept(smiles):
            return smiles, key, "target_table", smiles_quality(smiles)

    return "", "", "missing", "missing"


class _Unparseable:
    """Sentinel: RDKit is installed and rejected the string."""

    def __repr__(self) -> str:  # pragma: no cover
        return "<unparseable>"


_UNPARSEABLE = _Unparseable()
_RDKIT_FORMULA_CACHE: Dict[str, object] = {}


def _rdkit_formula(smiles: str):
    """Molecular formula via RDKit.

    Returns the formula string, ``_UNPARSEABLE`` when RDKit is available but
    rejects the input, or None when RDKit is not installed.  The distinction
    matters: an unparseable structure must be rejected, whereas a missing RDKit
    must not silently reject everything.  `resolve` therefore runs without
    RDKit, just with a weaker gate.
    """
    if smiles in _RDKIT_FORMULA_CACHE:
        return _RDKIT_FORMULA_CACHE[smiles]
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        value = CalcMolFormula(mol) if mol is not None else _UNPARSEABLE
    except ImportError:
        value = None
    _RDKIT_FORMULA_CACHE[smiles] = value
    return value


def resolve_gem(
    gem: str, data_dir: str, ref: Reference
) -> Tuple[List[dict], collections.Counter, collections.Counter]:
    gem_dir = os.path.join(data_dir, gem)
    metabolites = load_json_metabolites(os.path.join(gem_dir, f"{gem}.json"))
    targets = load_target_table(os.path.join(gem_dir, "Target metabolites_smile.csv"))

    rows: List[dict] = []
    sources: collections.Counter = collections.Counter()
    qualities: collections.Counter = collections.Counter()
    for met in metabolites:
        smiles, key, source, quality = resolve_metabolite(met, ref, targets)
        sources[source] += 1
        rows.append(
            {
                "Metabolite": met.get("id", ""),
                "Smiles": smiles,
                "MatchedKey": key,
                "HasSMILES": "1" if smiles else "0",
            }
        )
        qualities[quality] += 1
    return rows, sources, qualities


def write_rows(path: str, rows: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_resolve(args: argparse.Namespace) -> int:
    ref = Reference(args.reference_dir)
    results, failed = {}, []
    for gem in GEMS:
        rows, sources, _qualities = resolve_gem(gem, args.data_dir, ref)
        matched = sum(1 for row in rows if row["HasSMILES"] == "1")
        coverage = matched / len(rows) if rows else 0.0
        floor = MIN_COVERAGE.get(gem, 0.0)
        results[gem] = (rows, matched, len(rows), coverage, sources)
        flag = "" if coverage >= floor else f"  << below floor {floor:.0%}"
        print(f"{gem:<14}{matched:>5}/{len(rows):<5} {coverage:6.1%}{flag}")
        print(f"{'':<14}{dict(sources)}")
        if coverage < floor:
            failed.append(gem)

    if failed and not args.allow_low_coverage:
        print(
            f"\nERROR: coverage below floor for {', '.join(failed)}; nothing written.\n"
            "       Use --allow-low-coverage to override deliberately.",
            file=sys.stderr,
        )
        return 2

    for gem, (rows, *_rest) in results.items():
        out = os.path.join(args.data_dir, gem, "All metabolites_smile.csv")
        write_rows(out, rows)
        print(f"wrote {out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Coverage and provenance.

    Resolution is re-run rather than read back from the CSV, so the per-tier
    breakdown stays available without shipping columns nothing consumes.
    """
    ref = Reference(args.reference_dir, quiet=True)
    for gem in GEMS:
        rows, sources, quality = resolve_gem(gem, args.data_dir, ref)
        matched = sum(1 for row in rows if row["HasSMILES"] == "1")
        print(f"\n{gem}: {matched}/{len(rows)} = {matched / len(rows):.1%}")
        print(f"  source  {dict(sources)}")
        print(f"  quality {dict(quality)}")

        if args.validate_rdkit:
            metabolites = {
                m["id"]: m
                for m in load_json_metabolites(os.path.join(args.data_dir, gem, f"{gem}.json"))
            }
            unparsed, mismatched = [], []
            for row in rows:
                smiles = (row.get("Smiles") or "").strip()
                if not smiles or "*" in smiles:
                    continue
                formula = _rdkit_formula(smiles)
                if formula is _UNPARSEABLE:
                    unparsed.append(row["Metabolite"])
                    continue
                if formula is None:
                    continue
                want = heavy_atoms(metabolites.get(row["Metabolite"], {}).get("formula"))
                if want is not None and heavy_atoms(formula) != want:
                    mismatched.append(row["Metabolite"])
            print(f"  rdkit   unparsed={len(unparsed)} heavy-atom-mismatch={len(mismatched)}")
            if mismatched:
                print(f"          {mismatched[:12]}")
    return 0


def cmd_build_reference(args: argparse.Namespace) -> int:
    """Rebuild Data/reference/ from the upstream tables (maintainers only)."""
    from urllib.request import urlopen

    ref_dir = args.reference_dir
    os.makedirs(ref_dir, exist_ok=True)

    if args.aux_data_root:
        summary = os.path.join(args.aux_data_root, "all_bigg_compound_summary")
        if not os.path.isdir(summary):
            sys.exit(f"ERROR: --aux_data_root given but not found: {summary}")

        seen: Dict[str, tuple] = {}
        for row in _read_csv(os.path.join(summary, "all_compound_IDs_SMILES.csv")):
            bigg_id = (row.get("bigg_id") or "").strip()
            smiles = (row.get("Smiles") or "").strip()
            if bigg_id and smiles and smiles.lower() != "nan":
                seen.setdefault(
                    bigg_id,
                    (smiles, row.get("MNX_id", ""), row.get("KEGG_id", ""), row.get("SEED_id", "")),
                )
        with open(os.path.join(ref_dir, "bigg_smiles_reference.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["bigg_id", "Smiles", "MNX_id", "KEGG_id", "SEED_id"])
            for key in sorted(seen):
                writer.writerow([key, *seen[key]])
        print(f"bigg_smiles_reference.csv {len(seen)}")

    if args.refresh_struct:
        import gzip
        import io as _io

        print(f"downloading {MNX_STRUCT_URL}")
        with urlopen(MNX_STRUCT_URL) as stream:
            blob = stream.read()
        subject, smiles = None, {}
        subject_re = re.compile(r"^biggM:(\S+)")
        smiles_re = re.compile(r'mnx:stdSMILES\s+"([^"]+)"')
        with gzip.open(_io.BytesIO(blob), "rt", errors="replace") as handle:
            for line in handle:
                if line.startswith("@prefix"):
                    continue
                found = subject_re.match(line)
                if found:
                    subject = found.group(1)
                if subject:
                    value = smiles_re.search(line)
                    if value:
                        smiles.setdefault(subject, value.group(1))
        with open(os.path.join(ref_dir, "bigg_struct_smiles.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["bigg_base", "Smiles"])
            for key in sorted(smiles):
                writer.writerow([key, smiles[key]])
        print(f"bigg_struct_smiles.csv {len(smiles)}")

    if args.refresh_mnx:
        wanted = set()
        for gem in GEMS:
            for met in load_json_metabolites(os.path.join(args.data_dir, gem, f"{gem}.json")):
                wanted.update((met.get("annotation") or {}).get("metanetx.chemical") or [])
        print(f"streaming {MNX_DEPR_URL}")
        depr: Dict[str, List[str]] = collections.defaultdict(list)
        with urlopen(MNX_DEPR_URL) as stream:
            for raw in stream:
                line = raw.decode("utf-8", "replace")
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0] in wanted:
                    depr[parts[0]].append(parts[1])
        with open(os.path.join(ref_dir, "mnx_deprecated_subset.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["deprecated_ID", "ID"])
            for key in sorted(depr):
                for value in sorted(depr[key]):
                    writer.writerow([key, value])
        for values in depr.values():
            wanted.update(values)

        print(f"streaming {MNX_PROP_URL} ({len(wanted)} ids wanted)")
        out = []
        with urlopen(MNX_PROP_URL) as stream:
            for raw in stream:
                line = raw.decode("utf-8", "replace")
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 9 and parts[0] in wanted:
                    out.append([parts[0], parts[3], parts[4], parts[7], parts[8]])
        with open(os.path.join(ref_dir, "mnx_smiles_subset.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["MNX_id", "formula", "charge", "InChIKey", "Smiles"])
            writer.writerows(sorted(out))
        print(f"mnx_smiles_subset.csv {len(out)}")

    return 0


def main() -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Shared so that --data_dir works both before and after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data_dir", default=os.path.join(here, "Data"))
    common.add_argument("--reference_dir", default=os.path.join(here, "Data", "reference"))

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], parents=[common])
    sub = parser.add_subparsers(dest="command")

    resolve = sub.add_parser("resolve", parents=[common], help="rebuild All metabolites_smile.csv (default)")
    resolve.add_argument("--allow-low-coverage", action="store_true")

    report = sub.add_parser("report", parents=[common], help="coverage and quality summary")
    report.add_argument("--validate-rdkit", action="store_true")

    build = sub.add_parser("build-reference", parents=[common], help="rebuild Data/reference/ (maintainers)")
    build.add_argument("--aux_data_root", default=None)
    build.add_argument("--refresh-struct", action="store_true",
                       help="re-download MetaNetX bigg_struct.ttl.gz (~1 MB)")
    build.add_argument("--refresh-mnx", action="store_true",
                       help="re-stream MetaNetX chem_prop.tsv (~772 MB)")

    args = parser.parse_args()
    if args.command is None:
        args.command = "resolve"
        args.allow_low_coverage = False

    return {
        "resolve": cmd_resolve,
        "report": cmd_report,
        "build-reference": cmd_build_reference,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

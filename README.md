# Persistent Chiral Hodge Response

Research software and generated molecular geometry for a degree-two, source-driven persistent Hodge descriptor of molecular handedness.

Author: Vivaan Patni (GitHub: vxxqv).

## Scope

The software computes a reflection-odd triangle cochain, the unweighted persistent Laplacian with the inherited Euclidean inner product, and orientation-covariant signed readouts. Biological molecules are represented by generated conformer ensembles; these are computational geometries, not experimental measurements. Mirror controls test transformation laws and do not measure binding affinity or biological activity.

Manuscript sources, manuscript PDFs, plans, and submission materials are deliberately outside this repository and its archival package.

## Reproduction

Use Python 3.12. Install with `python -m pip install -e .[test]`, then run `python -m pytest` and `python -m pchr.experiment --output results/main`. Frozen molecular coordinates are in `data/conformers.json`; regeneration uses `python -m pchr.generate_data`. Figure generation uses `python -m pchr.figures --results results/main --output figures`.

## Mathematical conventions

All chain spaces are real, ordered simplices have the increasing vertex orientation, simplex bases are orthonormal, and degree-two complexes are constructed through dimension three. Rips radius is the maximum edge distance in angstroms. Source bandwidth is distinct from filtration radius. Persistent restricted-boundary null spaces use orthonormal columns. All test set sizes, tolerances, and numerical ranks are reported with their output records.

## Status

This repository is under active development. Results are authoritative only when accompanied by a completed run manifest and matching input hashes. Theorems are established by their written arguments, not by numerical tests.

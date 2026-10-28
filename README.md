# Persistent Chiral Hodge Response

Research software, selected source data and generated molecular geometries for persistent chiral Hodge responses and reflection-odd conformer-measure geometry. Version 0.2.0.

Author: Vivaan Patni (GitHub: vxxqv).

## Scope

The software computes a reflection-odd triangle cochain, the real persistent Laplacian with its inherited inner product, signed readouts, proper-rigid shape derivatives, odd Gaussian kernel means and covariance-resolvent scores. The extension includes 512 external parents / 5,514 source conformers, and a separately curated biological cohort of 143 same-assay binding comparisons / 103 parents / 65 human targets, with 1,648 generated conformers. Generated geometries are not experimental structures or equilibrium trajectories. Measured ChEMBL binding records are archived separately with their assay and target identities.

Manuscript sources, manuscript PDFs, plans, and submission materials are deliberately outside this repository and its archival package.

## Reproduction

Use Python 3.12. Install with `python -m pip install -e .[test,figures,external]`, then run `python -m pytest`. Exact executed package versions are in `environment_versions.json` and `requirements-lock.txt`. Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before multiprocessing. The executed environment is Windows; figure text uses Arial, and 3D rendering additionally requires working VTK/OpenGL.

- Original pilot: `python -m pchr.experiment --output replay/pilot`; coordinates are in `data/conformers.json`. `python -m pchr.generate_data` regenerates the source panel. `python -m pchr.geometry_experiment` reproduces its shape derivatives.
- Full external descriptor replay: `python scripts/replay_extension.py --output replay/external --recompute-descriptors --workers 4`. This recomputes every selected descriptor from archived coordinates, compares all component values and separation metrics, and reproduces the paired bootstrap. Omitting `--recompute-descriptors` checks scores from stored features only. The original full parquet and full embedding arrays are not needed.
- Parity diagnostic and calculus: copy `results/external_full` to a separate replay folder, then run `python -m pchr.ensemble_diagnostics replay/external_full`, `python -m pchr.ensemble_geometry replay/external_full` and `python -m pchr.kernel_shape_metric replay/external_full`.
- Biological curation: `python -m pchr.biology --raw data/extension_sources/chiral_cliffs --output replay/biology`. Compare its same-assay rows with `results/biology_cliffs`. The small 3DCS-linked acquisition uses the separate `data/extension_sources/3dcs` snapshot and `results/biology_linked`; it does not supply the main biological estimate.
- Biological generation and association: `python -m pchr.biology_ensembles replay/biology --workers 4`. This regenerates deterministic ETKDGv3/MMFF94s conformers when its `ensembles` cache is absent. Copy the archived `results/biology_cliffs/ensembles` there to reproduce associations from exactly the stored rounded geometries instead. Eight generated conformers per isomer have uniform descriptive weights; stored optimization energies belong to pre-rounding coordinates.

The redesigned figures use `python figures/rebuild.py --output figure_exports --cache figure_cache`, then `python figures/extension.py --output figure_exports --cache figure_cache`. They export PNG/PDF/SVG, captions and source data. Run `python figures/check_source.py figure_exports`, then `python figures/audit.py --figures figure_exports --qa figure_qa` for numerical and export checks. Source data for all 17 figure sets are also archived in `results/figure_source`. The exact font is `C:/Windows/Fonts/arial.ttf`; a different platform must record an explicit font substitution. The earlier `pchr.figures` module is retained for historical reproduction.

## Mathematical conventions

All chain spaces are real, ordered simplices have the increasing vertex orientation, simplex bases are orthonormal, and degree-two complexes are constructed through dimension three. Rips radius is the maximum edge distance in angstroms. Source bandwidth is distinct from filtration radius. Persistent restricted-boundary null spaces use orthonormal columns. All test set sizes, tolerances, and numerical ranks are reported with their output records.

## Status

The frozen protocol, outcome-blind structural selection, acquisition amendments and post-primary diagnostic status are retained. The primary external persistent-minus-source ES-AUC difference is -0.0425003, parent bootstrap 95% interval [-0.0523411,-0.0325988]. The primary biological Spearman correlation is 0.0498867 [-0.1536638,0.2462794]. These results establish no general benefit for propagation. No parameter or endpoint was replaced with a favourable result.

Theorems require their written proofs; tests are implementation checks. The kernel is characteristic on descriptor laws, not necessarily on molecular configurations. Generic finite-support Fisher divergence is a limitation, not a rigidity score. Rank bounds concern unconstrained labelled shapes after proper rigid motion. Biological correlations are exploratory, with parent resampling and a fixed observed target mix; they are not target-conditioned predictive validation, causal evidence or clinical guidance.

## Sources and licenses

Original software is MIT licensed. Third-party data retain their own licenses; the MIT license does not relicense them. See `data/LICENSES.md` and `data/extension_sources/provenance.json` for revisions, hashes, source indices and attribution. Selected 3DCS coordinates and released GemNet/MACE/FMG embeddings are CC BY-SA 4.0. ChEMBL raw and derived records are CC BY-SA 3.0. Original generated geometries, original numerical outputs and figure source data are CC BY 4.0 except where they derive from these third-party datasets, in which case the source share-alike license is retained. Manuscripts, bibliography files, plans, proof notes and submission files are excluded from the archival package.

# Reproduction guide

1. Download and extract the complete Zenodo ZIP linked in README. Verify it using
   `python verify_release.py`. Do not flatten its directories.
2. Use Python 3.12 and install the full lock from
   `research_output/revision/sci_revision/requirements_full.lock`.
3. Run `python research_output/revision/retrieve_benchmarks.py` for exact C/E source
   acquisition. Inspect `research_output/revision/protocol.json` and the source
   provenance JSON files. PH2 must be supplied privately using `prepare_ph2.py` and
   the documented source structure. `fetch_ph2.py` records the original legacy
   source; mirror scripts are provenance aids and do not grant redistribution rights.
4. Original main pipeline commands are documented in
   `research_output/revision/README_Reproducibility_Original.md`.
   In order: `train_revision.py`, `evaluate_revision.py`, `corruption_revision.py`,
   `development_sensitivity.py`, `summarize_revision.py`, `summarize_corruptions.py`.
   Read each script's arguments before starting a costly run. Fitted checkpoints
   are reused where the original scripts support resumption. For fresh fitting,
   use a separate copy without the corresponding output/checkpoint directories.
5. `transfer_reference.py` obtains the official frozen ImageNet backbone, fits
   classifier heads and evaluates them; its source weight URL and hash prefix
   check are retained. No external backbone is bundled.
6. Extension: run `sci_revision/optimized_cnn.py`, `evaluate_optimized_cnn.py`,
   `descriptor_noise.py` and `make_cnn_figure.py` (all under revision/).
   The JSON protocol fixes ten grouped seeds and two learning-rate candidates.
7. Theory check: `python research_output/revision/verify_finite_theory.py`.
   Clean-install audit evidence is in `sci_revision/analysis` and
   `sci_revision/replication_logs`; the isolation driver needs privately acquired
   source data before running. It replays predictions/bootstraps and three
   representative refits; it is not a full independent retraining study.

Document production uses the archived authoring scripts and installed Microsoft
Word/MathML-to-OMML resources on the authoring host. The scientific Python pipeline
does not require Word. The supplied manuscript files are the visually checked
publication artifacts; older authoring scripts may reproduce an earlier layout.

The public archive's nested `research_output/revision/manifest.sha256` preserves
the original isolation script's expected manifest format for the included tree.
The root `release_manifest.json` is authoritative for this public curation.
Some inherited authoring/audit programs refer to historical local file names;
the training/evaluation recipes and fixed source paths retain their original form.

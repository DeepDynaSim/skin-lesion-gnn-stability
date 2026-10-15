# Final Supplementary Reproducibility Archive S1

Final manuscript: Stable graph propagation does not ensure image robustness: finite-step analysis and component evaluation in dermoscopic classification. Author: Caglar Uyulan.

The package retains the original 202 scientific fits and adds 20 compact-CNN candidates: 150 main/control fits, 42 graph sensitivity fits, ten frozen-ImageNet heads and twenty compact-CNN fits. Ten compact-CNN selections refer to candidate states; they are not additional training runs. All 222 fitted states, split indices, histories, calibration records, predictions and analysis outputs are included. Three representative clean-environment refits are verification runs, not additional paper models. Final documents, new review responses and current checks are in `research_output/revision/sci_revision`.

## Source data and installation

Use Python 3.12 with `research_output/revision/sci_revision/requirements_full.lock`. Install with `python -m pip install -r research_output/revision/sci_revision/requirements_full.lock`. The CPU PyTorch index is specified. No installed libraries or authoring caches are bundled. The primary package versions and fresh-environment module paths are recorded. Windows/Word authoring requirements are separate from the ordinary Python scientific code.

Obtain C/E from the versioned release https://doi.org/10.5281/zenodo.11101338 using `research_output/revision/retrieve_benchmarks.py`; it validates all four exact source hashes. Source provenance, metadata and identity audits are retained. For PH2 obtain the source separately under its research/education terms, https://www.fc.up.pt/addi/ph2%20database.html. Retrieval and processing scripts document the legacy-link failure and mirror provenance. No PH2 raw image, original annotation table, mirror ZIP, resized image dataset, derived diagnosis CSV or clinical-image embedding cache is redistributed. Any locally obtained restricted PH2 files must remain outside public deposits.

## Replaying the scientific results

The original `README_Reproducibility_Original.md` documents the 150 main fits, 42 graph sensitivity fits, ten frozen heads and all original analyses. The original finalized documents are retained as audit context. The new extension uses:

1. `python research_output/revision/sci_revision/optimized_cnn.py` — twenty candidate fits over ten existing grouped partitions; completed fits are reused. Calibration is excluded from learning-rate selection.
2. `python research_output/revision/sci_revision/evaluate_optimized_cnn.py` — three-cohort inference, all 23 corruption ensembles and conditional bootstrap intervals.
3. `python research_output/revision/sci_revision/descriptor_noise.py` — full-cohort descriptor decomposition and ideal Gaussian check.
4. `python research_output/revision/sci_revision/make_cnn_figure.py` — added fitting/corruption illustration.
5. `python research_output/revision/sci_revision/prepare_manuscript.py` then `build_manuscript.py` in the same directory — final Word authoring with native OMML. The builder uses Microsoft's installed MathML-to-OMML XSL; native Word PDF export and the canonical authoring-host rasterizer are disclosed.

For fresh fitting, use a separate output tree with the candidate/model output directories absent, keeping code/protocols and acquired data. All actual recipes and seeds are locked in their JSON protocols. The limited CNN search did not improve every test metric; it is not an exhaustive optimum. The historical C test and all final comparisons are exploratory.

## Clean-installation evidence and rerun

The supplied `analysis/clean_replication.json`, `analysis/clean_new_cnn.json` and `replication_logs` are in `sci_revision`. A clean venv and isolated extracted tree were used on the same host. They reprocessed original PH2 inputs, regenerated bootstrap calculations, checked all saved-state predictions and refitted three representative recipes. This is not an independent-investigator study or complete retraining of all 222 fits.

To rerun the isolation workflow, create `sci_revision/replication_env` with `python -m venv`, install the full lock there, acquire the C/E and restricted PH2 inputs privately, and run `python research_output/revision/sci_revision/prepare_replication.py --archive <absolute path to this ZIP>`, followed by `replication_driver.py` and `launch_clean_new_cnn.py` from that directory. The driver uses the clean interpreter and retains complete logs. Dataset availability/licensing and the source paths must be satisfied first. No secrets or credentials are required.

`research_output/revision/manifest.sha256` hashes every included file except itself. ZIP CRC and every manifest hash are verified before delivery. Public persistent deposition remains pending; no public repository URL or archive DOI is fabricated. The archive can be inspected locally by the author/reviewer without redistributing PH2 source data.

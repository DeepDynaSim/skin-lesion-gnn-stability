# Skin lesion GNN stability and image robustness

**Stable graph propagation does not ensure image robustness: finite-step analysis and component evaluation in dermoscopic classification**

Caglar Uyulan — Faculty of Engineering and Architecture, Department of Mechanical
Engineering, İzmir Katip Celebi University, İzmir, Türkiye.
ORCID: https://orcid.org/0000-0002-6423-6720 · caglar.uyulan@ikcu.edu.tr

Version 1.0.0 · 15 October 2026

- Permanent research archive: https://doi.org/10.5281/zenodo.23393604
- Code repository: https://github.com/DeepDynaSim/skin-lesion-gnn-stability
- Full reproducibility download: https://zenodo.org/records/23393604/files/Skin_Lesion_GNN_Reproducibility_v1.0.0.zip?download=1

## What this release establishes

The study derives finite-step perturbation bounds for normalized graph
propagation and separates graph-only, node-only and joint image-corruption
effects. Stable propagation alone does not imply robustness to changes in
image-derived node features. The contrast penalty is evaluated as an ablation;
it is not claimed as a universal improvement. No clinical diagnostic validation
or state-of-the-art superiority is claimed. The historical C test was previously
inspected; the final comparisons are exploratory.

The complete archive contains **222 scientific fitted states**: 150 primary/control
fits, 42 graph sensitivity fits, ten frozen-ImageNet classifier heads, and twenty
compact-CNN candidates. Ten selected CNN entries point to existing candidates.
The three representative clean-environment refits are verification runs on the
same host, not 222 independently repeated fits. All primary/control and extension
protocols, group indices, histories, calibration records, predictions and analyses
are archived. PH2 is an external evaluation cohort and was not used in fitting,
hyperparameter choice or calibration.

The two source manuscript templates now have readable vector formulas on GitHub:
[original revision](research_output/revision/manuscript_template.md) and
[integrated revision](research_output/revision/sci_revision/manuscript_template.md).
They remain working templates: author, result, table and figure markers are filled
by the archived authoring pipeline. See the [formula/source guide](docs/MATH_RENDERING.md).

Representative reported results: C macro-F1 0.586 for ContrastPenalty, 0.589 for
the penalty-free control, 0.649 for ResNet18 and 0.585 for the compact CNN extension.
For Gaussian noise (sigma 0.05), ContrastPenalty macro-F1 is 0.589 for graph-only,
0.150 for node-only and 0.156 for joint perturbation (clean 0.586). The local
difference descriptors account for 90.7% of standardized perturbation energy;
this is not a percentage of classification loss. PH2 AUC is 0.507 for this graph
model and 0.806 for ResNet18. Negative results and limitations are retained.

## Start here

This Git repository keeps the source code and locked protocols small. Download
the complete DOI archive for fitted states, numerical outputs, full-resolution
figures and the editable manuscript. Extract it into a **separate directory**;
its `research_output/` layout is the original code's relative-path layout.

Run the following commands **from the root of the extracted complete archive**,
where `verify_release.py` is supplied. The small Git checkout does not contain that
verifier, the fitted states, or the finished manuscript/figure directories.

```powershell
cd /path/to/extracted/Skin_Lesion_GNN_Reproducibility_v1.0.0
python -m venv .venv
# Activate the environment with the command appropriate to your operating system.
python -m pip install -r research_output/revision/sci_revision/requirements_full.lock
python research_output/revision/retrieve_benchmarks.py
python verify_release.py
```

Python 3.12 and the pinned CPU scientific dependencies were used. Dataset C/E
retrieval validates all four exact source hashes. PH2 must be obtained separately
under its source terms; follow `REPRODUCE.md`. No source clinical image dataset,
restricted PH2 annotation table or image embedding cache is redistributed.

## Git checkout and complete archive

| Resource | Small Git checkout | Complete DOI archive |
|---|---|---|
| Pipeline source, protocols and dependency lock | Included | Included |
| Working manuscript templates | Included, with readable formulas and editable sources | Original authoring templates |
| 222 fitted scientific states and computed results | Download separately | Included |
| Final `manuscript/` and `figures/` directories | Download separately | Included |
| `verify_release.py`, `release_manifest.json`, `manifest.sha256` | Download separately | Included |
| `release_inventory.json`, release and source notices | Included | Archive curation is recorded separately |

The locations below describe the **complete extracted archive**, rather than
promising that its generated outputs are all present in the small Git repository:

- `research_output/revision`: original scientific pipeline, protocols and results.
- `research_output/revision/sci_revision`: compact-CNN extension, descriptor-noise
  analysis, fixed protocols and same-host clean-environment audit evidence.
- `manuscript/`: current integrated Word/PDF manuscript and S1 Text. Word formulas
  are native editable Office Math. These are manuscript artifacts, not evidence
  of acceptance or publication in a journal.
- `figures/`: 14 final figures in TIFF (400 ppi), PNG, SVG and PDF.
- `release_manifest.json`, `manifest.sha256`: file sizes, hashes and model-state
  counts. `verify_release.py` validates bytes without third-party packages.
- `curation_exclusions.json`: explains differences from the earlier local archive.

Documentation presentation fixes in Git do not change the frozen DOI archive or
its checksums. Current Git authoring scripts read the byte-preserved
`manuscript_template.md.in` sources so that native editable mathematics is retained;
the adjacent `.md` files are generated reading views.

Some historical audit files preserve the checks performed before public deposition.
Their statements that public deposition was pending describe that earlier stage;
the DOI above and `RELEASE_NOTES.md` identify this public release. Scientific
result files and fitted states have not been changed by public curation.

## Licensing and attribution

Original code: MIT (`LICENSE`). Original text and numerical plots: CC BY 4.0.
Source DermaMNIST metadata and dermoscopic example panels: CC BY-NC 4.0.
See `THIRD_PARTY_NOTICES.md` for precise exceptions and required source attribution.
The archive-level Zenodo license reflects the noncommercial third-party material;
it does not replace the MIT license on the original code.

No external funding. No competing interests declared. OpenAI tools assisted code,
drafting and document production; scientific and authorship responsibility remains
with the named author. This release does not submit the manuscript to a journal.

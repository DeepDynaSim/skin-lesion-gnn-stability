# Licenses and third-party attribution

The MIT license in `LICENSE` applies to original software by Caglar Uyulan.
Original manuscript text and original numerical graphs/tables are made available
under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), except for the
third-party material identified below. The combined archive includes material
under different licenses; a metadata-level license never overrides these notices.

## Corrected and extended DermaMNIST / HAM10000

The corrected/extended source release, **Investigating the Quality of DermaMNIST
and Fitzpatrick17k Dermatological Image Datasets**, is licensed CC BY-NC 4.0:
https://doi.org/10.5281/zenodo.11101338 . Source metadata were checked against the
official Zenodo record API on 15 October 2026. Source CSV metadata, identifiers,
and dermoscopic example panels retained in this archive remain subject to that
license (https://creativecommons.org/licenses/by-nc/4.0/).

HAM10000 attribution: (c) ViDIR Group, Department of Dermatology, Medical
University of Vienna; Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset,
a large collection of multi-source dermatoscopic images of common pigmented
skin lesions. Scientific Data 5, 180161 (2018).
https://doi.org/10.1038/sdata.2018.161 .
Corrected/extended release authors and source provenance are provided in its
Zenodo record and the archived benchmark provenance files. No claim of original
ownership is made over these data. Dataset images are not bundled as a reusable
training set. Illustrated crops/graphs/corruptions are adaptations for method
explanation, not newly acquired patients. The external-image exceptions apply
to `figures/S1Fig.*`, `figures/S2Fig.*`, the same panels embedded in the manuscript
and S1 Text, and historical example/graph figure files in `research_output`.

## PH2 and pretrained weights

PH2 images, original diagnosis/annotation tables, mirrors, resized PH2 datasets
and image embedding caches are excluded. Obtain PH2 independently under its
research/education conditions: https://www.fc.up.pt/addi/ph2%20database.html .
Aggregate performance outputs and non-image prediction/calibration records are
included. These files must not be interpreted as permission to redistribute PH2.

Official ImageNet pretrained MobileNetV3 weights are excluded; the original
`transfer_reference.py` downloads the official torchvision weight URL and checks
the published hash prefix. The 222 author-fitted study states are included;
their scientific use remains subject to applicable source-dataset terms.
Python, PyTorch/torchvision, NumPy, scikit-learn, Matplotlib and other dependencies
are obtained separately and retain their upstream licenses. No dependency
installation or external paper PDF is relicensed by this deposit.

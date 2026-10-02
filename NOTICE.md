# Third-party notices

The MIT licence in [LICENSE](LICENSE) covers the **source code** of medfm-lab only.

medfm-lab downloads model weights and datasets at runtime rather than bundling them, so no
third-party artefact is redistributed here — but the terms below govern what you may do with
what it fetches. Several are research- or non-commercial-only, and that restriction is not
lifted by this project's MIT licence.

## Model weights

| model | repo | licence | notes |
|---|---|---|---|
| RAD-DINO | `microsoft/rad-dino` | Microsoft Research License (MSRLA) | research use |
| RAD-DINO-MAIRA-2 | `microsoft/rad-dino-maira-2` | Microsoft Research License (MSRLA) | research use |
| CXformer-S / CXformer-B | `m42-health/CXformer-base` | CC-BY-NC-4.0 | **non-commercial**, research only |
| MedDINOv3 ViT-B/16 (CT-3M) | `ricklisz123/MedDINOv3-ViTB-16-CT-3M` | Apache-2.0 | |
| DINOv2 | `facebook/dinov2-*` | Meta DINOv2 licence | |
| DINOv3 | `facebook/dinov3-*` | Meta DINOv3 licence | gated; requires accepting terms |
| DINOv3 (timm mirrors) | `timm/vit_*_dinov3.*` | inherits Meta DINOv3 terms | used as the ungated fallback |

RAD-DINO and RAD-DINO-MAIRA-2 are derivatives of DINOv2, so Meta's terms may also apply.
CXformer additionally ships custom modelling code that the loader executes with
`trust_remote_code=True`; read it before running it on anything you care about.

## Datasets

| dataset | source | licence |
|---|---|---|
| MedMNIST+ v2 (224px) | [medmnist.com](https://medmnist.com) / Zenodo | CC BY 4.0 |

MedMNIST+ is itself derived from 16+ upstream datasets, each with its own terms. If you
publish work using a subset, cite and check the corresponding source dataset — the MedMNIST
documentation lists them and the required citations.

## No clinical use

Nothing in this project is validated for clinical decision-making, diagnosis or treatment.
The models are research artefacts, the bundled evaluation data is small and preprocessed,
and the interpretability views are diagnostic aids for researchers, not evidence of safety
or correctness.

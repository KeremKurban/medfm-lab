"""What each dataset and each backbone actually is, in clinical terms.

The class names come straight from `medmnist.INFO` (authoritative, offline). The clinical
notes — what a label means and what tends to distinguish it visually — are background from
each source dataset's own literature, not anything this tool inferred. They exist so that a
patch map is not just coloured squares.

Source-dataset and licence strings also come from `medmnist.INFO`.
"""

from __future__ import annotations

from typing import Optional


def _info(flag: str, key: str, default: str = "") -> str:
    try:
        from medmnist import INFO

        return str(INFO[flag].get(key, default))
    except Exception:
        return default


# flag -> guiding text. `classes` maps the integer label to "meaning — what you look for".
DATASET_GUIDE: dict[str, dict] = {
    "retinamnist": {
        "what": "Colour fundus photographs of the retina, graded by **severity of diabetic "
                "retinopathy** on the standard 0-4 scale. This is the hardest dataset in the "
                "collection and the one most often misread, because the label is just a "
                "number: the classes are *not* different diseases, they are one disease at "
                "five stages, and neighbouring stages look almost identical.",
        "classes": {
            0: "**No diabetic retinopathy** — clean fundus, no visible lesions.",
            1: "**Mild non-proliferative DR** — microaneurysms only: tiny red dots, often the "
               "first visible sign. Nothing else.",
            2: "**Moderate NPDR** — more than microaneurysms: dot and blot haemorrhages, hard "
               "exudates (yellow lipid deposits), venous beading, but short of the severe "
               "criteria.",
            3: "**Severe NPDR** — the '4-2-1' rule: haemorrhages in all four quadrants, "
               "venous beading in two or more, or intraretinal microvascular abnormalities in "
               "one or more. One step from proliferative disease.",
            4: "**Proliferative DR** — neovascularisation, i.e. new fragile vessels, with "
               "possible vitreous or preretinal haemorrhage.",
        },
        "edge": "Adjacent grades share most of their visual features, which is why near-miss "
                "errors dominate here. If the model confuses 2 with 3, that is a much smaller "
                "error than confusing 0 with 4 — plain accuracy will not tell you that, which "
                "is what quadratic weighted kappa is for.",
        "caveats": [
            "Only 1,600 source images — the smallest set here, so the probe curve is noisy.",
            "The probe reports plain accuracy, which understates an ordinal model. Read the "
            "grade ordering in mind, not just the headline number.",
            "Fundus photographs: the optic disc is the bright circle, vessels radiate from "
            "it, the macula is the darker area beside it.",
        ],
    },
    "pathmnist": {
        "what": "H&E-stained **colorectal cancer histology** patches from NCT-CRC-HE-100K. "
                "Nine tissue classes on a tile-by-tile basis — a classic tissue-type "
                "classification problem.",
        "classes": {
            0: "adipose — fat cells, large empty-looking vacuoles",
            1: "background — empty slide, no tissue",
            2: "debris — necrotic cell remnants",
            3: "lymphocytes — small round dark nuclei, dense immune infiltrate",
            4: "mucus — pale amorphous mucin pools",
            5: "smooth muscle — elongated nuclei in parallel bundles",
            6: "normal colon mucosa — regular glandular epithelium",
            7: "cancer-associated stroma — desmoplastic connective tissue around tumour",
            8: "colorectal adenocarcinoma epithelium — tumour glands, irregular and crowded",
        },
        "edge": "Discrimination rests on gland architecture, nuclear crowding and atypia, and "
                "the balance of epithelium against stroma. The two hardest pairs are normal "
                "mucosa versus adenocarcinoma epithelium, and stroma versus smooth muscle.",
        "caveats": ["Patches are 224px tiles, not whole slides. Whole-slide images are where "
                    "these backbones are known to degrade most."],
    },
    "bloodmnist": {
        "what": "Individual **peripheral blood cells** from healthy donors, captured by "
                "microscopy. Eight cell types — a morphology problem solved on shape.",
        "classes": {
            0: "basophil — rare, coarse dark granules that obscure the nucleus",
            1: "eosinophil — bright orange-red granules, usually two-lobed nucleus",
            2: "erythroblast — nucleated red cell precursor, very dark round nucleus",
            3: "immature granulocytes — myelocytes/metamyelocytes/promyelocytes, "
               "intermediate maturity, less lobated than mature forms",
            4: "lymphocyte — large round nucleus, thin rim of cytoplasm",
            5: "monocyte — kidney-bean nucleus, generous grey cytoplasm",
            6: "neutrophil — multi-lobed nucleus, fine granules",
            7: "platelet — tiny, no nucleus",
        },
        "edge": "What separates them is nuclear shape and lobation, granule colour and "
                "texture, cell size, and the nuclear-to-cytoplasmic ratio. Immature "
                "granulocytes versus neutrophils is the classic confusion.",
        "caveats": ["All cells come from *healthy* donors, so this is morphology, not "
                    "disease detection."],
    },
    "octmnist": {
        "what": "**Optical coherence tomography** cross-sections of the retina — the layered "
                "B-scan that shows retinal structure in depth. Four diagnostic categories.",
        "classes": {
            0: "choroidal neovascularization — abnormal vessels with subretinal fluid and "
               "disrupted outer retinal layers",
            1: "diabetic macular edema — intraretinal cysts and retinal thickening",
            2: "drusen — small dome-shaped deposits under the retinal pigment epithelium",
            3: "normal — intact, parallel retinal layers and a clean foveal dip",
        },
        "edge": "You are reading the layer structure: whether fluid has broken the layers "
                "apart, whether cysts are present, whether the RPE is elevated. Normal is "
                "defined by the layers staying put.",
        "caveats": ["Source images are greyscale; as_rgb duplicates the channel, which does "
                    "not add information."],
    },
    "dermamnist": {
        "what": "Dermatoscopic images of **pigmented skin lesions** from HAM10000. Seven "
                "diagnoses, heavily imbalanced — some classes have a few dozen examples.",
        "classes": {
            0: "actinic keratoses / intraepithelial carcinoma — scaly, erythematous, sun-damaged",
            1: "basal cell carcinoma — pearly, telangiectatic, often ulcerated",
            2: "benign keratosis-like lesions — seborrhoeic keratoses, waxy stuck-on appearance",
            3: "dermatofibroma — firm, dimpled, central scar-like area",
            4: "melanoma — asymmetric, irregular border, multiple colours",
            5: "melanocytic nevi — common moles, symmetric, uniform pigment",
            6: "vascular lesions — red to purple, vascular structures visible",
        },
        "edge": "The classic mnemonic is ABCD — asymmetry, border irregularity, colour "
                "variation, diameter — plus pigment-network and vascular patterns.",
        "caveats": ["**Licence is CC BY-NC 4.0: non-commercial use only**, unlike the other "
                    "sets here which are CC BY 4.0.",
                    "Severe class imbalance; a probe can score well by predicting the "
                    "majority class."],
    },
    "breastmnist": {
        "what": "**Breast ultrasound** images, simplified to a binary task: malignant versus "
                "normal-or-benign.",
        "classes": {
            0: "malignant",
            1: "normal, benign (the two source classes merged)",
        },
        "edge": "Malignancy signs on ultrasound are an irregular or spiculated shape, "
                "non-parallel orientation, posterior acoustic shadowing, and marked "
                "hypoechogenicity.",
        "caveats": ["Only 780 source images — expect noisy probe curves.",
                    "Benign and normal were merged, so a benign lesion is deliberately not "
                    "distinguishable from a healthy one."],
    },
    "pneumoniamnist": {
        "what": "**Paediatric chest X-rays**, binary: pneumonia versus normal. The simplest "
                "task here and the natural pairing with RAD-DINO.",
        "classes": {
            0: "normal — clear lung fields",
            1: "pneumonia — consolidation, air bronchograms, obscured heart or diaphragm "
               "borders",
        },
        "edge": "Look for opacification of lung that should be dark, and loss of the normal "
                "silhouettes.",
        "caveats": ["Paediatric images — adult CXR features transfer imperfectly.",
                    "Greyscale source; as_rgb duplicates the channel."],
    },
    "chestmnist": {
        "what": "**Chest X-rays (NIH ChestX-ray14)** with 14 text-mined findings, as "
                "*multi-label* — several can be positive at once. The probe collapses it to "
                "one finding so the task stays single-label.",
        "classes": {
            0: "atelectasis — collapsed lung regions",
            1: "cardiomegaly — enlarged heart silhouette",
            2: "effusion — fluid at the lung base",
            3: "infiltration — ill-defined opacity",
            4: "mass — larger focal opacity, >3 cm",
            5: "nodule — small focal opacity, <3 cm",
            6: "pneumonia — infection with consolidation",
            7: "pneumothorax — air in the pleural space, absent lung markings",
            8: "consolidation — solid-appearing lung, air bronchograms",
            9: "edema — pulmonary fluid, often with vascular congestion",
            10: "emphysema — hyperinflated, flattened diaphragms",
            11: "fibrosis — chronic scarring, volume loss",
            12: "pleural — pleural thickening",
            13: "hernia — diaphragmatic hernia",
        },
        "edge": "Most findings reduce to where opacity appears and what silhouette is lost. "
                "Labels were mined from reports, so they are noisy by construction.",
        "caveats": ["Report-mined labels carry known errors.",
                    "Mass versus nodule is a size threshold, and infiltration versus "
                    "consolidation is genuinely ambiguous."],
    },
    "organamnist": {
        "what": "Axial **abdominal CT** slices (LiTS), classifying which of 11 organs the "
                "central cropped region shows.",
        "classes": {
            0: "bladder", 1: "femur-left", 2: "femur-right", 3: "heart", 4: "kidney-left",
            5: "kidney-right", 6: "liver", 7: "lung-left", 8: "lung-right", 9: "pancreas",
            10: "spleen",
        },
        "edge": "Organs separate by Hounsfield density, shape, and position within the body. "
                "Left/right pairs differ essentially only in mirrored position, which is a "
                "hard but interesting test of whether features encode geometry.",
        "caveats": ["Hounsfield units are windowed to an abdominal range before saving, so "
                    "absolute density is lost.",
                    "Left/right pairs are near-duplicates apart from position."],
    },
    "organcmnist": {
        "what": "Coronal **abdominal CT** slices (LiTS), same 11 organs as organamnist.",
        "classes": {},
        "edge": "Same as the axial version, viewed coronal — the same anatomy in a different "
                "plane, so comparisons between the three organ sets isolate plane, not "
                "pathology.",
        "caveats": ["See organamnist."],
    },
    "organsmnist": {
        "what": "Sagittal **abdominal CT** slices (LiTS), same 11 organs as organamnist.",
        "classes": {},
        "edge": "Same as the axial version, viewed sagittal.",
        "caveats": ["See organamnist."],
    },
    "tissuemnist": {
        "what": "**Human kidney cortex cells** from BBBC051, segmented and sorted into eight "
                "cell types.",
        "classes": {
            0: "Collecting Duct, Connecting Tubule",
            1: "Distal Convoluted Tubule",
            2: "Glomerular endothelial cells",
            3: "Interstitial endothelial cells",
            4: "Leukocytes",
            5: "Podocytes",
            6: "Proximal Tubule Segments",
            7: "Thick Ascending Limb",
        },
        "edge": "Classes are nephron segments and cell types told apart by shape and position "
                "within the cortex. Note the source data is *multiplexed fluorescence*, "
                "collapsed here to greyscale by taking the maximum across seven channels — so "
                "you are looking at intensity patterns, not a stained bright-field image.",
        "caveats": ["This is the only set here that is not a photograph of tissue; it is "
                    "fluorescence-derived, so the usual histological intuition does not "
                    "transfer."],
    },
}


MODEL_GUIDE: dict[str, dict] = {
    "dinov2-base": {
        "what": "The original DINOv2 ViT-B/14, trained on 142M natural images. Used here as "
                "the **natural-image baseline**: whatever a medical backbone does, compare it "
                "against this to see what the medical training actually bought.",
        "best_for": "Baseline for any modality; no register tokens, so you can see what "
                    "registers fix.",
    },
    "dinov2-base-reg": {
        "what": "Identical to DINOv2-base but with four register tokens, added in the "
                "'Vision Transformers Need Registers' paper specifically to stop a few patch "
                "tokens absorbing global context.",
        "best_for": "Demonstrating the outlier-token problem — compare its norm and attention "
                    "profiles against the plain version.",
    },
    "dinov3-vits16": {
        "what": "Smallest DINOv3, distilled from the 7B teacher. Fast enough to iterate on.",
        "best_for": "Quick experiments and the patch-token tab, where speed matters.",
    },
    "dinov3-vits16plus": {
        "what": "Slightly wider DINOv3 small variant. Also available through an ungated timm "
                "mirror, so it works without a Hugging Face licence click.",
        "best_for": "A DINOv3 backbone with no gating friction.",
    },
    "dinov3-vitb16": {
        "what": "DINOv3 ViT-B/16 — the natural workhorse, and the architecture MedDINOv3 is "
                "built on.",
        "best_for": "The right base for phase-2 domain adaptation; big enough to be useful, "
                    "small enough to fine-tune with adapters on one Mac.",
    },
    "dinov3-vitl16": {
        "what": "DINOv3 ViT-L/16, 300M parameters. The largest that is comfortable on 32 GB.",
        "best_for": "Checking whether scale helps on your modality — DINOv3 does *not* follow "
                    "a clean scaling law in the medical domain, so this is worth measuring "
                    "rather than assuming.",
    },
    "dinov3-convnext-base": {
        "what": "A convolutional DINOv3 rather than a transformer. No attention maps at all.",
        "best_for": "A control: if a ConvNet matches the ViT on frozen features, the "
                    "transformer's attention is not where the value lies.",
    },
    "rad-dino": {
        "what": "Microsoft's DINOv2 ViT-B/14, continually trained on roughly 800k chest "
                "X-rays from five public datasets. Still a natural-image-shaped architecture, "
                "so its attention and patch maps look and behave conventionally.",
        "best_for": "Chest X-ray work — `pneumoniamnist` and `chestmnist`. The best-behaved "
                    "medical backbone here.",
    },
    "rad-dino-maira2": {
        "what": "The same model trained on more data, as used in MAIRA-2 for grounded "
                "radiology report generation.",
        "best_for": "Slightly stronger CXR features than RAD-DINO, but under Microsoft's "
                    "licence.",
    },
    "meddinov3-vitb16": {
        "what": "DINOv3 ViT-B/16 domain-adapted on CT-3M, 3.87M axial CT slices from 16 "
                "datasets — essentially the phase-2 experiment already done at scale. "
                "Released as a raw DINO-format checkpoint, not a transformers model.",
        "best_for": "CT work — `organamnist`, `organcmnist`, `organsmnist`. On organ "
                    "classification it holds its accuracy into the late blocks where "
                    "RAD-DINO decays, which is exactly the adaptation effect to look for.",
    },
    "cxformer-base": {
        "what": "M42's DINOv2-small-based chest X-ray model, trained on 600k+ CXRs with "
                "register tokens and teacher centering. Ships custom modelling code.",
        "best_for": "A second opinion on CXR. **CC BY-NC: non-commercial only.**",
        "caveats": ["Its attention entropy reads as near-uniform at every layer. Either its "
                    "custom attention implementation differs from stock DINOv2 or it wants "
                    "its own preprocessing — treat its attention tab as unverified."],
    },
}


def dataset_guide_markdown(flag: str) -> str:
    """Render the guide for one dataset, or an empty string if there is no entry."""
    g = DATASET_GUIDE.get(flag)
    if not g:
        return ""
    from . import data as mdata

    meta = mdata.dataset_meta(flag)
    lic = _info(flag, "license", "unknown")
    warn = ""
    if "NC" in lic.upper():
        warn = (f"\n\n> ⚠ **Licence: {lic} — non-commercial use only.** This is more "
                f"restrictive than the other sets here.")

    lines = [
        f"#### {flag} · {meta['description']}",
        f"*{meta['modality']} · {meta['task']} · {meta['n_classes']} classes · "
        f"{meta['n_samples']['train']:,} train / {meta['n_samples']['test']:,} test · "
        f"licence {lic}*",
        "",
        g["what"],
        "",
    ]
    if g.get("classes"):
        lines.append("**What the classes are:**")
        lines.append("")
        for k, v in g["classes"].items():
            lines.append(f"- `{k}` — {v}")
        lines.append("")
    if g.get("edge"):
        lines += [f"**What tends to distinguish them:** {g['edge']}", ""]
    if g.get("caveats"):
        lines.append("**Caveats:**")
        lines.append("")
        for c in g["caveats"]:
            lines.append(f"- {c}")
    lines.append(warn)
    return "\n".join(lines)


def model_guide_markdown(key: str) -> str:
    g = MODEL_GUIDE.get(key)
    if not g:
        return ""
    out = [f"**{g['what']}**"]
    if g.get("best_for"):
        out.append(f"\n*Best for:* {g['best_for']}")
    for c in g.get("caveats", []):
        out.append(f"\n> ⚠ {c}")
    return "\n".join(out)


def glossary_markdown() -> str:
    from .glossary import GLOSSARY

    lines = [
        "# Glossary",
        "",
        "Every term the app shows, in plain language. Hover any metric name on the Inference "
        "or Residual tabs for a one-line reminder.",
        "",
    ]
    for term, (short, body) in GLOSSARY.items():
        lines += [f"### {term}", f"*{short}*", "", body, ""]
    return "\n".join(lines)

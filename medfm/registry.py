"""Model catalogue.

Every entry describes how to load an encoder and how to preprocess for it, so the
rest of the package never needs model-specific branches.

`gated=True` means the Hugging Face repo requires accepting a license while logged
in. Those downloads fail with a 401/403 until the user accepts on the model page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# RAD-DINO was trained with MIMIC-CXR intensity statistics.
CXR_MEAN = (0.5056, 0.5056, 0.5056)
CXR_STD = (0.252, 0.252, 0.252)
# Symmetric mapping for single-channel CT/OCT rendered as RGB.
HALF_MEAN = (0.5, 0.5, 0.5)
HALF_STD = (0.5, 0.5, 0.5)


@dataclass
class ModelSpec:
    key: str
    label: str
    hf_id: Optional[str] = None
    timm_id: Optional[str] = None
    family: str = "dinov2"          # dinov2 | dinov3 | medical
    patch_size: int = 14
    embed_dim: int = 768
    num_blocks: int = 12
    image_size: int = 224           # default square input for this encoder
    mean: tuple = IMAGENET_MEAN
    std: tuple = IMAGENET_STD
    domain: str = "natural"
    modality: str = "any"
    params_m: float = 0.0
    gated: bool = False
    notes: str = ""
    tags: tuple = field(default_factory=tuple)
    # For models released as a raw DINO-format state dict rather than a hub repo.
    ckpt: Optional[dict] = None
    trust_remote_code: bool = False

    @property
    def grid_default(self) -> int:
        return self.image_size // self.patch_size


MODELS: dict[str, ModelSpec] = {
    # ---------------------------------------------------------------- DINOv2
    "dinov2-base": ModelSpec(
        key="dinov2-base",
        label="DINOv2 ViT-B/14 (Meta, natural)",
        hf_id="facebook/dinov2-base",
        family="dinov2",
        patch_size=14,
        embed_dim=768,
        num_blocks=12,
        image_size=224,
        domain="natural",
        params_m=86,
        notes="Ungated baseline. No register tokens.",
        tags=("baseline", "natural"),
    ),
    "dinov2-base-reg": ModelSpec(
        key="dinov2-base-reg",
        label="DINOv2 ViT-B/14 + registers (Meta, natural)",
        hf_id="facebook/dinov2-with-registers-base",
        family="dinov2",
        patch_size=14,
        embed_dim=768,
        num_blocks=12,
        image_size=224,
        domain="natural",
        params_m=86,
        notes="4 register tokens — useful for studying high-norm outlier tokens.",
        tags=("baseline", "natural", "registers"),
    ),
    # ---------------------------------------------------------------- DINOv3
    "dinov3-vits16": ModelSpec(
        key="dinov3-vits16",
        label="DINOv3 ViT-S/16 distilled (Meta, natural)",
        hf_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        timm_id="vit_small_patch16_dinov3.lvd1689m",
        family="dinov3",
        patch_size=16,
        embed_dim=384,
        num_blocks=12,
        image_size=256,
        domain="natural",
        params_m=21,
        gated=True,
        notes="Smallest DINOv3. Gated on HF — accept the license first.",
        tags=("dinov3", "small"),
    ),
    "dinov3-vits16plus": ModelSpec(
        key="dinov3-vits16plus",
        label="DINOv3 ViT-S+/16 distilled (Meta, natural)",
        hf_id="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        timm_id="vit_small_plus_patch16_dinov3.lvd1689m",
        family="dinov3",
        patch_size=16,
        embed_dim=384,
        num_blocks=12,
        image_size=256,
        domain="natural",
        params_m=29,
        gated=True,
        notes="Ungated mirror exists on timm under timm/vit_small_plus_patch16_dinov3.lvd1689m.",
        tags=("dinov3", "small", "timm-ungated"),
    ),
    "dinov3-vitb16": ModelSpec(
        key="dinov3-vitb16",
        label="DINOv3 ViT-B/16 distilled (Meta, natural)",
        hf_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        timm_id="vit_base_patch16_dinov3.lvd1689m",
        family="dinov3",
        patch_size=16,
        embed_dim=768,
        num_blocks=12,
        image_size=256,
        domain="natural",
        params_m=86,
        gated=True,
        notes="The natural workhorse for domain adaptation experiments.",
        tags=("dinov3", "base"),
    ),
    "dinov3-vitl16": ModelSpec(
        key="dinov3-vitl16",
        label="DINOv3 ViT-L/16 distilled (Meta, natural)",
        hf_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        timm_id="vit_large_patch16_dinov3.lvd1689m",
        family="dinov3",
        patch_size=16,
        embed_dim=1024,
        num_blocks=24,
        image_size=256,
        domain="natural",
        params_m=300,
        gated=True,
        notes="Comfortable on 32 GB at 256-384px in fp32; training needs adapters.",
        tags=("dinov3", "large"),
    ),
    "dinov3-convnext-base": ModelSpec(
        key="dinov3-convnext-base",
        label="DINOv3 ConvNeXt-Base (Meta, natural)",
        hf_id="facebook/dinov3-convnext-base-pretrain-lvd1689m",
        family="dinov3",
        patch_size=32,          # effective stride of the last stage
        embed_dim=1024,
        num_blocks=4,           # 4 stages
        image_size=256,
        domain="natural",
        params_m=89,
        gated=True,
        notes="Non-transformer control: no attention maps, but still has a feature hierarchy.",
        tags=("dinov3", "convnext", "control"),
    ),
    # ---------------------------------------------------------------- medical
    "rad-dino": ModelSpec(
        key="rad-dino",
        label="RAD-DINO ViT-B/14 (Microsoft, chest X-ray)",
        hf_id="microsoft/rad-dino",
        family="medical",
        patch_size=14,
        embed_dim=768,
        num_blocks=12,
        image_size=518,
        mean=CXR_MEAN,
        std=CXR_STD,
        domain="medical",
        modality="chest-x-ray",
        params_m=86,
        notes="DINOv2 ViT-B/14 continually trained on 5 public CXR sets (~800k images).",
        tags=("medical", "cxr", "dinov2"),
    ),
    "rad-dino-maira2": ModelSpec(
        key="rad-dino-maira2",
        label="RAD-DINO-MAIRA-2 (Microsoft, chest X-ray)",
        hf_id="microsoft/rad-dino-maira-2",
        family="medical",
        patch_size=14,
        embed_dim=768,
        num_blocks=12,
        image_size=518,
        mean=CXR_MEAN,
        std=CXR_STD,
        domain="medical",
        modality="chest-x-ray",
        params_m=86,
        notes="Trained on more data than RAD-DINO. MSRLA licence.",
        tags=("medical", "cxr", "dinov2"),
    ),
    "meddinov3-vitb16": ModelSpec(
        key="meddinov3-vitb16",
        label="MedDINOv3 ViT-B/16 (CT-3M, CT/MRI)",
        hf_id=None,
        timm_id="vit_base_patch16_dinov3",
        family="medical",
        patch_size=16,
        embed_dim=768,
        num_blocks=12,
        image_size=224,
        mean=HALF_MEAN,
        std=HALF_STD,
        domain="medical",
        modality="ct",
        params_m=86,
        notes="DINOv3 ViT-B domain-adapted on 3.87M axial CT slices (16 datasets). "
              "Apache-2.0. Released as a raw DINO reference-format model.pth; loaded "
              "into the timm ViT-B/16 DINOv3 architecture (RoPE + 4 storage tokens).",
        tags=("medical", "ct", "dinov3"),
        ckpt={
            "repo": "ricklisz123/MedDINOv3-ViTB-16-CT-3M",
            "file": "model.pth",
            "prefix": "backbone.",
            "arch": "vit_base_patch16_dinov3",
        },
    ),
    "cxformer-base": ModelSpec(
        key="cxformer-base",
        label="CXformer-B ViT-B (M42, chest X-ray)",
        hf_id="m42-health/CXformer-base",
        family="medical",
        patch_size=14,
        embed_dim=768,
        num_blocks=12,
        image_size=518,
        mean=CXR_MEAN,
        std=CXR_STD,
        domain="medical",
        modality="chest-x-ray",
        params_m=87,
        notes="DINOv2-small base + registers. CC-BY-NC, research only. "
              "Ships custom modelling code (trust_remote_code).",
        tags=("medical", "cxr", "dinov2", "remote-code"),
        trust_remote_code=True,
    ),
}


def get_spec(key: str) -> ModelSpec:
    key = resolve_key(key)
    return MODELS[key]


def list_specs(tag: Optional[str] = None, family: Optional[str] = None) -> list[ModelSpec]:
    out = list(MODELS.values())
    if tag:
        out = [s for s in out if tag in s.tags]
    if family:
        out = [s for s in out if s.family == family]
    return out


def available_choices() -> list[str]:
    """Display labels for the model dropdown, ungated + medical first.

    Returns PLAIN STRINGS, not (name, value) tuples, on purpose. In Gradio 6 the Python
    layer validates `value` against a choice's *value* while the frontend validates it
    against the choice's *name*, so tuple choices produce contradictory errors and the app
    silently fails to initialise. Plain strings make name == value, and `resolve_key`
    maps the label back to a key inside the callback.
    """
    def rank(s: ModelSpec):
        return (s.gated, s.family != "medical", s.params_m)

    return [s.label for s in sorted(MODELS.values(), key=rank)]


def label_for(key: str) -> str:
    return MODELS[resolve_key(key)].label


def resolve_key(value: str) -> str:
    """Accept a model key or its display label; return the key.

    Defensive by design: a dropdown may hand back either the label or the key depending on
    the widget layer, and both must work.
    """
    if value in MODELS:
        return value
    for k, s in MODELS.items():
        if s.label == value:
            return k
    for k, s in MODELS.items():
        if not value:
            continue
        if value == k or value.startswith(k + " ") or (len(value) > 3 and value[:12] == s.label[:12]):
            return k
    for k, s in MODELS.items():
        if value and k in value:
            return k
    raise KeyError(
        f"unrecognised backbone {value!r}. Available keys: {', '.join(sorted(MODELS))}"
    )


def resolve_timm_id(spec: ModelSpec) -> Optional[str]:
    """Resolve a timm model id, probing the installed timm for a near match."""
    if spec.timm_id:
        return spec.timm_id
    if spec.family != "dinov3":
        return None
    try:
        import timm
    except ImportError:
        return None
    cands = [m for m in timm.list_models("*dinov3*", pretrained=True)]
    if spec.embed_dim == 384 and spec.num_blocks == 12:
        pref = [m for m in cands if "small" in m]
    elif spec.embed_dim == 768:
        pref = [m for m in cands if "base" in m]
    else:
        pref = [m for m in cands if "large" in m]
    return pref[0] if pref else (cands[0] if cands else None)

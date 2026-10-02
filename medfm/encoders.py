"""Uniform encoder wrapper.

Handles the two loading backends (Hugging Face transformers, timm) and returns one
consistent result object: pooled vector, patch tokens, the full residual stream and
optionally attention matrices.

Token bookkeeping
-----------------
A ViT input is `[cls] [reg_0 .. reg_{R-1}] [patch_0 .. patch_{P-1}]`. Rather than
trust per-model config flags, we derive the prefix length from the actual sequence:
`num_prefix = T - (H // patch) * (W // patch)`. That survives register tokens, extra
prefix tokens, and odd image sizes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .attention import install_attention_hooks, remove_hooks
from .registry import ModelSpec, get_spec, resolve_timm_id

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pick_device(pref: Optional[str] = None) -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class EncoderOutput:
    pooled: np.ndarray            # (D,)          cls token (or pooled output)
    patch_tokens: np.ndarray      # (P, D)        patch tokens from the last block
    grid: tuple[int, int]         # (h, w)        patch grid
    hidden_states: np.ndarray     # (L+1, T, D)   residual stream, embedding + every block
    attentions: Optional[np.ndarray] = None   # (L, heads, T, T)
    num_prefix: int = 1
    input_size: tuple[int, int] = (0, 0)
    model_key: str = ""


class Encoder:
    """Loads a backbone and runs inference with optional attention capture."""

    def __init__(self, spec: ModelSpec, device: Optional[str] = None,
                 dtype: torch.dtype = torch.float32, backend: str = "hf",
                 hf_token: Optional[str] = None, preprocess: str = "model"):
        self.spec = spec
        self.device = pick_device(device)
        self.dtype = dtype
        self.backend = backend
        self.preprocess = preprocess
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.model = None
        self.processor = None
        self._attn_store = None
        self._attn_handles = []
        self._attention_supported = True
        self.ckpt_report = None
        self._resid_store: Optional[list] = None
        self._resid_handles: list = []

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, key: str, device: Optional[str] = None, backend: str = "auto",
             hf_token: Optional[str] = None, dtype: torch.dtype = torch.float32,
             preprocess: str = "model") -> "Encoder":
        spec = get_spec(key)
        if backend == "auto":
            backend = "hf" if spec.hf_id else "timm"
        enc = cls(spec, device=device, dtype=dtype, backend=backend, hf_token=hf_token,
                  preprocess=preprocess)
        if backend == "hf":
            try:
                enc._load_hf()
            except PermissionError:
                # Gated HF repo but an ungated timm mirror exists -> use that instead.
                if resolve_timm_id(spec):
                    enc.backend = "timm"
                    enc._load_timm()
                else:
                    raise
        else:
            enc._load_timm()
        enc.model.eval()
        return enc

    def _load_hf(self):
        from transformers import AutoImageProcessor, AutoModel

        kwargs = {"dtype": self.dtype}
        if self.spec.trust_remote_code:
            kwargs["trust_remote_code"] = True
        try:
            self.model = AutoModel.from_pretrained(
                self.spec.hf_id, attn_implementation="eager", token=self.hf_token, **kwargs
            )
        except TypeError:
            self.model = AutoModel.from_pretrained(self.spec.hf_id, token=self.hf_token, **kwargs)
        except Exception as e:
            if _looks_gated(e):
                raise PermissionError(
                    f"{self.spec.hf_id} is gated. Accept the licence on "
                    f"https://huggingface.co/{self.spec.hf_id} while logged in, then retry."
                ) from e
            raise
        self.model.to(self.device)
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                self.spec.hf_id, token=self.hf_token,
                trust_remote_code=self.spec.trust_remote_code,
            )
        except Exception:
            self.processor = None  # fall back to the registry's mean/std transform

    def _load_timm(self):
        import timm

        tid = resolve_timm_id(self.spec)
        if not tid:
            raise ValueError(f"no timm id resolvable for {self.spec.key}")
        self.timm_id = tid
        self.processor = None
        self.ckpt_report = None

        if self.spec.ckpt:
            from .checkpoints import download_checkpoint, load_into_timm

            path = download_checkpoint(
                self.spec.ckpt["repo"], self.spec.ckpt.get("file", "model.pth"),
                token=self.hf_token,
            )
            self.model, self.ckpt_report = load_into_timm(
                self.spec.ckpt.get("arch", tid), path,
                prefix=self.spec.ckpt.get("prefix", "backbone."),
                num_classes=0,
            )
        else:
            self.model = timm.create_model(tid, pretrained=True, num_classes=0)

        self.model.to(self.device)

    # --------------------------------------------------------------- transforms

    def transform(self, image) -> torch.Tensor:
        """PIL image (any mode) -> normalised (3, S, S) float tensor.

        Prefers the model's own HF image processor (correct per-model resize and
        intensity statistics); falls back to the registry's mean/std transform. Pass
        `preprocess="uniform"` to force the shared transform, which is what you want when
        comparing several models pixel-for-pixel.
        """
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = Image.fromarray(_to_uint8(image))
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.processor is not None and self.preprocess == "model":
            try:
                pv = self.processor(images=image, return_tensors="pt")["pixel_values"]
                return pv[0].to(torch.float32)
            except Exception:
                pass

        S = self.spec.image_size
        image = image.resize((S, S), Image.BICUBIC)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray(self.spec.mean, dtype=np.float32)
        std = np.asarray(self.spec.std, dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(arr))

    def effective_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """The mean/std actually applied by transform(), whichever path was taken."""
        if self.preprocess == "model" and self.processor is not None:
            m = getattr(self.processor, "image_mean", None)
            s = getattr(self.processor, "image_std", None)
            if m is not None and s is not None:
                return (np.asarray(m, dtype=np.float32), np.asarray(s, dtype=np.float32))
        return (np.asarray(self.spec.mean, dtype=np.float32),
                np.asarray(self.spec.std, dtype=np.float32))

    def preview(self, image) -> np.ndarray:
        """The image exactly as the network receives it, de-normalised back to viewable RGB.

        Without this the user never sees that the model got a 518px resize, a centre crop, or
        a different intensity mapping than the file they uploaded.
        """
        pv = self.transform(image).detach().cpu().numpy()      # (3, S, S)
        mean, std = self.effective_stats()
        arr = pv * std.reshape(3, 1, 1) + mean.reshape(3, 1, 1)
        arr = np.clip(arr, 0.0, 1.0)
        return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)

    def to_batch(self, images) -> torch.Tensor:
        if not isinstance(images, (list, tuple)):
            images = [images]
        return torch.stack([self.transform(im) for im in images]).to(self.device, dtype=self.dtype)

    # ------------------------------------------------------------------- forward

    def enable_attention_capture(self):
        if self._attn_store is None:
            self._attn_store, self._attn_handles = install_attention_hooks(self.model)

    def disable_attention_capture(self):
        remove_hooks(self._attn_handles)
        self._attn_handles = []
        self._attn_store = None

    @torch.no_grad()
    def __call__(self, images, capture_attention: bool = False,
                 capture_residual: bool = True) -> EncoderOutput:
        pv = self.to_batch(images)
        B, _, H, W = pv.shape
        grid = (H // self.spec.patch_size, W // self.spec.patch_size)

        attentions = None
        if self.backend == "hf":
            hidden_states, attentions, pooled, final_tokens = self._forward_hf(
                pv, capture_attention, capture_residual)
        else:
            hidden_states, attentions, pooled, final_tokens = self._forward_timm(
                pv, capture_attention, capture_residual)

        T = final_tokens.shape[0]
        num_prefix = max(T - grid[0] * grid[1], 0)
        patch_tokens = final_tokens[num_prefix:, :]

        return EncoderOutput(
            pooled=pooled,
            patch_tokens=patch_tokens,
            grid=grid,
            hidden_states=hidden_states,
            attentions=attentions,
            num_prefix=num_prefix,
            input_size=(H, W),
            model_key=self.spec.key,
        )

    def _forward_hf(self, pv, capture_attention: bool, capture_residual: bool):
        want_attn = capture_attention and self._attention_supported
        if capture_attention and not want_attn:
            pass
        call_kwargs = dict(pixel_values=pv, output_hidden_states=capture_residual)
        if want_attn:
            call_kwargs["output_attentions"] = True
            self.enable_attention_capture()  # belt and braces: built-in + hooks

        try:
            out = self.model(**call_kwargs)
        except (TypeError, ValueError):
            # Model rejects output_attentions -> rely on hooks only.
            self._attention_supported = False
            call_kwargs.pop("output_attentions", None)
            self.enable_attention_capture()
            out = self.model(**call_kwargs)

        hs = out.hidden_states if getattr(out, "hidden_states", None) is not None else (out.last_hidden_state,)
        hidden = torch.stack([h[0].float() for h in hs], dim=0)          # (L+1, T, D)

        final_tokens = out.last_hidden_state[0].float().cpu().numpy()
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[0, 0]
        pooled = pooled[0].float().cpu().numpy()

        attentions = None
        if capture_attention:
            builtin = getattr(out, "attentions", None)
            if builtin:
                attentions = torch.stack([a[0].float() for a in builtin], dim=0).cpu().numpy()
            elif self._attn_store is not None:
                st = self._attn_store.stacked()
                if st is not None:
                    attentions = st.cpu().numpy()
            if self._attn_store is not None:
                self._attn_store.clear()

        return hidden.cpu().numpy(), attentions, pooled, final_tokens

    def _forward_timm(self, pv, capture_attention: bool, capture_residual: bool):
        if capture_attention:
            self.enable_attention_capture()
        if capture_residual:
            self.enable_residual_capture()

        out = self.model.forward_features(pv)
        if isinstance(out, dict):
            tokens = out.get("x") or out.get("last_hidden_state")
        elif out.dim() == 3:
            tokens = out
        else:
            raise RuntimeError(f"unexpected timm output shape {tuple(out.shape)}")

        tokens = tokens[0].float()                                       # (T, D)
        grid = (pv.shape[-2] // self.spec.patch_size, pv.shape[-1] // self.spec.patch_size)
        num_prefix = tokens.shape[0] - grid[0] * grid[1]
        pooled = tokens[0].cpu().numpy()

        hidden = None
        if capture_residual and self._resid_store:
            st = self._resid_store
            parts = ([st["embed"]] if st.get("embed") is not None else []) + \
                    [t for t in st["blocks"] if t is not None]
            if parts and parts[0].shape[1] == tokens.shape[0]:
                # layer 0 = block-0 input (patch embed + position), then each block output
                hidden = torch.stack([p[0].float() for p in parts], dim=0)
            self._resid_store = None
        if hidden is None:
            hidden = tokens.unsqueeze(0)                                 # (1, T, D) fallback

        attentions = None
        if capture_attention and self._attn_store is not None:
            st_a = self._attn_store.stacked()
            attentions = st_a.cpu().numpy() if st_a is not None else None
            self._attn_store.clear()
        return hidden.cpu().numpy(), attentions, pooled, tokens.detach().cpu().numpy()

    # ------------------------------------------------- timm residual-stream hooks

    def enable_residual_capture(self):
        """Capture the residual stream of a timm ViT by hooking every transformer block.

        Layer 0 is the input to block 0 (patch embedding + position), layers 1..L are the
        block outputs. Without this, timm backbones would expose only the final tokens and
        layer-wise analysis would be impossible.
        """
        if self._resid_store is not None:
            return
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            return
        store: dict = {"embed": None, "blocks": [None] * len(blocks)}
        handles = []
        for i, blk in enumerate(blocks):
            def hook(_m, inp, out, idx=i):
                t = out[0] if isinstance(out, (tuple, list)) else out
                store["blocks"][idx] = t.detach()

            def pre_hook(_m, inp, idx=i):
                if idx == 0 and inp:
                    store["embed"] = inp[0].detach()

            handles.append(blk.register_forward_pre_hook(pre_hook))
            handles.append(blk.register_forward_hook(hook))
        self._resid_store = store
        self._resid_handles = handles

    def disable_residual_capture(self):
        remove_hooks(self._resid_handles)
        self._resid_handles = []
        self._resid_store = None

    # ---------------------------------------------------------------- residual

    @torch.no_grad()
    def residual_stream(self, images) -> np.ndarray:
        """(L+1, T, D) residual stream for a single image."""
        out = self(images, capture_attention=False, capture_residual=True)
        return out.hidden_states

    @torch.no_grad()
    def embed(self, images, pool: str = "cls", batch_size: int = 8) -> np.ndarray:
        """Pooled embeddings for many images: (N, D)."""
        if not isinstance(images, (list, tuple)):
            images = [images]
        vecs = []
        for i in range(0, len(images), batch_size):
            chunk = images[i:i + batch_size]
            for im in chunk:
                out = self(im, capture_attention=False, capture_residual=False)
                if pool == "cls":
                    v = out.pooled
                elif pool == "mean":
                    v = out.patch_tokens.mean(axis=0)
                elif pool == "max":
                    v = out.patch_tokens.max(axis=0)
                else:
                    raise ValueError("pool must be cls/mean/max")
                vecs.append(np.asarray(v, dtype=np.float32))
        return np.stack(vecs, axis=0)

    @torch.no_grad()
    def layer_features(self, images, pool: str = "mean", batch_size: int = 8,
                       progress=None) -> np.ndarray:
        """Per-layer pooled features for many images: (N, L+1, D).

        Pooling inside the loop keeps memory flat — materialising full token-level
        residual streams for a few hundred images would need gigabytes.
        """
        if not isinstance(images, (list, tuple)):
            images = [images]
        rows = []
        for i, im in enumerate(images):
            out = self(im, capture_attention=False, capture_residual=True)
            H = out.hidden_states[:, out.num_prefix:, :]          # (L+1, P, D)
            if pool == "mean":
                v = H.mean(axis=1)
            elif pool == "max":
                v = H.max(axis=1)
            elif pool == "cls":
                v = out.hidden_states[:, 0, :]
            else:
                raise ValueError("pool must be mean/max/cls")
            rows.append(v.astype(np.float32))
            if progress is not None and (i % 5 == 0 or i == len(images) - 1):
                progress((i + 1) / len(images), desc=f"layer features {i + 1}/{len(images)}")
        return np.stack(rows, axis=0)                             # (N, L+1, D)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    if a.max() <= 1.0 + 1e-6:
        a = a * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def _looks_gated(err: Exception) -> bool:
    s = str(err).lower()
    return any(t in s for t in ("gated", "401", "403", "unauthorized", "forbidden", "agreement"))


# ------------------------------------------------------------------ model cache

_CACHE: dict[tuple, Encoder] = {}


def get_encoder(key: str, device: Optional[str] = None, backend: str = "auto",
                hf_token: Optional[str] = None, preprocess: str = "model") -> Encoder:
    """Load (or reuse) an encoder. Keeps at most 2 models resident."""
    ck = (key, device, backend, preprocess)
    if ck in _CACHE:
        return _CACHE[ck]
    enc = Encoder.load(key, device=device, backend=backend, hf_token=hf_token,
                       preprocess=preprocess)
    _CACHE[ck] = enc
    if len(_CACHE) > 2:
        oldest = next(iter(_CACHE))
        if oldest != ck:
            try:
                _CACHE[oldest].model.to("cpu")
            except Exception:
                pass
            _CACHE.pop(oldest, None)
            import gc

            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    return enc

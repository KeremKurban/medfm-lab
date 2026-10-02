"""medfm — a local lab for medical vision foundation models.

Layers:
    registry   model catalogue (DINOv2/v3 + medical adaptations)
    encoders   uniform loader/inference wrapper (HF transformers + timm)
    attention  attention extraction, capture hooks, rollout, entropy
    embeddings patch-token feature maps, PCA/UMAP, similarity, registers
    residual   residual-stream capture, CKA, layer-wise linear probing
    data       MedMNIST+ at 224px, galleries and probe splits
    probe      frozen-feature linear / kNN probes
    viz        matplotlib figure helpers
"""

__version__ = "0.1.0"

"""Plain-language glossary for every term the UI shows.

Two registers, because they are used in different places:

    SHORT[term]  — a hover tooltip, one clause, no jargon.
    GLOSSARY[term] (title, body) — the full entry, for the Glossary tab.

Keep the short form genuinely short: it appears in an <abbr title> on a metrics row.
"""

from __future__ import annotations

# Hover tooltips, keyed by the exact label a metric is displayed under.
SHORT: dict[str, str] = {
    "embedding norm":
        "Length of the summary (cls) vector — how strongly the model is responding overall.",
    "mean |value|":
        "Average absolute size of the embedding entries. Near zero means a nearly dead vector.",
    "effective rank":
        "How many independent directions the patch features actually use, out of the full "
        "embedding width. Low = the model is compressing into a few directions.",
    "anisotropy":
        "Average cosine similarity between all pairs of patch vectors. High means every "
        "patch points the same way, i.e. a collapsed, uninformative representation.",
    "patch grid":
        "The image cut into a square grid of patches. A 224px image with 16px patches gives "
        "14x14 = 196 patches. Each patch becomes one token.",
    "patch size":
        "The side length in pixels of one patch. ViT-14 models cut 14px squares, ViT-16 "
        "models 16px squares. Smaller patches = finer detail, more compute.",
    "cls token":
        "An extra learned token prepended to the patch sequence. It has no pixels of its "
        "own; it aggregates the image and is normally used as the image embedding.",
    "register tokens":
        "Extra tokens (4 in DINOv2-with-registers and DINOv3) that give the model somewhere "
        "to park global information so it does not hijack ordinary patch tokens.",
    "attention":
        "For each token, how much it looks at every other token. High attention from the cls "
        "token to a patch means that patch contributed strongly to the summary.",
    "attention head":
        "One independent attention pattern inside a block. A ViT-B has 12 per block; they "
        "often specialise, so a single average can hide what is going on.",
    "attention rollout":
        "Attention multiplied through every layer, which approximates the path information "
        "takes from an input patch to the output. Usually far more readable than one layer.",
    "attention entropy":
        "How spread out or focused an attention pattern is, on a 0-1 scale. 1.0 = perfectly "
        "uniform (looking at everything equally, so uninformative). 0 = all on one patch.",
    "PCA":
        "Principal component analysis: finds the directions of greatest variation. Three "
        "components mapped to red/green/blue turn a feature vector into a colour.",
    "foreground mask":
        "Pixels that pass a brightness threshold on the first principal component. DINO "
        "models tend to put subject-versus-background in that component, so it behaves like "
        "a free segmentation.",
    "token UMAP":
        "A 2-D projection of all patch vectors that tries to keep similar ones close. Same "
        "colour group = the model represents those patches alike.",
    "k-means":
        "A simple clustering that groups similar vectors into k sets. Used here only to "
        "colour the UMAP, so the grouping is not meaningful on its own.",
    "cosine similarity":
        "Similarity between two vectors by angle rather than length: 1.0 identical "
        "direction, 0 unrelated, -1 opposite. Standard for comparing embeddings.",
    "positional bias":
        "The tendency of patch features to encode where a patch is, not just what it "
        "contains. DINOv3 is strong here, which can fake a convincing similarity map.",
    "positional debiasing":
        "Projects out the coordinate directions from the features so similarity reflects "
        "content instead of location.",
    "outlier token":
        "A patch token whose norm is far above the rest. These absorb global context, are "
        "not anatomy, and visibly distort attention maps.",
    "residual stream":
        "The running vector each token carries through the network, updated by every block. "
        "Everything the model knows at a given depth is linearly readable from it.",
    "CKA":
        "Centred kernel alignment: how similarly two layers represent the same tokens, from "
        "0 (unrelated) to 1 (identical). Bright off-diagonal blocks mean redundant layers.",
    "representation drift":
        "Cosine similarity between a token's vector at one layer and the next. Low values "
        "mean that block substantially rewrote the representation.",
    "linear probe":
        "A plain logistic regression trained on frozen features at one layer. If a simple "
        "linear model can read the label there, the features already contain it.",
    "kNN retrieval":
        "Finds the reference images whose embeddings are closest to this one. If the "
        "neighbours share the label, the representation is already grouped by diagnosis.",
    "quadratic weighted kappa":
        "Agreement score for ordered labels that penalises being two grades off more than "
        "one. The right metric for ordinal tasks like diabetic retinopathy grading.",
    "MPS":
        "Apple's Metal Performance Shaders backend — PyTorch on the Mac's GPU. Unsupported "
        "operations fall back to the CPU rather than failing.",
    "preprocessing":
        "model = each backbone's own image processor, which is correct for that model. "
        "uniform = one shared resize and intensity mapping, for comparing backbones fairly.",
    "ground truth":
        "The known correct answer for this image. MedMNIST+ samples have one; uploaded "
        "images do not.",
}


# Full entries for the Glossary tab: term -> (one-line summary, longer explanation).
GLOSSARY: dict[str, tuple[str, str]] = {
    "Patch, patch grid, patch size": (
        "How the image is chopped up before the model sees it.",
        "A vision transformer cannot ingest pixels directly, so the image is cut into a grid "
        "of fixed-size squares — 14x14 px for ViT-14, 16x16 px for ViT-16. Each square is "
        "flattened and projected into one *patch token*. A 224px image with 16px patches "
        "becomes a 14x14 grid of 196 tokens, and every spatial map in this app is drawn at "
        "that grid resolution. This is the single most important thing to hold in your head: "
        "**the model's spatial resolution is the patch grid, not the pixels.**",
    ),
    "CLS token (and register tokens)": (
        "The summary token, plus scratch space the model uses internally.",
        "A learned token is prepended to the patch sequence with no pixels of its own; after "
        "the blocks it has gathered a summary of the image, and it is what most pipelines use "
        "as *the* embedding. DINOv2-with-registers and DINOv3 add four more such tokens "
        "(*registers*) so the model has somewhere to stash global information instead of "
        "abusing ordinary patch tokens — which is what causes high-norm outliers.",
    ),
    "Effective rank": (
        "How many independent directions the features actually use.",
        "A ViT-B patch vector has 768 numbers, but '768 dimensions' does not mean 768 "
        "dimensions of real information. Effective rank is the participation ratio of the "
        "covariance spectrum. If it reads ~30 out of 768, the model is expressing everything "
        "in roughly thirty directions. Compare it between backbones: a domain-adapted model "
        "usually keeps more usable structure on its own modality.",
    ),
    "Anisotropy": (
        "Whether all patch vectors point the same way.",
        "The mean cosine similarity between every pair of patch vectors. Near zero means "
        "patches are spread across directions and distinguishable; above roughly 0.7 means "
        "they are crowding into one direction, which is representation collapse — the model "
        "would struggle to tell any two patches apart. Watch this after any fine-tuning.",
    ),
    "Attention, heads, rollout, entropy": (
        "What the model looks at, and how sharply.",
        "In every block each token asks how relevant every other token is (*attention*) and "
        "mixes in their information. Each block has several parallel *heads* that can "
        "specialise. *Rollout* multiplies attention through all layers to show the effective "
        "path from an input patch to the summary. *Entropy* scores how spread out a pattern "
        "is: 1.0 is uniform, so the head is looking everywhere equally and telling you "
        "nothing; near 0 means it sits on one patch. A head at 0.99 entropy is not "
        "'cautious', it is uninformative.",
    ),
    "PCA and the PCA→RGB map": (
        "Compressing features to colour so your eye can read them.",
        "Principal component analysis finds the directions of greatest variation in the patch "
        "vectors. Taking the top three and mapping them to red, green and blue turns a "
        "768-dimensional vector per patch into one colour — this is exactly the visualisation "
        "the DINO papers use. **Read it as: patches that share a colour are represented "
        "similarly by the model.** Coherent coloured anatomy means the backbone sees "
        "structure; confetti means it does not. Because each patch is one pixel of this "
        "image, it is drawn enlarged with hard edges.",
    ),
    "Foreground mask (PC1)": (
        "A free segmentation the model never had to be trained for.",
        "Plot the first principal component as brightness and threshold it. These backbones "
        "usually encode *subject versus background* in that first component, so the threshold "
        "behaves like a segmentation map — the DINO papers use this exact trick to cut out "
        "the object. The tinted region is what the model treats as foreground. It is a sanity "
        "check on whether the model separates tissue from empty space, nothing more.",
    ),
    "Token UMAP": (
        "A map of which patches the model considers alike.",
        "UMAP squeezes the patch vectors down to 2-D while trying to keep similar ones close, "
        "so points that land together are represented similarly. Colours come from k-means, "
        "which just groups them into k buckets — the grouping is only there to make the "
        "picture readable and carries no meaning by itself. The useful question is whether "
        "clusters line up with anatomy.",
    ),
    "Positional bias and debiasing": (
        "Why a similarity map can look convincing and mean nothing.",
        "Patch features encode not only content but *where the patch is*. In DINOv3 this is "
        "strong enough that a similarity map often looks like a cross centred on the query "
        "patch — that is the coordinates talking, not the anatomy. Debiasing projects the "
        "coordinate directions out of the features; if a map changes a lot when you toggle "
        "it, suspect position rather than content.",
    ),
    "Outlier tokens": (
        "A few patches that hijack the model's attention.",
        "Vision transformers routinely dump global context into a handful of patch tokens "
        "whose norms end up an order of magnitude above the rest. They are not anatomy, they "
        "act as spare registers, and they are the usual reason an attention map seems to "
        "ignore the image. Count them before concluding the model is broken.",
    ),
    "Residual stream": (
        "What each token carries as it passes through the network.",
        "Every token is a vector, and each block adds its output back onto it — that running "
        "vector is the residual stream. Crucially, everything the model knows at depth L is "
        "linearly readable from the stream at layer L, which is what makes layer-wise probing "
        "meaningful rather than decorative. Watch the norm growth: it usually climbs steeply, "
        "and a few tokens shoot up much faster than the rest.",
    ),
    "CKA": (
        "How similar two layers' representations are.",
        "Centred kernel alignment compares the similarity structure of the same tokens at two "
        "different layers, scored 0 to 1. It answers 'did this block actually change "
        "anything?'. A bright block of near-1 values along the diagonal means those layers "
        "are largely redundant, which is a good hint about where not to spend adaptation "
        "capacity.",
    ),
    "Representation drift": (
        "How much each individual block rewrites the representation.",
        "The cosine similarity between a token's vector at layer l and at layer l+1. Values "
        "close to 1 mean the block barely touched it; a dip marks a block doing real work. "
        "Companion to CKA: CKA compares whole layers, drift shows where the change happens.",
    ),
    "Linear probe (layer-wise)": (
        "The measurement that tells you whether to adapt at all.",
        "A plain logistic regression trained on the frozen features at every layer in turn. "
        "If a late layer already separates the classes, the features contain the label and "
        "you need a classifier head, not domain adaptation. If accuracy peaks mid-network "
        "and then *falls*, the later layers have specialised away from your modality — the "
        "classic signature of a backbone from the wrong domain, and exactly when adaptation "
        "pays. The printed block range is derived from this curve.",
    ),
    "Ordinal tasks and QWK": (
        "Why accuracy is the wrong score for retinamnist.",
        "Some labels are ordered — diabetic retinopathy grades 0 to 4, where being one grade "
        "off is not the same as being four off. Plain accuracy treats every mistake as equal, "
        "so it understates a model that is close but not exact. Quadratic weighted kappa "
        "penalises distance from the true grade and is the standard metric for these tasks. "
        "The probe here reports accuracy for every dataset; for retinamnist read it with "
        "that caveat in mind.",
    ),
    "kNN retrieval": (
        "Testing whether the representation is already organised by diagnosis.",
        "Embed a labelled reference set (the gallery) once, then find the reference images "
        "closest to your query. If the neighbours share its label, the backbone's feature "
        "space is already grouped clinically and a simple head will work. If they are random, "
        "the features do not separate this modality. **The query is excluded from its own "
        "gallery** — otherwise the top hit is the image itself at similarity 1.0, which looks "
        "like a perfect result and measures nothing.",
    ),
    "Cosine similarity": (
        "How alike two vectors are, ignoring their length.",
        "The cosine of the angle between them: 1.0 same direction, 0 unrelated, -1 opposite. "
        "Standard for comparing embeddings because it ignores magnitude differences that "
        "carry no meaning. It is what attention, retrieval and drift all use here.",
    ),
    "MPS (Apple GPU) and fallbacks": (
        "How this runs on your Mac.",
        "Metal Performance Shaders is Apple's GPU backend for PyTorch. Some operations are "
        "unimplemented on it, so the app sets a fallback flag that silently runs those on the "
        "CPU instead of crashing. Outputs therefore match CUDA numerically to within normal "
        "floating-point tolerance, just with a few CPU round-trips.",
    ),
    "Preprocessing: model vs uniform": (
        "Two ways to feed an image in.",
        "*model* uses each backbone's own image processor — correct resolution and intensity "
        "statistics for that specific model, and what you want in normal use. *uniform* "
        "forces one shared resize and mean/std across every backbone so that the only "
        "difference between two runs is the weights; use it when comparing backbones.",
    ),
}

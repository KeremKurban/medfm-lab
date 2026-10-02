"""Headless exercise of every UI callback, plus a regression check on the UI value path.

The value-path check matters: a component's *display label* and its *internal value* are
different things, and Gradio resolves the configured initial value through the label list.
A callback called directly with a hard-coded key therefore proves nothing about what the
browser actually sends. `check_ui_values` simulates that round-trip for every dropdown.

Run:  .venv/bin/python scripts/test_app.py
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import app as lab
from medfm import data as mdata
from medfm import registry

RESULTS: list[tuple[str, str, str]] = []


def step(name: str, fn, *args, **kwargs):
    try:
        res = fn(*args, **kwargs)
        if isinstance(res, tuple):
            detail = ", ".join(
                "None" if r is None
                else (f"arr{r.shape}" if isinstance(r, np.ndarray)
                      else (f"str[{len(r)}]" if isinstance(r, str) else type(r).__name__))
                for r in res
            )
        else:
            detail = f"str[{len(res)}]" if isinstance(res, str) else type(res).__name__
        RESULTS.append((name, "OK", detail))
        print(f"[OK]   {name:28s} -> {detail}")
        return res
    except Exception as e:
        RESULTS.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name:28s} -> {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def gradio_roundtrip(comp):
    """What value would the browser actually send for this component's current selection?

    Gradio stores choices as (display_label, internal_value). If the configured `value`
    matches a *label*, Gradio resolves it to that choice's value. Reproducing this is the
    only way to catch label/value swap bugs without a browser.
    """
    choices = getattr(comp, "choices", None) or []
    labels, values = [], []
    for c in choices:
        if isinstance(c, (tuple, list)) and len(c) == 2:
            labels.append(c[0])
            values.append(c[1])
        else:
            labels.append(c)
            values.append(c)
    v = getattr(comp, "value", None)
    if v in labels:
        return values[labels.index(v)], labels
    return v, labels


def backbone_dropdowns(demo):
    for _cid, comp in demo.blocks.items():
        if type(comp).__name__ != "Dropdown":
            continue
        if "backbone" in (getattr(comp, "label", "") or "").lower():
            yield comp


def check_ui_values(demo) -> tuple[bool, list[str]]:
    """Validate every dropdown against what the widget layers actually enforce.

    Invariants, all learned the hard way on Gradio 6:
      * choices must be PLAIN STRINGS — with (name, value) tuples the Python layer validates
        `value` against the value while the frontend validates against the name, producing
        contradictory errors and a silently dead app;
      * the configured `value` must be one of the choices;
      * whatever the widget hands the callback must be resolvable — a backbone label must map
        to a model key, a dataset label must map back to a medmnist flag.
    """
    problems = []
    seen = 0
    for _cid, comp in demo.blocks.items():
        if type(comp).__name__ != "Dropdown":
            continue
        label = getattr(comp, "label", "") or ""
        if not label:
            continue
        seen += 1
        choices = list(getattr(comp, "choices", None) or [])
        # Gradio normalises plain strings to (x, x) tuples internally, so the invariant to
        # assert is name == value — an intentional (name, value) pair is exactly what breaks.
        mismatched = [c for c in choices
                      if isinstance(c, (tuple, list)) and len(c) == 2 and c[0] != c[1]]
        if mismatched:
            problems.append(f"[{label}] {len(mismatched)} choices have name != value "
                            f"(e.g. {mismatched[0]!r})")
        names = [c[0] if isinstance(c, (tuple, list)) else c for c in choices]
        values = [c[1] if isinstance(c, (tuple, list)) and len(c) == 2 else c for c in choices]
        v = getattr(comp, "value", None)
        if v not in names and v not in values:
            problems.append(f"[{label}] value {v!r} is not among its own choices")
        if names != values:
            problems.append(f"[{label}] choice names and values diverge")

        is_model = "backbone" in label.lower()
        for c in values:
            try:
                if is_model:
                    registry.resolve_key(c)
                elif c in mdata.MEDMNIST_2D or " — " in str(c):
                    mdata.flag_from_option(c)
            except Exception as e:
                problems.append(f"[{label}] choice {c!r} unresolvable: {e}")

    print(f"[OK]   ui value path               -> {seen} dropdowns: plain-string choices, "
          f"values in range, all resolvable back to keys/flags")
    if seen < 4:
        problems.append(f"only {seen} dropdowns found — expected at least 4")
    return (not problems), problems


def main():
    print("=== building app (validates wiring) ===")
    try:
        demo = lab.build_app()
        print(f"[OK]   build_app -> {len(demo.blocks)} blocks/components")
        RESULTS.append(("build_app", "OK", f"{len(demo.blocks)} blocks"))
    except Exception as e:
        RESULTS.append(("build_app", "FAIL", str(e)))
        traceback.print_exc()
        print("\nApp failed to build; stopping.")
        return

    print("\n=== UI value path (the label/value regression) ===")
    ok, problems = check_ui_values(demo)
    RESULTS.append(("ui_value_path", "OK" if ok else "FAIL",
                    "clean" if ok else "; ".join(problems)))
    for p in problems:
        print(f"[FAIL] ui value path             -> {p}")

    # exercise the real UI path: exact strings the dropdowns hand back
    sent = lab.DEFAULT_MODEL
    dataset = lab.DATASET_CHOICES[0]
    print(f"\n=== exercising callbacks with the browser-supplied value {sent!r} ===")

    step("load_model(label form)", lab.load_model, sent, "model")
    step("load_model(key form)", lab.load_model, "rad-dino", "model")
    r = step("fetch_sample", lab.fetch_sample, dataset, 3)

    # Provenance regression: analyse the fetched sample — it must stay labelled as a sample,
    # not be relabelled as an upload by the change event Gradio re-emits.
    if r is not None:
        a = step("analyze (sample provenance)", lab.analyze, r[0], True, 8)
        summary, status = (a[0], a[7]) if a else ("", "")
        if "test[" in status and "Uploaded" not in status:
            RESULTS.append(("sample provenance", "OK", status[:40]))
            print(f"[OK]   sample provenance           -> {status[:70]}")
        else:
            RESULTS.append(("sample provenance", "FAIL", status[:60]))
            print(f"[FAIL] sample provenance           -> {status[:60]}")
        # the "what the model saw" panel must match the resolution the network consumed
        if a and a[2] is not None:
            seen_shape = a[2].shape
            ih, iw = lab.LAB.out.input_size
            ok_seen = seen_shape[0] == ih and seen_shape[1] == iw
            RESULTS.append(("model-input panel", "OK" if ok_seen else "FAIL", str(seen_shape)))
            print(f"[{'OK' if ok_seen else 'FAIL'}]   model-input panel          -> "
                  f"{seen_shape[:2]} vs model input {(ih, iw)}")
        # and the PCA panel must be upscaled for display, not left at grid resolution
        if a and a[3] is not None:
            g = lab.LAB.out.grid
            ok_up = a[3].shape[0] > g[0] * 2
            RESULTS.append(("PCA panel upscaled", "OK" if ok_up else "FAIL",
                            f"{a[3].shape[:2]} from grid {g}"))
            print(f"[{'OK' if ok_up else 'FAIL'}]   PCA panel upscaled         -> "
                  f"grid {g} drawn at {a[3].shape[:2]}")
        # ground truth must be stated in the summary
        if a and "ground truth" in a[0]:
            RESULTS.append(("ground truth shown", "OK", ""))
            print("[OK]   ground truth shown         -> summary states the label")
        else:
            RESULTS.append(("ground truth shown", "FAIL", ""))
            print("[FAIL] ground truth shown         -> no label in summary")
    else:
        step("analyze", lab.analyze, lab.LAB.image, True, 8)

    # and a genuinely different image must be detected as an upload
    altered = np.ascontiguousarray(lab.LAB.image.copy())
    altered[0:8, 0:8] = 255 - altered[0:8, 0:8]
    a2 = step("analyze (upload provenance)", lab.analyze, altered, True, 8)
    status2 = a2[7] if a2 else ""
    if "Uploaded" in status2:
        RESULTS.append(("upload provenance", "OK", status2[:40]))
        print(f"[OK]   upload provenance           -> {status2[:70]}")
    else:
        RESULTS.append(("upload provenance", "FAIL", status2[:60]))
        print(f"[FAIL] upload provenance           -> {status2[:60]}")

    step("render_attention", lab.render_attention, 11, 0, "mean", 0.0, "turbo")
    step("render_embeddings", lab.render_embeddings, 6, 10, 10, False, 0.25)
    step("render_embeddings(debias)", lab.render_embeddings, 6, 10, 10, True, 0.25)
    step("render_residual", lab.render_residual)
    step("build_gallery", lab.build_gallery, dataset, 24, "cls")
    # re-fetch so there is a ground-truth label for the retrieval verdict (the upload step
    # above deliberately cleared it)
    lab.fetch_sample(dataset, 5)
    ag = step("analyze with gallery", lab.analyze, lab.LAB.image, True, 4)
    if ag and ("nearest neighbour is a" in ag[0] and "ground truth" in ag[0]):
        RESULTS.append(("kNN verdict shown", "OK", ""))
        print("[OK]   kNN verdict shown          -> summary compares retrieval to the label")
    else:
        RESULTS.append(("kNN verdict shown", "FAIL", ""))
        print("[FAIL] kNN verdict shown          -> no agreement statement in summary")

    # regression: the query must not be able to match itself in the gallery. A self-hit scores
    # 1.0 and always agrees with the label, so retrieval looks perfect while measuring nothing.
    if ag:
        top1 = None
        for line in ag[0].splitlines():
            if line.startswith("- top-1 neighbour:"):
                top1 = float(line.rsplit("similarity", 1)[1].strip())
        in_gallery = lab.sample_index_in_gallery()
        if top1 is None:
            RESULTS.append(("self-match excluded", "FAIL", "no top-1 line"))
            print("[FAIL] self-match excluded        -> no top-1 line in summary")
        elif in_gallery and top1 >= 0.999:
            RESULTS.append(("self-match excluded", "FAIL", f"top1={top1}"))
            print(f"[FAIL] self-match excluded        -> top-1 similarity {top1} (self-hit)")
        else:
            RESULTS.append(("self-match excluded", "OK", f"top1={top1:.3f}"))
            print(f"[OK]   self-match excluded        -> top-1 similarity {top1:.3f} "
                  f"(query was in the gallery: {in_gallery})")
    step("render_cka_multi", lab.render_cka_multi, dataset, 4)
    step("run_layer_probe", lab.run_layer_probe, dataset, 5, "mean")
    step("registry_table", lab.registry_table)

    print("\n=== switching model (2nd backbone) ===")
    ct = next(d for d in lab.DATASET_CHOICES if d.startswith("organamnist"))
    step("load meddinov3", lab.load_model, registry.label_for("meddinov3-vitb16"), "uniform")
    step("fetch ct sample", lab.fetch_sample, ct, 0)
    step("analyze ct", lab.analyze, lab.LAB.image, True, 5)
    step("attention ct", lab.render_attention, 4, 1, "max", 0.3, "magma")
    step("embeddings ct", lab.render_embeddings, 8, 5, 5, False, 0.3)
    step("residual ct", lab.render_residual)

    print("\n=== empty states (friendly text, never a traceback or stack) ===")
    lab.LAB.clear_analysis()
    lab.LAB.encoder = None
    for name, fn, args in [
        ("attention empty", lab.render_attention, (0, 0, "mean", 0.0, "turbo")),
        ("embeddings empty", lab.render_embeddings, (6, 1, 1, False, 0.25)),
        ("residual empty", lab.render_residual, ()),
        ("probe empty", lab.run_layer_probe, (lab.DATASET_CHOICES[0], 5, "mean")),
        ("gallery empty", lab.build_gallery, (lab.DATASET_CHOICES[0], 16, "cls")),
    ]:
        r = step(name, fn, *args)
        txt = r[0] if isinstance(r, tuple) else r
        if isinstance(txt, str) and ("Traceback" in txt or "File \"" in txt):
            RESULTS.append((name + " leak", "FAIL", "traceback leaked into the UI"))
            print(f"[FAIL] {name} leaked a traceback into the UI")

    print("\n=== reference content (dataset guides, glossary, hover tooltips) ===")
    from medfm import glossary as gl
    from medfm import guides

    bad_guides = [f for f in mdata.MEDMNIST_2D
                  if len(guides.dataset_guide_markdown(f) or "") < 200]
    if bad_guides:
        RESULTS.append(("dataset guides", "FAIL", str(bad_guides)))
        print(f"[FAIL] dataset guides             -> thin or missing: {bad_guides}")
    else:
        RESULTS.append(("dataset guides", "OK",
                        f"{len(mdata.MEDMNIST_2D)} datasets"))
        print(f"[OK]   dataset guides             -> all {len(mdata.MEDMNIST_2D)} have "
              f"class-level descriptions")

    # the specific complaint: retinamnist labels are bare digits 0-4
    ret = guides.dataset_guide_markdown("retinamnist")
    grades = all(f"`{i}`" in ret for i in range(5))
    has_meaning = "microaneurysm" in ret.lower() and "neovascularisation" in ret.lower()
    if grades and has_meaning:
        RESULTS.append(("retinamnist explained", "OK", "5 grades with clinical meaning"))
        print("[OK]   retinamnist explained      -> all 5 ordinal grades given clinical meaning")
    else:
        RESULTS.append(("retinamnist explained", "FAIL", f"grades={grades} meaning={has_meaning}"))
        print(f"[FAIL] retinamnist explained      -> grades={grades} meaning={has_meaning}")

    missing_tips = [t for t in gl.GLOSSARY if not gl.SHORT.get(t.split(" (")[0])]
    n_short = len(gl.SHORT)
    if n_short < 15:
        RESULTS.append(("glossary tooltips", "FAIL", f"only {n_short}"))
        print(f"[FAIL] glossary tooltips          -> only {n_short} hover definitions")
    else:
        RESULTS.append(("glossary tooltips", "OK", f"{n_short} hover definitions"))
        print(f"[OK]   glossary tooltips          -> {n_short} hover definitions, "
              f"{len(gl.GLOSSARY)} full entries")

    # the empty-state block above cleared the analysis, so rebuild a minimal one
    if lab.LAB.out is None or lab.LAB.encoder is None:
        lab.load_model(lab.DEFAULT_MODEL, "model")
        lab.fetch_sample(lab.DEFAULT_DATASET, 3)
        lab.analyze(lab.LAB.image, False, 4)

    mhtml = lab._metrics_html(lab.LAB.out, lab.LAB.encoder.spec, lab.LAB.out.pooled)
    n_abbr = mhtml.count("<abbr title=")
    if n_abbr >= 5:
        RESULTS.append(("metric tooltips", "OK", f"{n_abbr} abbr"))
        print(f"[OK]   metric tooltips            -> {n_abbr} hover definitions in the "
              f"metrics block")
    else:
        RESULTS.append(("metric tooltips", "FAIL", f"{n_abbr} abbr"))
        print(f"[FAIL] metric tooltips            -> only {n_abbr} definitions rendered")

    model_gaps = [k for k in registry.MODELS if not guides.model_guide_markdown(k)]
    if model_gaps:
        RESULTS.append(("model guides", "FAIL", str(model_gaps)))
        print(f"[FAIL] model guides               -> missing for {model_gaps}")
    else:
        RESULTS.append(("model guides", "OK", f"{len(registry.MODELS)} models"))
        print(f"[OK]   model guides               -> all {len(registry.MODELS)} backbones "
              f"described")

    print("\n=== summary ===")
    n_ok = sum(1 for _, s, _ in RESULTS if s == "OK")
    print(f"{n_ok}/{len(RESULTS)} checks OK")
    for n, s, d in RESULTS:
        if s != "OK":
            print(f"  FAIL {n}: {d}")


if __name__ == "__main__":
    main()

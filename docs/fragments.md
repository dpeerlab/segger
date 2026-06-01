# Fragments: recovering unassigned transcripts

> Status: design + implementation guide for `feat/fragments`. This is the single home for all
> fragment work in segger.

## Why fragments?

Image-based spatial transcriptomics (Xenium, MERSCOPE, CosMx) assigns each transcript to a cell by
nucleus-seeded segmentation + expansion. Across 14 public datasets, **up to ~40% of transcripts
remain unassigned**, and troutpy ([*Exploration of RNA outside segmented cells in spatial
transcriptomics reveals extrasomatic RNA organization*, bioRxiv, Dec 2025](https://www.biorxiv.org/content/10.64898/2025.12.07.692889v1))
shows that **~1/3 of that unassigned RNA is not technical noise** — it is real biology that current
segmentation drops on the floor. It is **enriched around cells with complex morphology — neurons,
glia, endothelial cells** — and corresponds to transcripts sitting in **protrusions and extrasomatic
compartments** (axons, dendrites, processes) that reach beyond the nuclear-centred boundary.

Two failure modes follow:

1. **Truncated cells.** A neuron's soma is segmented, but its dendritic/axonal transcripts fall
   outside the expanded boundary and are dropped. The cell is real but *amputated* — its shape
   collapses to a blob and its expression profile loses the genes that live in processes.
2. **Missed cells.** Cells with weak/absent nuclear signal (or off the staining plane) never get a
   seed at all, so a spatially-coherent cluster of transcripts is left fully unassigned.

segger's strength is the **learned transcript embedding** `f(t_i)` — a GNN representation that fuses
gene identity with spatial neighbourhood. Crucially, **the embedding is most trustworthy where the
graph is rich**: near boundaries/stainings, where a transcript has many tx–tx and tx–bd edges, the
embedding strongly reflects the local cell. Far from any staining, edges are sparse and the
embedding is weaker. Fragment recovery is designed around this gradient: lean on the embedding where
it is strong (extend confidently), fall back to spatial+embedding community structure where it is
weak (cluster cautiously), and refuse to invent cells from pure noise.

**Invariant (non-negotiable):** fragment recovery is **additive**. It only ever touches transcripts
that primary segmentation left unassigned (`segger_cell_id is null`); assigned transcripts are never
moved or re-segmented. Every stage coalesces *into nulls* only.

## The two mechanisms

Recovery runs as an ordered pipeline on the unassigned pool. Order matters biologically:
protrusion RNA should rejoin its source cell *before* we consider it an orphan.

### Stage A — Extend (attach to a source cell) — *primary for neurons*

For each unassigned transcript that lies near an existing cell, segger already computed (in
`predict_step`) its best candidate boundary and the tx↔bd embedding cosine `max_sim` — even when
that similarity fell *below* the per-gene acceptance threshold and the transcript was rejected.
Stage A revisits those near-miss rejections under a **relaxed** threshold:

> attach `t_i` to its best candidate cell `c_j` iff
> `sim(t_i, c_j) ≥ θ_gene(feature_i) − extend_shift`  (or `≥ extend_min_similarity` if fixed).

`θ_gene` is the per-gene threshold already computed in `writer.assign_transcripts_to_cells`. The
relaxation is deliberately small and is gated on the **real GNN tx–bd cosine**, which is exactly the
signal that is strong near stainings — so extension pulls in protrusion transcripts that genuinely
resemble the adjacent cell, and recovers elongated/neuronal shape by *growing the real cell* rather
than fabricating a neighbour. Extended transcripts **keep the existing cell id** and are tagged
`segger_assignment_source = "extended"`.

This is the "pixel/boundary extension based on segger scores" idea, made principled: not a
morphological dilation, but a score-gated re-test of rejected neighbours of real cells.

### Stage B — Cluster residual orphans into de-novo fragments

Transcripts still unassigned after Stage A are the candidates for new "fragment cells" — these model
the *missed-cell* case. This is `assign_fragments(xy, emb, config)` (`prediction/fragment.py`):

1. Spatial k-NN graph over orphan coordinates; edge weight = embedding cosine; prune edges below
   `edge_threshold`. (k-NN + embedding weighting = arbitrary/elongated shapes survive as connected
   chains, but only when expression is consistent along them.)
2. Connected components; split any component larger than `max_transcripts` via recursive Leiden
   (cuGraph GPU) / threshold-cut (CPU) so one runaway chain can't swallow a region.
3. Merge mutually-best adjacent communities when mean-embedding cosine ≥ `merge_threshold` and the
   union stays within the size cap (this is the **"merge small fragments together"** step).
4. Drop communities below `min_transcripts` → noise (`-1`), i.e. refuse to invent cells from a
   handful of scattered transcripts.

Output: new cell ids `fragment-<id>`, tagged `segger_assignment_source = "fragment"`.

### Stage C (optional) — Extend fragments

After Stage B, optionally re-run Stage A's logic at the *fragment* level: attach a whole small
fragment to an adjacent real cell when their mean embeddings are concordant. Flag: `--extend-fragments`.

## Why this design (and what was rejected)

- **Connected components on static per-gene reference vectors** (the original implementation):
  rejected. Two transcripts of the same gene had cosine ≈ 1 regardless of context, so it clustered
  by gene co-occurrence, not by learned cell identity, and a single bridging edge could merge
  unrelated blobs (CC is transitive — no density/community notion, no noise label).
- **Learned-embedding Leiden communities + noise rejection** (current Stage B): accepted. Leiden is
  the scverse-standard community detection; reusing it on both GPU and CPU keeps results consistent
  and avoids a bespoke heuristic. It handles elongated shapes (graph chains) while resisting
  over-merge (modularity + size cap).
- **Extension before clustering** (Stage A): accepted. troutpy's central finding is that
  complex-morphology RNA *belongs to* a nearby cell. Clustering it into a standalone "fragment" would
  create a fake cell adjacent to the real one and *worsen* both cells' profiles. Attaching first is
  the biologically correct and metric-improving move (recovers shape without inflating cell counts).
- **Morphological pixel dilation of boundaries**: rejected as the primary tool. It ignores
  expression and would pull in contaminating neighbours; the score-gated re-test achieves the same
  shape recovery with specificity.

## Output schema / naming contract

Single source of truth for downstream (segbench, export, troutpy concordance):

| `segger_cell_id` | `segger_assignment_source` | meaning |
|---|---|---|
| `<cell id>` | `primary` | assigned by primary segmentation |
| `<cell id>` | `extended` | unassigned tx attached to existing cell (Stage A) |
| `fragment-<id>` | `fragment` | de-novo fragment cell (Stage B) |
| `null` | `null` | remains unassigned (noise / below threshold) |

`utils/fragment_outputs.py` keys object type on the `fragment-` prefix and provenance on
`segger_assignment_source`, so exports/metrics can split cell / extended / fragment / unassigned
cleanly.

## Parameters

| Param | Default | Role |
|---|---|---|
| `--fragment-mode` | off | enable recovery |
| `--extend-mode` | off | enable Stage A (separate switch) |
| `--extend-min-similarity` | none | fixed tx–bd cosine for attach (overrides per-gene) |
| `--extend-similarity-shift` | 0.05 | subtractive relaxation on per-gene θ for attach |
| `--fragment-min-transcripts` | 50 | min size of a fragment cell (noise floor) |
| `--fragment-max-transcripts` | 5000 | size cap (triggers Leiden split) |
| `--fragment-n-neighbors` | 15 | spatial k-NN degree |
| `--fragment-edge-threshold` | 0.0 | prune k-NN edges below this embedding cosine |
| `--fragment-resolution` | 1.0 | Leiden resolution |
| `--fragment-merge-threshold` | 0.6 | min mean-embedding cosine to merge communities |

## Validation

Recovery must *improve* quality, not just raise the assignment rate. Evaluate fragment-off vs
extend-only vs extend+cluster vs the old-CC baseline with segbench (`segbench/src/segbench/metrics/`):

- **Recovery** — `transcript_assignment` rate should rise.
- **Specificity (guardrails)** — `MECR` (lower better) and positive marker purity / `marker_purity`
  (higher better) must NOT degrade; this catches over-extension bridging unlike cells.
- **Morphology** — `geometric.py` `elongation` / `circularity` / `geometric_coherency`: recovered
  neuron cells/extensions should look realistically elongated, not artificially round.
- **Biology** — `scripts/analyze_fragment_biology.py`: assignment recovery, per-cell-type
  composition, marker dotplots, spatial maps, and **troutpy concordance** (recovered objects should
  be "cell-like" by local composition). Run on mouse brain where neurons are the target morphology.
- **Bake-off** — under identical Stage A, compare Leiden vs HDBSCAN for Stage B on
  geometric_coherency + MECR + PMP and lock the default empirically.

## Implementation status / roadmap

**Done (current `feat/fragments`):** learned-embedding k-NN graph + Leiden communities + merge +
noise rejection (`prediction/fragment.py`); learned tx embeddings surfaced from `predict_step`;
fragment CLI flags; `tests/test_fragment_mode.py`.

**Roadmap:**

1. **Stage 0 — consolidate.** Commit the WIP; port `utils/fragment_outputs.py` and the export tests
   (`test_fragment_exports.py`, `test_spatialdata_fragment_exports.py`) from `v2-incremental`; delete
   dead code; drop the obsolete `--fragment-similarity-threshold` flag.
2. **Stage A — cell extension (new).** Surface best candidate `(bd, sim)` from `predict_step`; add
   `prediction/recovery.py` with `extend_cells(...)` + `recover_unassigned(...)` composing A→B→(C);
   make `writer._apply_fragment_mode` a thin adapter writing `segger_assignment_source`; add
   `--extend-*` CLI flags.
3. **Stage B/C — polish.** Add CPU Leiden (`leidenalg`+`igraph`, soft dep with `_threshold_cut`
   fallback) for GPU/CPU parity; expose `--extend-fragments`.
4. **Tests.** `test_recovery_extend.py`, `test_fragment_cpu_leiden.py`, ported export tests, a writer
   integration test on mock predictions.

### Module layout

```
prediction/
  fragment.py     # assign_fragments(xy, emb, config)  — Stage B/C (unchanged API)
  recovery.py     # extend_cells(...) + recover_unassigned(...)  — orchestrator (new)
utils/
  fragment_outputs.py   # object_type / provenance splitting (ported)
```

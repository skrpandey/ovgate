# Released per-detection signals

`detections_coco.csv` and `detections_open_images.csv`: one row per detection
with `image`, `cls`, `confidence`, `clip_score`, `margin`, `softmax_scaled`,
`truncated`, and the non-circular correctness label `y`.

Every number in the paper's gate table is reproducible from these two files
alone, with no model inference and no access to the source imagery. They carry
no participant data: both are derived from public benchmark images
(COCO-2017 val, Open Images V7 val).

`embeddings_{coco,open_images}.npz`: the underlying arrays, row-aligned with
the CSVs: `E` (n x 768 unit-norm cutout embeddings), `bbox` (n x 4), `image`,
`cls`, plus the label-set text embeddings `T`, `labels`, and the model's
`logit_scale`. These make every read-out recomputable from scratch (the CSVs'
`clip_score` and `margin` reproduce to within 4e-4) and support crop-level
visualization and label-set-size analyses without any model inference. Derived
entirely from public benchmark imagery.

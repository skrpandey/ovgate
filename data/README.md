# Released per-detection signals

`detections_coco.csv` and `detections_open_images.csv`: one row per detection
with `image`, `cls`, `confidence`, `clip_score`, `margin`, `softmax_scaled`,
`truncated`, and the non-circular correctness label `y`.

Every number in the paper's gate table is reproducible from these two files
alone, with no model inference and no access to the source imagery. They carry
no participant data: both are derived from public benchmark images
(COCO-2017 val, Open Images V7 val).

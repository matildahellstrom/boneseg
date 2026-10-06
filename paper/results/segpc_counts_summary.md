# Counting plasma cells on SegPC-2021

Minimum cell size chosen on 60 training images by cell F1; the 199 validation images with outlined cells.

| Method | Condition | Min. cell size | Precision | Recall | F1 | Count bias per image | Count ICC | Official score |
|---|---|---|---|---|---|---|---|---|
| boneseg, clicks | 10+20 clicks | 16000 px | 0.450 | 0.468 | 0.458 | +0.20 | 0.63 | 0.513 |
| boneseg, clicks | 3+6 clicks | 16000 px | 0.311 | 0.295 | 0.303 | -0.25 | 0.41 | 0.385 |
| boneseg learned model | 5 labelled training images | 16000 px | 0.328 | 0.292 | 0.309 | -0.54 | 0.42 | 0.398 |
| Random forest | 10+20 clicks | 16000 px | 0.205 | 0.204 | 0.205 | -0.03 | 0.38 | 0.328 |
| Random forest | 3+6 clicks | 16000 px | 0.153 | 0.154 | 0.154 | +0.02 | 0.25 | 0.281 |
| SAM ViT-B | 10+20 clicks | 16000 px | 0.471 | 0.342 | 0.396 | -1.36 | 0.58 | 0.417 |
| SAM ViT-B | 3+6 clicks | 12000 px | 0.435 | 0.246 | 0.315 | -2.16 | 0.19 | 0.371 |

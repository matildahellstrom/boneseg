# Plasma cells on SegPC-2021

Settings tuned on 60 training images; scored on the 199 validation images (21 nikon, 178 olympus). Official score: mean over expert cells of the best IoU of any predicted whole cell (extra cells are not penalized). Precision counts unoutlined cells as errors, so it is a lower bound.

| Method | Condition | Official score (95% CI) | Official, nikon | Official, olympus | Precision | Recall | F1 | Dice near cells | Nucleus Dice | Cytoplasm Dice | Count bias | Count ICC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Otsu threshold | no input | 0.417 (0.397–0.438) | 0.294 | 0.432 | 0.006 | 0.270 | 0.011 | 0.717 | – | – | +226.15 | -0.00 |
| boneseg, clicks | 10+20 clicks | 0.550 (0.523–0.576) | 0.652 | 0.537 | 0.038 | 0.502 | 0.071 | 0.862 | – | – | +59.97 | 0.02 |
| boneseg, nucleus + cytoplasm clicks | 10+20 clicks | 0.433 (0.406–0.459) | 0.594 | 0.414 | 0.015 | 0.294 | 0.029 | 0.853 | 0.844 | 0.687 | +90.75 | 0.01 |
| SAM ViT-B, point prompts | 10+20 clicks | 0.475 (0.447–0.502) | 0.580 | 0.462 | 0.009 | 0.377 | 0.017 | 0.746 | – | – | +206.95 | 0.00 |
| micro-SAM ViT-B LM, point prompts | 10+20 clicks | 0.394 (0.359–0.431) | 0.465 | 0.386 | 0.016 | 0.203 | 0.029 | 0.890 | – | – | +59.82 | 0.03 |
| Random forest, clicks (ilastik-style) | 10+20 clicks | 0.366 (0.346–0.386) | 0.388 | 0.364 | 0.005 | 0.216 | 0.010 | 0.807 | – | – | +196.30 | 0.00 |
| boneseg, clicks | 3+6 clicks | 0.439 (0.416–0.463) | 0.514 | 0.431 | 0.023 | 0.328 | 0.044 | 0.793 | – | – | +64.75 | 0.01 |
| boneseg, nucleus + cytoplasm clicks | 3+6 clicks | 0.356 (0.334–0.379) | 0.434 | 0.347 | 0.008 | 0.187 | 0.016 | 0.701 | 0.581 | 0.523 | +107.67 | 0.01 |
| SAM ViT-B, point prompts | 3+6 clicks | 0.389 (0.361–0.420) | 0.465 | 0.381 | 0.012 | 0.249 | 0.023 | 0.680 | – | – | +98.97 | 0.01 |
| micro-SAM ViT-B LM, point prompts | 3+6 clicks | 0.359 (0.327–0.392) | 0.420 | 0.351 | 0.016 | 0.178 | 0.029 | 0.805 | – | – | +52.09 | 0.02 |
| Random forest, clicks (ilastik-style) | 3+6 clicks | 0.327 (0.308–0.345) | 0.326 | 0.327 | 0.005 | 0.163 | 0.010 | 0.751 | – | – | +144.70 | 0.00 |
| boneseg learned model, 5 labelled images | 5 labelled training images | 0.418 (0.391–0.446) | 0.500 | 0.409 | 0.069 | 0.309 | 0.112 | 0.866 | – | – | +17.46 | 0.06 |

## Paired differences in the official score against boneseg, same images and clicks

| Compared with | Condition | Difference (95% CI) | Images where boneseg is better |
|---|---|---|---|
| SAM ViT-B, point prompts | 10+20 clicks | +0.075 (+0.052 to +0.099) | 67% |
| SAM ViT-B, point prompts | 3+6 clicks | +0.050 (+0.023 to +0.075) | 62% |
| micro-SAM ViT-B LM, point prompts | 10+20 clicks | +0.155 (+0.122 to +0.187) | 74% |
| micro-SAM ViT-B LM, point prompts | 3+6 clicks | +0.081 (+0.049 to +0.111) | 67% |
| Random forest, clicks (ilastik-style) | 10+20 clicks | +0.183 (+0.163 to +0.204) | 90% |
| Random forest, clicks (ilastik-style) | 3+6 clicks | +0.113 (+0.096 to +0.131) | 81% |
| boneseg, nucleus + cytoplasm clicks | 10+20 clicks | +0.117 (+0.098 to +0.136) | 80% |
| boneseg, nucleus + cytoplasm clicks | 3+6 clicks | +0.083 (+0.063 to +0.103) | 73% |

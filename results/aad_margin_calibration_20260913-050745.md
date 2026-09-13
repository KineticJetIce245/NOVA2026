# Auditory decision-margin calibration

Generated `2026-09-13T05:00:58-04:00` by `python -B -m scripts.auditory.margin_calibration --out results --contracts 64ch 20ch --histories 5 10 30 60`.

Cache `22e5546fa44abe0f`, 320 trials, held out by story (`held_out_story_folds`, `rep_*` folded into its base): 4 folds, 80-80 held-out trials each.

The controller is the session's: `AttentionController` with `min_switch_windows=3`, `max_age=3`, `attenuation_db=6`, frames every 0.25s, warm-up 2s. Every state in the tables below comes out of `AttentionController.update`; the margin is the only thing that moves between columns.

Coverage is the fraction of published frames that report A or B. `balanced (decided)` is the mean of the two per-class hit rates *within* the decided frames, so a margin that only commits on A reads 0.50 there however good its raw accuracy looks; `majority (decided)` is the null it must beat. `unavail` is warm-up plus the window-length wait, which is not the same failure as `uncertain` and is not counted as coverage.

## 64ch, 5s windows

Pinned against `model.score` on real windows: worst |Δscore| = 4.58e-16 (1e-09 tolerance).

| margin | coverage | accuracy (decided) | balanced (decided) | balanced per story (min–max) | stories above chance | majority (decided) | recall A/B (decided) | recall A/B (all frames) | windows decided | unavail | uncertain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.5242 | 0.6840 | 0.6884 | 0.6680–0.6979 | 4/4 | 0.6500 | 0.6737/0.7030 | 0.3477/0.3796 | 38802/71744 (0.541) | 8960 | 131232 |
| 0.10 | 0.3327 | 0.7248 | 0.7295 | 0.7103–0.7509 | 4/4 | 0.6388 | 0.7127/0.7462 | 0.2294/0.2639 | 24630/71744 (0.343) | 8960 | 187672 |
| 0.15 | 0.2094 | 0.7568 | 0.7612 | 0.7385–0.7799 | 4/4 | 0.6388 | 0.7453/0.7771 | 0.1510/0.1730 | 15506/71744 (0.216) | 8960 | 224004 |
| 0.20 | 0.1294 | 0.7851 | 0.7898 | 0.7643–0.8051 | 4/4 | 0.6380 | 0.7727/0.8070 | 0.0966/0.1113 | 9586/71744 (0.134) | 8960 | 247572 |
| 0.25 | 0.0744 | 0.8074 | 0.8125 | 0.7778–0.8294 | 4/4 | 0.6367 | 0.7938/0.8312 | 0.0569/0.0661 | 5508/71744 (0.077) | 8960 | 263788 |
| 0.30 | 0.0404 | 0.8251 | 0.8307 | 0.7885–0.8488 | 4/4 | 0.6320 | 0.8095/0.8519 | 0.0313/0.0372 | 2991/71744 (0.042) | 8960 | 273804 |
| 0.35 | 0.0204 | 0.8339 | 0.8389 | 0.8065–0.8547 | 4/4 | 0.6226 | 0.8186/0.8592 | 0.0158/0.0195 | 1518/71744 (0.021) | 8960 | 279676 |
| 0.40 | 0.0100 | 0.8539 | 0.8574 | 0.8315–0.8887 | 4/4 | 0.6333 | 0.8440/0.8708 | 0.0081/0.0094 | 745/71744 (0.010) | 8960 | 282740 |
| 0.45 | 0.0039 | 0.8846 | 0.8905 | 0.8493–0.8971 | 4/4 | 0.6503 | 0.8710/0.9100 | 0.0033/0.0036 | 288/71744 (0.004) | 8960 | 284552 |
| 0.50 | 0.0015 | 0.9107 | 0.9139 | 0.8224–1.0000 | 4/4 | 0.6429 | 0.9028/0.9250 | 0.0013/0.0015 | 114/71744 (0.002) | 8960 | 285248 |
| 0.60 | 0.0003 | 1.0000 | 1.0000 | 0.5000–1.0000 | 2/4 | 0.6364 | 1.0000/1.0000 | 0.0003/0.0003 | 22/71744 (0.000) | 8960 | 285608 |

| margin | false selection changes /min | switch delay median (s) | switch delays reported | missed switches | candidate flips | frames decided→undecided | first decision (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 2.427 | n/a | 0 | 0 | 1481 | 5986 | 10.0 |
| 0.10 | 1.790 | n/a | 0 | 0 | 323 | 5949 | 12.0 |
| 0.15 | 1.216 | n/a | 0 | 0 | 67 | 4715 | 16.0 |
| 0.20 | 0.740 | n/a | 0 | 0 | 12 | 3365 | 21.5 |
| 0.25 | 0.435 | n/a | 0 | 0 | 0 | 2200 | 30.0 |
| 0.30 | 0.231 | n/a | 0 | 0 | 0 | 1368 | 39.0 |
| 0.35 | 0.117 | n/a | 0 | 0 | 0 | 774 | 61.0 |
| 0.40 | 0.056 | n/a | 0 | 0 | 0 | 413 | 79.0 |
| 0.45 | 0.019 | n/a | 0 | 0 | 0 | 177 | 98.0 |
| 0.50 | 0.007 | n/a | 0 | 0 | 0 | 69 | 117.0 |
| 0.60 | 0.000 | n/a | 0 | 0 | 0 | 15 | 154.0 |

## 64ch, 10s windows

Pinned against `model.score` on real windows: worst |Δscore| = 3.61e-16 (1e-09 tolerance).

| margin | coverage | accuracy (decided) | balanced (decided) | balanced per story (min–max) | stories above chance | majority (decided) | recall A/B (decided) | recall A/B (all frames) | windows decided | unavail | uncertain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.5119 | 0.7497 | 0.7552 | 0.7293–0.7751 | 4/4 | 0.6453 | 0.7364/0.7740 | 0.3684/0.4136 | 37876/70144 (0.540) | 15360 | 128476 |
| 0.10 | 0.3139 | 0.7991 | 0.8046 | 0.7731–0.8318 | 4/4 | 0.6363 | 0.7846/0.8245 | 0.2374/0.2771 | 23241/70144 (0.331) | 15360 | 186800 |
| 0.15 | 0.1806 | 0.8413 | 0.8474 | 0.7998–0.8767 | 4/4 | 0.6262 | 0.8232/0.8717 | 0.1410/0.1732 | 13370/70144 (0.191) | 15360 | 226088 |
| 0.20 | 0.0938 | 0.8721 | 0.8786 | 0.8355–0.8931 | 4/4 | 0.6180 | 0.8511/0.9061 | 0.0747/0.0956 | 6945/70144 (0.099) | 15360 | 251652 |
| 0.25 | 0.0423 | 0.9069 | 0.9149 | 0.8919–0.9264 | 4/4 | 0.6191 | 0.8813/0.9486 | 0.0350/0.0450 | 3128/70144 (0.045) | 15360 | 266832 |
| 0.30 | 0.0172 | 0.9385 | 0.9442 | 0.9216–0.9541 | 4/4 | 0.6128 | 0.9189/0.9695 | 0.0147/0.0190 | 1271/70144 (0.018) | 15360 | 274224 |
| 0.35 | 0.0060 | 0.9477 | 0.9490 | 0.9182–0.9522 | 4/4 | 0.5818 | 0.9414/0.9565 | 0.0050/0.0070 | 442/70144 (0.006) | 15360 | 277536 |
| 0.40 | 0.0018 | 0.9474 | 0.9494 | 0.9000–1.0000 | 4/4 | 0.5639 | 0.9333/0.9655 | 0.0014/0.0022 | 134/70144 (0.002) | 15360 | 278764 |
| 0.45 | 0.0007 | 0.9811 | 0.9839 | 0.9500–1.0000 | 4/4 | 0.5849 | 0.9677/1.0000 | 0.0006/0.0009 | 53/70144 (0.001) | 15360 | 279084 |
| 0.50 | 0.0002 | 1.0000 | 1.0000 | 0.5000–0.5000 | 0/4 | 0.6471 | 1.0000/1.0000 | 0.0002/0.0002 | 17/70144 (0.000) | 15360 | 279228 |
| 0.60 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/70144 (0.000) | 15360 | 279296 |

| margin | false selection changes /min | switch delay median (s) | switch delays reported | missed switches | candidate flips | frames decided→undecided | first decision (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 1.443 | n/a | 0 | 0 | 223 | 4537 | 15.0 |
| 0.10 | 0.874 | n/a | 0 | 0 | 10 | 3687 | 19.5 |
| 0.15 | 0.481 | n/a | 0 | 0 | 0 | 2572 | 27.5 |
| 0.20 | 0.234 | n/a | 0 | 0 | 0 | 1585 | 38.0 |
| 0.25 | 0.098 | n/a | 0 | 0 | 0 | 851 | 50.0 |
| 0.30 | 0.026 | n/a | 0 | 0 | 0 | 374 | 79.0 |
| 0.35 | 0.011 | n/a | 0 | 0 | 0 | 162 | 112.5 |
| 0.40 | 0.003 | n/a | 0 | 0 | 0 | 46 | 115.0 |
| 0.45 | 0.001 | n/a | 0 | 0 | 0 | 20 | 120.0 |
| 0.50 | 0.000 | n/a | 0 | 0 | 0 | 7 | 129.5 |
| 0.60 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |

## 64ch, 30s windows

Pinned against `model.score` on real windows: worst |Δscore| = 2.08e-16 (1e-09 tolerance).

| margin | coverage | accuracy (decided) | balanced (decided) | balanced per story (min–max) | stories above chance | majority (decided) | recall A/B (decided) | recall A/B (all frames) | windows decided | unavail | uncertain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.4551 | 0.8747 | 0.8802 | 0.8364–0.9097 | 4/4 | 0.6226 | 0.8575/0.9030 | 0.3680/0.4565 | 33703/63744 (0.529) | 40960 | 119600 |
| 0.10 | 0.2294 | 0.9312 | 0.9338 | 0.8867–0.9548 | 4/4 | 0.6064 | 0.9216/0.9460 | 0.1942/0.2515 | 16984/63744 (0.266) | 40960 | 186092 |
| 0.15 | 0.0890 | 0.9724 | 0.9736 | 0.9363–0.9883 | 4/4 | 0.5857 | 0.9664/0.9809 | 0.0763/0.1065 | 6590/63744 (0.103) | 40960 | 227464 |
| 0.20 | 0.0250 | 0.9918 | 0.9928 | 0.9781–1.0000 | 4/4 | 0.5652 | 0.9856/1.0000 | 0.0211/0.0320 | 1848/63744 (0.029) | 40960 | 246336 |
| 0.25 | 0.0053 | 1.0000 | 1.0000 | 1.0000–1.0000 | 4/4 | 0.5179 | 1.0000/1.0000 | 0.0039/0.0081 | 393/63744 (0.006) | 40960 | 252128 |
| 0.30 | 0.0009 | 1.0000 | 1.0000 | 0.0000–1.0000 | 1/4 | 0.6269 | 1.0000/1.0000 | 0.0005/0.0017 | 67/63744 (0.001) | 40960 | 253428 |
| 0.35 | 0.0001 | 1.0000 | 1.0000 | 0.0000–0.5000 | 0/4 | 0.9000 | 1.0000/1.0000 | 0.0000/0.0004 | 10/63744 (0.000) | 40960 | 253656 |
| 0.40 | 0.0000 | 1.0000 | 0.5000 | 0.0000–0.5000 | 0/4 | 1.0000 | n/a/1.0000 | 0.0000/0.0000 | 1/63744 (0.000) | 40960 | 253692 |
| 0.45 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/63744 (0.000) | 40960 | 253696 |
| 0.50 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/63744 (0.000) | 40960 | 253696 |
| 0.60 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/63744 (0.000) | 40960 | 253696 |

| margin | false selection changes /min | switch delay median (s) | switch delays reported | missed switches | candidate flips | frames decided→undecided | first decision (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.397 | n/a | 0 | 0 | 2 | 2040 | 34.0 |
| 0.10 | 0.127 | n/a | 0 | 0 | 0 | 1331 | 49.5 |
| 0.15 | 0.033 | n/a | 0 | 0 | 0 | 707 | 73.0 |
| 0.20 | 0.002 | n/a | 0 | 0 | 0 | 251 | 112.0 |
| 0.25 | 0.000 | n/a | 0 | 0 | 0 | 66 | 125.0 |
| 0.30 | 0.000 | n/a | 0 | 0 | 0 | 12 | 140.0 |
| 0.35 | 0.000 | n/a | 0 | 0 | 0 | 2 | 101.0 |
| 0.40 | 0.000 | n/a | 0 | 0 | 0 | 1 | 144.0 |
| 0.45 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |
| 0.50 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |
| 0.60 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |

## 64ch, 60s windows

Pinned against `model.score` on real windows: worst |Δscore| = 1.25e-16 (1e-09 tolerance).

| margin | coverage | accuracy (decided) | balanced (decided) | balanced per story (min–max) | stories above chance | majority (decided) | recall A/B (decided) | recall A/B (all frames) | windows decided | unavail | uncertain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.3844 | 0.9460 | 0.9483 | 0.9094–0.9723 | 4/4 | 0.5809 | 0.9342/0.9625 | 0.3159/0.4565 | 28477/54144 (0.526) | 79360 | 102024 |
| 0.10 | 0.1600 | 0.9862 | 0.9865 | 0.9660–0.9942 | 4/4 | 0.5784 | 0.9845/0.9885 | 0.1380/0.1963 | 11845/54144 (0.219) | 79360 | 168148 |
| 0.15 | 0.0406 | 1.0000 | 1.0000 | 1.0000–1.0000 | 4/4 | 0.5440 | 1.0000/1.0000 | 0.0334/0.0545 | 3000/54144 (0.055) | 79360 | 203332 |
| 0.20 | 0.0050 | 1.0000 | 1.0000 | 1.0000–1.0000 | 4/4 | 0.7849 | 1.0000/1.0000 | 0.0016/0.0117 | 372/54144 (0.007) | 79360 | 213808 |
| 0.25 | 0.0003 | 1.0000 | 0.5000 | 0.0000–0.5000 | 0/4 | 1.0000 | n/a/1.0000 | 0.0000/0.0009 | 22/54144 (0.000) | 79360 | 215208 |
| 0.30 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/54144 (0.000) | 79360 | 215296 |
| 0.35 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/54144 (0.000) | 79360 | 215296 |
| 0.40 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/54144 (0.000) | 79360 | 215296 |
| 0.45 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/54144 (0.000) | 79360 | 215296 |
| 0.50 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/54144 (0.000) | 79360 | 215296 |
| 0.60 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/54144 (0.000) | 79360 | 215296 |

| margin | false selection changes /min | switch delay median (s) | switch delays reported | missed switches | candidate flips | frames decided→undecided | first decision (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.109 | n/a | 0 | 0 | 1 | 1040 | 64.0 |
| 0.10 | 0.017 | n/a | 0 | 0 | 0 | 621 | 79.0 |
| 0.15 | 0.000 | n/a | 0 | 0 | 0 | 244 | 118.0 |
| 0.20 | 0.000 | n/a | 0 | 0 | 0 | 38 | 131.5 |
| 0.25 | 0.000 | n/a | 0 | 0 | 0 | 4 | 245.0 |
| 0.30 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |
| 0.35 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |
| 0.40 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |
| 0.45 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |
| 0.50 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |
| 0.60 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |

## 20ch, 5s windows

Pinned against `model.score` on real windows: worst |Δscore| = 2.64e-16 (1e-09 tolerance).

| margin | coverage | accuracy (decided) | balanced (decided) | balanced per story (min–max) | stories above chance | majority (decided) | recall A/B (decided) | recall A/B (all frames) | windows decided | unavail | uncertain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.5155 | 0.6481 | 0.6514 | 0.6361–0.6579 | 4/4 | 0.6449 | 0.6403/0.6625 | 0.3224/0.3570 | 38165/71744 (0.532) | 8960 | 133792 |
| 0.10 | 0.3189 | 0.6806 | 0.6856 | 0.6610–0.7004 | 4/4 | 0.6415 | 0.6679/0.7033 | 0.2069/0.2367 | 23604/71744 (0.329) | 8960 | 191736 |
| 0.15 | 0.1968 | 0.7059 | 0.7105 | 0.6801–0.7249 | 4/4 | 0.6383 | 0.6937/0.7273 | 0.1320/0.1524 | 14568/71744 (0.203) | 8960 | 227696 |
| 0.20 | 0.1155 | 0.7358 | 0.7407 | 0.7108–0.7550 | 4/4 | 0.6382 | 0.7231/0.7582 | 0.0807/0.0932 | 8550/71744 (0.119) | 8960 | 251676 |
| 0.25 | 0.0663 | 0.7545 | 0.7609 | 0.7144–0.7774 | 4/4 | 0.6400 | 0.7379/0.7838 | 0.0474/0.0551 | 4914/71744 (0.068) | 8960 | 266164 |
| 0.30 | 0.0345 | 0.7768 | 0.7836 | 0.7517–0.7978 | 4/4 | 0.6487 | 0.7608/0.8065 | 0.0258/0.0288 | 2563/71744 (0.036) | 8960 | 275516 |
| 0.35 | 0.0169 | 0.7888 | 0.7965 | 0.7652–0.8172 | 4/4 | 0.6458 | 0.7699/0.8231 | 0.0127/0.0145 | 1258/71744 (0.018) | 8960 | 280716 |
| 0.40 | 0.0074 | 0.8257 | 0.8418 | 0.8246–0.8479 | 4/4 | 0.6532 | 0.7893/0.8942 | 0.0058/0.0068 | 549/71744 (0.008) | 8960 | 283516 |
| 0.45 | 0.0031 | 0.8097 | 0.8307 | 0.6970–0.8850 | 4/4 | 0.6637 | 0.7667/0.8947 | 0.0024/0.0027 | 228/71744 (0.003) | 8960 | 284792 |
| 0.50 | 0.0012 | 0.8488 | 0.8816 | 0.4118–0.9333 | 3/4 | 0.7093 | 0.8033/0.9600 | 0.0010/0.0010 | 86/71744 (0.001) | 8960 | 285352 |
| 0.60 | 0.0001 | 0.8333 | 0.9000 | 0.0000–0.5000 | 0/4 | 0.8333 | 0.8000/1.0000 | 0.0001/0.0000 | 6/71744 (0.000) | 8960 | 285672 |

| margin | false selection changes /min | switch delay median (s) | switch delays reported | missed switches | candidate flips | frames decided→undecided | first decision (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 2.611 | n/a | 0 | 0 | 1570 | 6049 | 10.0 |
| 0.10 | 1.958 | n/a | 0 | 0 | 363 | 5905 | 12.0 |
| 0.15 | 1.327 | n/a | 0 | 0 | 75 | 4617 | 15.0 |
| 0.20 | 0.802 | n/a | 0 | 0 | 9 | 3210 | 22.0 |
| 0.25 | 0.468 | n/a | 0 | 0 | 1 | 2070 | 34.0 |
| 0.30 | 0.242 | n/a | 0 | 0 | 0 | 1219 | 47.0 |
| 0.35 | 0.115 | n/a | 0 | 0 | 0 | 661 | 65.0 |
| 0.40 | 0.046 | n/a | 0 | 0 | 0 | 328 | 83.0 |
| 0.45 | 0.024 | n/a | 0 | 0 | 0 | 141 | 112.5 |
| 0.50 | 0.007 | n/a | 0 | 0 | 0 | 55 | 118.0 |
| 0.60 | 0.001 | n/a | 0 | 0 | 0 | 6 | 148.5 |

## 20ch, 10s windows

Pinned against `model.score` on real windows: worst |Δscore| = 1.67e-16 (1e-09 tolerance).

| margin | coverage | accuracy (decided) | balanced (decided) | balanced per story (min–max) | stories above chance | majority (decided) | recall A/B (decided) | recall A/B (all frames) | windows decided | unavail | uncertain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.4872 | 0.7065 | 0.7108 | 0.6841–0.7326 | 4/4 | 0.6428 | 0.6957/0.7258 | 0.3300/0.3718 | 36068/70144 (0.514) | 15360 | 135740 |
| 0.10 | 0.2890 | 0.7548 | 0.7591 | 0.7261–0.7720 | 4/4 | 0.6378 | 0.7435/0.7747 | 0.2075/0.2387 | 21398/70144 (0.305) | 15360 | 194152 |
| 0.15 | 0.1570 | 0.7865 | 0.7921 | 0.7595–0.8107 | 4/4 | 0.6392 | 0.7721/0.8121 | 0.1173/0.1354 | 11623/70144 (0.166) | 15360 | 233040 |
| 0.20 | 0.0775 | 0.8215 | 0.8298 | 0.7977–0.8449 | 4/4 | 0.6361 | 0.7992/0.8604 | 0.0597/0.0714 | 5741/70144 (0.082) | 15360 | 256464 |
| 0.25 | 0.0321 | 0.8419 | 0.8535 | 0.8391–0.8682 | 4/4 | 0.6304 | 0.8089/0.8982 | 0.0248/0.0314 | 2382/70144 (0.034) | 15360 | 269836 |
| 0.30 | 0.0123 | 0.8609 | 0.8785 | 0.8270–0.9356 | 4/4 | 0.6501 | 0.8200/0.9369 | 0.0099/0.0119 | 912/70144 (0.013) | 15360 | 275672 |
| 0.35 | 0.0041 | 0.8845 | 0.9130 | 0.8250–0.9861 | 4/4 | 0.6865 | 0.8365/0.9895 | 0.0036/0.0038 | 304/70144 (0.004) | 15360 | 278084 |
| 0.40 | 0.0012 | 0.9205 | 0.9300 | 0.9048–1.0000 | 4/4 | 0.5682 | 0.8600/1.0000 | 0.0009/0.0015 | 88/70144 (0.001) | 15360 | 278944 |
| 0.45 | 0.0003 | 0.8421 | 0.8500 | 0.0000–1.0000 | 2/4 | 0.5263 | 0.7000/1.0000 | 0.0001/0.0004 | 19/70144 (0.000) | 15360 | 279220 |
| 0.50 | 0.0000 | 1.0000 | 0.5000 | 0.0000–0.5000 | 0/4 | 1.0000 | n/a/1.0000 | 0.0000/0.0001 | 3/70144 (0.000) | 15360 | 279284 |
| 0.60 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/70144 (0.000) | 15360 | 279296 |

| margin | false selection changes /min | switch delay median (s) | switch delays reported | missed switches | candidate flips | frames decided→undecided | first decision (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 1.543 | n/a | 0 | 0 | 218 | 4669 | 15.0 |
| 0.10 | 0.983 | n/a | 0 | 0 | 14 | 3649 | 20.0 |
| 0.15 | 0.523 | n/a | 0 | 0 | 0 | 2394 | 31.0 |
| 0.20 | 0.254 | n/a | 0 | 0 | 0 | 1456 | 46.0 |
| 0.25 | 0.106 | n/a | 0 | 0 | 0 | 688 | 67.0 |
| 0.30 | 0.039 | n/a | 0 | 0 | 0 | 291 | 90.5 |
| 0.35 | 0.013 | n/a | 0 | 0 | 0 | 109 | 118.0 |
| 0.40 | 0.003 | n/a | 0 | 0 | 0 | 35 | 117.0 |
| 0.45 | 0.001 | n/a | 0 | 0 | 0 | 11 | 118.0 |
| 0.50 | 0.000 | n/a | 0 | 0 | 0 | 2 | 355.5 |
| 0.60 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |

## 20ch, 30s windows

Pinned against `model.score` on real windows: worst |Δscore| = 1.39e-16 (1e-09 tolerance).

| margin | coverage | accuracy (decided) | balanced (decided) | balanced per story (min–max) | stories above chance | majority (decided) | recall A/B (decided) | recall A/B (all frames) | windows decided | unavail | uncertain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.4186 | 0.8292 | 0.8353 | 0.8096–0.8626 | 4/4 | 0.6294 | 0.8117/0.8589 | 0.3239/0.3922 | 31004/63744 (0.486) | 40960 | 130348 |
| 0.10 | 0.1819 | 0.8920 | 0.8960 | 0.8414–0.9185 | 4/4 | 0.6224 | 0.8798/0.9123 | 0.1509/0.1845 | 13460/63744 (0.211) | 40960 | 200084 |
| 0.15 | 0.0583 | 0.9539 | 0.9588 | 0.9333–0.9928 | 4/4 | 0.6075 | 0.9360/0.9816 | 0.0502/0.0661 | 4310/63744 (0.068) | 40960 | 236512 |
| 0.20 | 0.0154 | 0.9761 | 0.9797 | 0.9508–1.0000 | 4/4 | 0.5889 | 0.9595/1.0000 | 0.0131/0.0186 | 1134/63744 (0.018) | 40960 | 249172 |
| 0.25 | 0.0026 | 1.0000 | 1.0000 | 1.0000–1.0000 | 4/4 | 0.5258 | 1.0000/1.0000 | 0.0019/0.0041 | 195/63744 (0.003) | 40960 | 252920 |
| 0.30 | 0.0005 | 1.0000 | 1.0000 | 0.5000–0.5000 | 0/4 | 0.7941 | 1.0000/1.0000 | 0.0001/0.0011 | 34/63744 (0.001) | 40960 | 253560 |
| 0.35 | 0.0001 | 1.0000 | 0.5000 | 0.0000–0.5000 | 0/4 | 1.0000 | n/a/1.0000 | 0.0000/0.0003 | 8/63744 (0.000) | 40960 | 253664 |
| 0.40 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/63744 (0.000) | 40960 | 253696 |
| 0.45 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/63744 (0.000) | 40960 | 253696 |
| 0.50 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/63744 (0.000) | 40960 | 253696 |
| 0.60 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/63744 (0.000) | 40960 | 253696 |

| margin | false selection changes /min | switch delay median (s) | switch delays reported | missed switches | candidate flips | frames decided→undecided | first decision (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.480 | n/a | 0 | 0 | 1 | 2108 | 37.0 |
| 0.10 | 0.167 | n/a | 0 | 0 | 0 | 1252 | 53.0 |
| 0.15 | 0.034 | n/a | 0 | 0 | 0 | 510 | 75.5 |
| 0.20 | 0.002 | n/a | 0 | 0 | 0 | 165 | 110.0 |
| 0.25 | 0.000 | n/a | 0 | 0 | 0 | 30 | 159.0 |
| 0.30 | 0.000 | n/a | 0 | 0 | 0 | 9 | 121.0 |
| 0.35 | 0.000 | n/a | 0 | 0 | 0 | 1 | 327.0 |
| 0.40 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |
| 0.45 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |
| 0.50 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |
| 0.60 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |

## 20ch, 60s windows

Pinned against `model.score` on real windows: worst |Δscore| = 8.33e-17 (1e-09 tolerance).

| margin | coverage | accuracy (decided) | balanced (decided) | balanced per story (min–max) | stories above chance | majority (decided) | recall A/B (decided) | recall A/B (all frames) | windows decided | unavail | uncertain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.3337 | 0.9203 | 0.9232 | 0.8906–0.9472 | 4/4 | 0.5938 | 0.9076/0.9389 | 0.2724/0.3746 | 24731/54144 (0.457) | 79360 | 116968 |
| 0.10 | 0.1058 | 0.9784 | 0.9796 | 0.9615–0.9908 | 4/4 | 0.5846 | 0.9728/0.9864 | 0.0911/0.1276 | 7826/54144 (0.145) | 79360 | 184128 |
| 0.15 | 0.0222 | 0.9951 | 0.9958 | 0.9873–1.0000 | 4/4 | 0.5791 | 0.9916/1.0000 | 0.0193/0.0275 | 1641/54144 (0.030) | 79360 | 208748 |
| 0.20 | 0.0018 | 1.0000 | 1.0000 | 0.5000–1.0000 | 3/4 | 0.6838 | 1.0000/1.0000 | 0.0009/0.0037 | 136/54144 (0.003) | 79360 | 214752 |
| 0.25 | 0.0000 | 1.0000 | 0.5000 | 0.0000–0.5000 | 0/4 | 1.0000 | n/a/1.0000 | 0.0000/0.0000 | 1/54144 (0.000) | 79360 | 215292 |
| 0.30 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/54144 (0.000) | 79360 | 215296 |
| 0.35 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/54144 (0.000) | 79360 | 215296 |
| 0.40 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/54144 (0.000) | 79360 | 215296 |
| 0.45 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/54144 (0.000) | 79360 | 215296 |
| 0.50 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/54144 (0.000) | 79360 | 215296 |
| 0.60 | 0.0000 | n/a | 0.0000 | 0.0000–0.0000 | 0/4 | n/a | n/a/n/a | 0.0000/0.0000 | 0/54144 (0.000) | 79360 | 215296 |

| margin | false selection changes /min | switch delay median (s) | switch delays reported | missed switches | candidate flips | frames decided→undecided | first decision (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.158 | n/a | 0 | 0 | 0 | 1144 | 65.0 |
| 0.10 | 0.023 | n/a | 0 | 0 | 0 | 513 | 89.0 |
| 0.15 | 0.002 | n/a | 0 | 0 | 0 | 151 | 116.0 |
| 0.20 | 0.000 | n/a | 0 | 0 | 0 | 12 | 235.5 |
| 0.25 | 0.000 | n/a | 0 | 0 | 0 | 1 | 330.0 |
| 0.30 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |
| 0.35 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |
| 0.40 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |
| 0.45 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |
| 0.50 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |
| 0.60 | 0.000 | n/a | 0 | 0 | 0 | 0 | n/a |

## Recommendation

Rule, fixed before the curve was read: a point qualifies if it commits inside every held-out story with balanced accuracy above chance in each, and if its accuracy on decided frames beats the majority rate of its own decided subset; among the qualifying points of a window length, keep the best coverage and then the highest balanced accuracy within 0.02 of it. Coverage is the price; the balanced accuracy of the committed frames is what is bought.

Across every window length measured at 64ch, the rule picks margin 0.05 at 60s windows: coverage 0.384, accuracy 0.9460 and balanced accuracy 0.9483 on the decided frames (per story 0.9094 to 0.9723, 4/4 above chance), against a majority rate of 0.5809 inside its own decided subset. Costs: per-class recall over all frames 0.3159/0.4565 (an abstention is a miss), first decision after a median of 64.0s, 79360 frames unavailable rather than uncertain, and 0.109 false selection changes per minute. `models/auditory_kuleuven.npz` is a 5s model, so what can be run today is margin 0.05 at 5s: coverage 0.524, accuracy 0.6840, balanced accuracy 0.6884 on decided frames (per story 0.6680 to 0.6979, 4/4 above chance), first decision 10.0s, 2.427 false selection changes per minute. The longer window buys 0.260 balanced accuracy and 2.318 fewer false changes per minute at 0.140 coverage, and it costs a model fitted at that length. For comparison, the current default margin 0.5 at 5s covers 0.0015 of frames with 448 decided frames in the whole held-out corpus.

The rule's answer at each window length, so the choice can be moved without re-running the sweep:

| window (s) | margin | coverage | accuracy (decided) | balanced (decided) | per story | first decision (s) | false changes /min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.05 | 0.5242 | 0.6840 | 0.6884 | 0.6680–0.6979 (4/4) | 10.0 | 2.427 |
| 10 | 0.05 | 0.5119 | 0.7497 | 0.7552 | 0.7293–0.7751 (4/4) | 15.0 | 1.443 |
| 30 | 0.05 | 0.4551 | 0.8747 | 0.8802 | 0.8364–0.9097 (4/4) | 34.0 | 0.397 |
| 60 | 0.05 | 0.3844 | 0.9460 | 0.9483 | 0.9094–0.9723 (4/4) | 64.0 | 0.109 |

## What these numbers do not say

- They are window- and frame-level, from one dataset (KU Leuven replay) with the authors' own stimuli. They say nothing about the ANT test set and nothing about a live participant.
- The unit of every count is a *window* or a *frame*, never a subject. The story folds are the independent unit (four of them), so the differences between neighbouring margins are well inside the variation between stories and must not be read as a ranking.
- A margin tuned here is a demo operating point, not evidence of decoding quality. The decoder's own numbers are step 5's.
- Frames overlap their neighbours (one-second hop on a 5s window), so consecutive frames are far from independent and the effective sample size is much smaller than the frame count.
- Latency is modelled as zero: the controller is updated with `now` equal to the window's evidence end. A real offload queue adds at most one window.
- `unavailable` frames (warm-up and the wait for the first full window) are reported separately and are not `uncertain`; a longer window buys accuracy at the price of a longer silent start.

## Plumbing (for the step that runs the demo)

The margin is not a constant to edit; it is a run-policy choice, and the run
record must be able to state it (plan section 3.17 item 5, decision D-26).

1. `RunPolicy` (`src/nova2026/auditory/session.py`) gains
   `margin: float = MIN_MARGIN` (imported from `nova2026.auditory.config`), so an
   existing caller that passes no margin keeps `0.5` exactly, and `to_dict()`
   records the value under `"margin"`.
2. `AttentionSession.__init__` builds `AttentionController(margin=policy.margin,
   max_age=..., min_switch_windows=..., attenuation_db=...)` where it currently
   builds `AttentionController()`; the `controller=` argument keeps winning when
   one is supplied.
3. The demo CLI (`scripts/auditory_ui/demorun.py`, `demo.py`) gains `--margin`
   with default `None` meaning "whatever the policy default is", and the run
   record writes the effective value beside `check_channels`.
4. **Window length is not a policy field and cannot be one.** The session reads
   `window_seconds` from the decoder's contract, so a longer window needs a
   model fitted at that length. `models/auditory_kuleuven.npz` is a 5 s model;
   any 30 s/60 s operating point requires a new model, which is a separate
   deliverable and not this task's to write.
5. The proof that the default is unchanged is in
   `scripts/auditory/tests/test_margin_calibration.py`: `MIN_MARGIN` is still
   `0.5`, `AttentionController()` still carries `margin == 0.5`, and the
   calibrated constant is a different name that nothing defaults to.


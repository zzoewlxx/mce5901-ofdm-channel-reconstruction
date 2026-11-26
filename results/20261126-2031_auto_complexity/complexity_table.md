Per estimate (one vector); CPU time with 1 BLAS thread(s); '-' = minimal equals as implemented.

| Setting | Pilots | Method | Complex MACs | min. | Real MACs | Params (real) | min. | Batch us/vector | Latency us |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| freq64 | 16 | Linear interpolation | 1,024 | 48 | 4,096 | 1,024 | 96 | 1.58 | 3.6 |
| freq64 | 16 | Cubic spline | 1,024 | 137 | 4,096 | 1,024 | 274 | 1.4 | 11.6 |
| freq64 | 16 | LMMSE, precomputed G | 1,024 | 768 | 4,096 | 2,176 | 1,632 | 2.55 | 19.7 |
| freq64 | 16 | LMMSE, per-sample SNR | 1,288 | 1,032 | 5,152 | 2,705 | 2,193 | 3.48 | 76.0 |
| freq64 | 16 | MLP 2x128 | 8,224 | - | 32,896 | 33,248 | - | 13.6 | 101 |
| freq64 | 16 | LMMSE + residual MLP 2x128 | 9,512 | 9,256 | 38,048 | 35,953 | 35,441 | 18.5 | 200 |
| freq64 | 16 | GMM K=16 (mix_estimate) | 21,248 | 14,720 | 84,992 | 131,088 | 28,688 | 61.2 | 5,242 |
| freq64 | 16 | GMM K=16, precomputed | 16,640 | 14,720 | 66,560 | 32,784 | 28,688 | 28.0 | 666 |
| freq64 | 8 | Linear interpolation | 512 | 56 | 2,048 | 512 | 112 | 0.932 | 10.3 |
| freq64 | 8 | Cubic spline | 512 | 129 | 2,048 | 512 | 258 | 1.09 | 10.4 |
| freq64 | 8 | LMMSE, precomputed G | 512 | 448 | 2,048 | 1,152 | 1,008 | 2 | 16.7 |
| freq64 | 8 | LMMSE, per-sample SNR | 580 | 516 | 2,320 | 1,289 | 1,161 | 2.52 | 54.9 |
| freq64 | 8 | MLP 2x128 | 8,224 | - | 32,896 | 33,264 | - | 14.7 | 96.5 |
| freq64 | 8 | LMMSE + residual MLP 2x128 | 8,804 | 8,740 | 35,216 | 34,553 | 34,425 | 20.1 | 195 |
| freq64 | 8 | GMM K=16 (mix_estimate) | 9,856 | 7,872 | 39,424 | 131,088 | 15,376 | 41.8 | 2,711 |
| freq64 | 8 | GMM K=16, precomputed | 8,320 | 7,872 | 33,280 | 16,400 | 15,376 | 26.9 | 759 |
| freq64 | 4 | Linear interpolation | 256 | 60 | 1,024 | 256 | 120 | 1.07 | 9.5 |
| freq64 | 4 | Cubic spline | 256 | 120 | 1,024 | 256 | 240 | 0.897 | 2.7 |
| freq64 | 4 | LMMSE, precomputed G | 256 | 240 | 1,024 | 640 | 600 | 2.14 | 17.2 |
| freq64 | 4 | LMMSE, per-sample SNR | 274 | 258 | 1,096 | 677 | 645 | 2.29 | 50.6 |
| freq64 | 4 | MLP 2x128 | 8,224 | - | 32,896 | 33,272 | - | 14.9 | 28.2 |
| freq64 | 4 | LMMSE + residual MLP 2x128 | 8,498 | 8,482 | 33,992 | 33,949 | 33,917 | 18.5 | 65.9 |
| freq64 | 4 | GMM K=16 (mix_estimate) | 4,928 | 4,064 | 19,712 | 131,088 | 7,952 | 22.5 | 1,179 |
| freq64 | 4 | GMM K=16, precomputed | 4,160 | 4,064 | 16,640 | 8,208 | 7,952 | 10.3 | 257 |
| spacefreq | 128 | Joint LMMSE, precomputed G | 65,536 | 49,152 | 262,144 | 132,096 | 99,072 | 20.9 | 23.0 |
| spacefreq | 128 | Joint LMMSE, per-sample SNR | 81,984 | 65,600 | 327,936 | 164,993 | 132,225 | 26.8 | 65.6 |
| spacefreq | 128 | Joint MLP 2x512 | 196,736 | - | 786,944 | 788,736 | - | 69.5 | 248 |
| spacefreq | 128 | Joint LMMSE + residual MLP 2x512 | 278,720 | 262,336 | 1,114,880 | 953,729 | 920,961 | 189 | 990 |
| spacefreq | 128 | GMM K=32 (mix_estimate) | 2,633,728 | 1,841,152 | 10,534,912 | 16,777,248 | 3,670,048 | 1,691 | 439,300 |
| spacefreq | 128 | GMM K=32, precomputed | 2,101,248 | 1,841,152 | 8,404,992 | 4,194,336 | 3,670,048 | 1,040 | 5,136 |
| spacefreq | 128 | GMM K=64 (mix_estimate) | 5,267,456 | 3,682,304 | 21,069,824 | 33,554,496 | 7,340,096 | 1,800 | 436,600 |
| spacefreq | 128 | GMM K=64, precomputed | 4,202,496 | 3,682,304 | 16,809,984 | 8,388,672 | 7,340,096 | 1,225 | 5,570 |
| spacefreq | 32 | Joint LMMSE, precomputed G | 16,384 | 15,360 | 65,536 | 33,792 | 31,680 | 11.2 | 12.2 |
| spacefreq | 32 | Joint LMMSE, per-sample SNR | 17,424 | 16,400 | 69,696 | 35,873 | 33,825 | 11.6 | 29.5 |
| spacefreq | 32 | Joint MLP 2x512 | 196,736 | - | 786,944 | 788,928 | - | 69.4 | 261 |
| spacefreq | 32 | Joint LMMSE + residual MLP 2x512 | 214,160 | 213,136 | 856,640 | 824,801 | 822,753 | 100 | 332 |
| spacefreq | 32 | GMM K=32 (mix_estimate) | 566,272 | 509,440 | 2,265,088 | 16,777,248 | 1,015,840 | 424 | 22,350 |
| spacefreq | 32 | GMM K=32, precomputed | 525,312 | 509,440 | 2,101,248 | 1,048,608 | 1,015,840 | 265 | 1,312 |
| spacefreq | 32 | GMM K=64 (mix_estimate) | 1,132,544 | 1,018,880 | 4,530,176 | 33,554,496 | 2,031,680 | 847 | 46,950 |
| spacefreq | 32 | GMM K=64, precomputed | 1,050,624 | 1,018,880 | 4,202,496 | 2,097,216 | 2,031,680 | 889 | 4,890 |

- freq64/16 GMM K=16 (mix_estimate): mix_estimate also spends about 458,752 complex MACs per CALL rebuilding its inverses (included in the latency; spread over the chunk in the batch figure).
- freq64/8 GMM K=16 (mix_estimate): mix_estimate also spends about 90,112 complex MACs per CALL rebuilding its inverses (included in the latency; spread over the chunk in the batch figure).
- freq64/4 GMM K=16 (mix_estimate): mix_estimate also spends about 19,456 complex MACs per CALL rebuilding its inverses (included in the latency; spread over the chunk in the batch figure).
- spacefreq/128 GMM K=32 (mix_estimate): mix_estimate also spends about 469,762,048 complex MACs per CALL rebuilding its inverses (included in the latency; spread over the chunk in the batch figure).
- spacefreq/128 GMM K=64 (mix_estimate): mix_estimate also spends about 939,524,096 complex MACs per CALL rebuilding its inverses (included in the latency; spread over the chunk in the batch figure).
- spacefreq/32 GMM K=32 (mix_estimate): mix_estimate also spends about 19,922,944 complex MACs per CALL rebuilding its inverses (included in the latency; spread over the chunk in the batch figure).
- spacefreq/32 GMM K=64 (mix_estimate): mix_estimate also spends about 39,845,888 complex MACs per CALL rebuilding its inverses (included in the latency; spread over the chunk in the batch figure).

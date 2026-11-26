Per estimate (one vector); CPU time with 1 BLAS thread(s); '-' = minimal equals as implemented.

| Setting | Pilots | Method | Complex MACs | min. | Real MACs | Params (real) | min. | Batch us/vector | Latency us |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| freq64 | 16 | Linear interpolation | 1,024 | 48 | 4,096 | 1,024 | 96 | 0.372 | 2 |
| freq64 | 16 | Cubic spline | 1,024 | 137 | 4,096 | 1,024 | 274 | 0.377 | 2 |
| freq64 | 16 | LMMSE, precomputed G | 1,024 | 768 | 4,096 | 2,176 | 1,632 | 0.719 | 2.8 |
| freq64 | 16 | LMMSE, per-sample SNR | 1,288 | 1,032 | 5,152 | 2,705 | 2,193 | 0.941 | 34.4 |
| freq64 | 16 | MLP 2x128 | 8,224 | - | 32,896 | 33,248 | - | 11.3 | 58.9 |
| freq64 | 16 | LMMSE + residual MLP 2x128 | 9,512 | 9,256 | 38,048 | 35,953 | 35,441 | 15.5 | 108 |
| freq64 | 16 | GMM K=16 (mix_estimate) | 21,248 | 14,720 | 84,992 | 131,088 | 28,688 | 45.0 | 3,232 |
| freq64 | 16 | GMM K=16, precomputed | 16,640 | 14,720 | 66,560 | 32,784 | 28,688 | 29.7 | 415 |
| freq64 | 8 | Linear interpolation | 512 | 56 | 2,048 | 512 | 112 | 0.77 | 9.8 |
| freq64 | 8 | Cubic spline | 512 | 129 | 2,048 | 512 | 258 | 0.829 | 10.2 |
| freq64 | 8 | LMMSE, precomputed G | 512 | 448 | 2,048 | 1,152 | 1,008 | 1.69 | 14.7 |
| freq64 | 8 | LMMSE, per-sample SNR | 580 | 516 | 2,320 | 1,289 | 1,161 | 1.97 | 32.7 |
| freq64 | 8 | MLP 2x128 | 8,224 | - | 32,896 | 33,264 | - | 11.0 | 57.5 |
| freq64 | 8 | LMMSE + residual MLP 2x128 | 8,804 | 8,740 | 35,216 | 34,553 | 34,425 | 15.6 | 139 |
| freq64 | 8 | GMM K=16 (mix_estimate) | 9,856 | 7,872 | 39,424 | 131,088 | 15,376 | 35.3 | 3,864 |
| freq64 | 8 | GMM K=16, precomputed | 8,320 | 7,872 | 33,280 | 16,400 | 15,376 | 21.9 | 605 |
| freq64 | 4 | Linear interpolation | 256 | 60 | 1,024 | 256 | 120 | 0.73 | 5.2 |
| freq64 | 4 | Cubic spline | 256 | 120 | 1,024 | 256 | 240 | 0.87 | 5.2 |
| freq64 | 4 | LMMSE, precomputed G | 256 | 240 | 1,024 | 640 | 600 | 1.49 | 14.8 |
| freq64 | 4 | LMMSE, per-sample SNR | 274 | 258 | 1,096 | 677 | 645 | 1.61 | 33.3 |
| freq64 | 4 | MLP 2x128 | 8,224 | - | 32,896 | 33,272 | - | 11.2 | 56.2 |
| freq64 | 4 | LMMSE + residual MLP 2x128 | 8,498 | 8,482 | 33,992 | 33,949 | 33,917 | 15.6 | 131 |
| freq64 | 4 | GMM K=16 (mix_estimate) | 4,928 | 4,064 | 19,712 | 131,088 | 7,952 | 30.0 | 1,952 |
| freq64 | 4 | GMM K=16, precomputed | 4,160 | 4,064 | 16,640 | 8,208 | 7,952 | 16.1 | 519 |
| spacefreq | 128 | Joint LMMSE, precomputed G | 65,536 | 49,152 | 262,144 | 132,096 | 99,072 | 39.9 | 42.5 |
| spacefreq | 128 | Joint LMMSE, per-sample SNR | 81,984 | 65,600 | 327,936 | 164,993 | 132,225 | 50.5 | 121 |
| spacefreq | 128 | Joint MLP 2x512 | 196,736 | - | 786,944 | 788,736 | - | 128 | 415 |
| spacefreq | 128 | Joint LMMSE + residual MLP 2x512 | 278,720 | 262,336 | 1,114,880 | 953,729 | 920,961 | 195 | 679 |
| spacefreq | 128 | GMM K=32 (mix_estimate) | 2,633,728 | 1,841,152 | 10,534,912 | 16,777,248 | 3,670,048 | 1,710 | 419,400 |
| spacefreq | 128 | GMM K=32, precomputed | 2,101,248 | 1,841,152 | 8,404,992 | 4,194,336 | 3,670,048 | 1,235 | 3,821 |
| spacefreq | 128 | GMM K=64 (mix_estimate) | 5,267,456 | 3,682,304 | 21,069,824 | 33,554,496 | 7,340,096 | 3,395 | 819,000 |
| spacefreq | 128 | GMM K=64, precomputed | 4,202,496 | 3,682,304 | 16,809,984 | 8,388,672 | 7,340,096 | 2,376 | 4,155 |
| spacefreq | 32 | Joint LMMSE, precomputed G | 16,384 | 15,360 | 65,536 | 33,792 | 31,680 | 6.76 | 5.5 |
| spacefreq | 32 | Joint LMMSE, per-sample SNR | 17,424 | 16,400 | 69,696 | 35,873 | 33,825 | 7.87 | 13.4 |
| spacefreq | 32 | Joint MLP 2x512 | 196,736 | - | 786,944 | 788,928 | - | 41.8 | 135 |
| spacefreq | 32 | Joint LMMSE + residual MLP 2x512 | 214,160 | 213,136 | 856,640 | 824,801 | 822,753 | 59.5 | 158 |
| spacefreq | 32 | GMM K=32 (mix_estimate) | 566,272 | 509,440 | 2,265,088 | 16,777,248 | 1,015,840 | 253 | 33,700 |
| spacefreq | 32 | GMM K=32, precomputed | 525,312 | 509,440 | 2,101,248 | 1,048,608 | 1,015,840 | 332 | 413 |
| spacefreq | 32 | GMM K=64 (mix_estimate) | 1,132,544 | 1,018,880 | 4,530,176 | 33,554,496 | 2,031,680 | 1,098 | 130,400 |
| spacefreq | 32 | GMM K=64, precomputed | 1,050,624 | 1,018,880 | 4,202,496 | 2,097,216 | 2,031,680 | 426 | 1,592 |

- freq64/16 GMM K=16 (mix_estimate): mix_estimate also spends about 458,752 complex MACs per CALL rebuilding its inverses (included in the latency; spread over the chunk in the batch figure).
- freq64/8 GMM K=16 (mix_estimate): mix_estimate also spends about 90,112 complex MACs per CALL rebuilding its inverses (included in the latency; spread over the chunk in the batch figure).
- freq64/4 GMM K=16 (mix_estimate): mix_estimate also spends about 19,456 complex MACs per CALL rebuilding its inverses (included in the latency; spread over the chunk in the batch figure).
- spacefreq/128 GMM K=32 (mix_estimate): mix_estimate also spends about 469,762,048 complex MACs per CALL rebuilding its inverses (included in the latency; spread over the chunk in the batch figure).
- spacefreq/128 GMM K=64 (mix_estimate): mix_estimate also spends about 939,524,096 complex MACs per CALL rebuilding its inverses (included in the latency; spread over the chunk in the batch figure).
- spacefreq/32 GMM K=32 (mix_estimate): mix_estimate also spends about 19,922,944 complex MACs per CALL rebuilding its inverses (included in the latency; spread over the chunk in the batch figure).
- spacefreq/32 GMM K=64 (mix_estimate): mix_estimate also spends about 39,845,888 complex MACs per CALL rebuilding its inverses (included in the latency; spread over the chunk in the batch figure).

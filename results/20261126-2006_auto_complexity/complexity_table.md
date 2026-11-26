Per estimate (one vector); CPU time with 1 BLAS thread(s); '-' = minimal equals as implemented.

| Setting | Pilots | Method | Complex MACs | min. | Real MACs | Params (real) | min. | Batch us/vector | Latency us |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| spacefreq | 32 | GMM K=64 (mix_estimate) | 1,132,544 | 1,018,880 | 4,530,176 | 33,554,496 | 2,031,680 | 1,118 | 56,610 |
| spacefreq | 32 | GMM K=64, precomputed | 1,050,624 | 1,018,880 | 4,202,496 | 2,097,216 | 2,031,680 | 770 | 5,364 |

- spacefreq/32 GMM K=64 (mix_estimate): mix_estimate also spends about 39,845,888 complex MACs per CALL rebuilding its inverses (included in the latency; spread over the chunk in the batch figure).

# Build notes

## Benchmarks (single client, model time only, 200 requests)
| Setup | p50 | p95 | p99 |
|---|---|---|---|
| Local, GPU (RTX 4060 Laptop) | 6.4 ms | 17.7 ms | 27.2 ms |
| Docker container, CPU | 16.4 ms | 24.9 ms | 31.2 ms |

Model: DistilRoBERTa fine-tuned on Banking77 (77 intents).
Test accuracy 92.4%, macro-F1 0.924, 4 epochs, ~67s training on GPU.
Image: 979 MB compressed (CPU-only torch).

## Issues hit and fixes
- `datasets>=4` dropped script-based datasets; Banking77 loads from PolyAI's CSVs instead.
- Dependency resolver silently picked datasets 1.1.1 / evaluate 0.2.0; pinned explicitly.
- First GPU request took ~1.9s from CUDA warm-up; added a warm-up inference at startup.
- pip timed out on the 196 MB torch wheel; added `--timeout 120 --retries 10`.
- `pkill -f uvicorn` broke Docker's port forward; recreating the container fixed it.

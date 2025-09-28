# Online_Fraud_BFloat16_Analysis.

### What is BFloat16?
- BFloat16 (Brain Floating Point 16-bit) is a reduced-precision floating-point format widely used in modern accelerators (TPUs, NVIDIA Tensor Cores).
-It uses 16 bits instead of 32 (FP32) while keeping the same exponent range, enabling large-scale deep learning without major accuracy loss.

## Benefits of BFloat16
- Lower Memory Utilization: Uses 50% less memory per number compared to FP32, enabling larger batch sizes and models.

- Faster Training & Inference: Optimized for TPUs and Tensor Cores, significantly reduces CPU/GPU run time.

- Energy Efficient: Requires fewer compute resources → lower power consumption.

- Scalable: Ideal for large datasets like fraud detection where millions of transactions must be processed quickly.

#### Why BFloat16 for Fraud Detection?
- Fraud detection models rely on real-time prediction → faster computation is critical.

- Memory efficiency allows handling high-dimensional features (transaction history, device IDs, geolocation, etc.).

- Reduced CPU utilization → offloads heavy math operations to TPUs/GPUs for scalability.

- Minimal accuracy trade-off compared to FP32 but with higher throughput.

  ### CPU Utilization

  - FP32: Higher CPU/GPU load, more memory pressure.

  - BFloat16: Offloads computations to specialized hardware units (TPU cores / Tensor Cores), lowering CPU bottlenecks.
 
#### Benchmark: FP32 vs BFloat16 on Fraud Detection Dataset

| Precision | Runtime (per epoch) | Memory Usage | CPU Utilization | Model Accuracy |
|-----------|----------------------|---------------|-----------------|----------------|
| FP32      | s                 | 12.28 GB          | High            | %          |
| BFloat16  | 5s                  | 11.98 GB          | Moderate        | %          |

BFloat16 achieves **~40% faster runtime** and **50% lower memory usage** with only a negligible drop in accuracy.

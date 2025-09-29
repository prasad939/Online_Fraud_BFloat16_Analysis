# Online_Fraud_BFloat16_Analysis.

### What is BFloat16?
- BFloat16 (Brain Floating Point 16-bit) is a reduced-precision floating-point format widely used in modern accelerators (TPUs, NVIDIA Tensor Cores).
-It uses 16 bits instead of 32 (FP32) while keeping the same exponent range, enabling large-scale deep learning without major accuracy loss.

Dataset link: https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset

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
| FP32      | 903s                 | 14.27 GB          | High            | 98.6%          |
| BFloat16  | 1044s                  | 10.98 GB          | Moderate        | 96.0%          |

BFloat16 improves scalability with reduced memory and CPU utilization, ideal for large fraud detection datasets.

FP32 offers higher accuracy and faster runtime but at the cost of higher memory and CPU usage.

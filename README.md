# Intel Scene Classification: Research to Production-Ready MLOps

This project demonstrates the complete lifecycle of a Machine Learning model, transitioning from research in Google Colab to a production-ready, containerized API on GCP Vertex AI. The primary objective was to build a robust MLOps pipeline capable of successfully classifying high-resolution production images using an ensemble model trained on low-resolution data, with all deployment fully automated via GitHub Actions.


## Phase 1: Research & Optimization (Google Colab)

* **Model Foundation:** Utilized **Transfer Learning** with `MobileNetV2` for an optimal balance between accuracy and low latency.
* **Initial Research (Pyramid & Sliding Window):**
   * *Sliding Window:* Allowed the model to scan images at a granular level to prevent feature loss in high-resolution data.
   * *Image Pyramids:* Tested multi-scale representations to improve scale invariance.
   * *Conclusion:* While accurate, these were discarded for production due to high latency.


* **Selected Strategy:** **Fast Ensemble** – A dual-stream approach using OpenCV enhancement (LAB space CLAHE + Bilateral Filtering) to maintain both global context and local detail.



### * Performance Analysis: Baseline vs. Fast Ensemble (V5)
#### 1.  **Why Ensemble method Won:**

* **Dual-Stream Processing:** While the Baseline focused on global structure, the Ensemble's second stream used **LAB Color Space** and **CLAHE** to pull detail from shadows and highlights.
* **Noise Reduction:** The integration of **Bilateral Filtering** ensured that while we enhanced edges, we didn't introduce "artifacts" that could confuse the neural network.
* **Edge Preservation:** The model proved significantly more robust at identifying "Buildings" and "Forests" in varying lighting conditions where the baseline saw only flat colors.

#### 2. **Key Performance Insights**

* **Accuracy Stability:** On complex "Street" and "Building" scenes, the Ensemble method showed higher confidence (e.g., `building6.jpg` improved from **84.6%** to **92.7%**).
* **Efficiency Gains:** The **Latency** was  **30-40% faster** than the Baseline for high-res images due to OpenCV pre-processing optimizing data flow.


| Category | Avg. Baseline Time | Avg. Ensemble Time | Accuracy Trend |
| --- | --- | --- | --- |
| **Buildings** | 1.13s | **0.87s** | Significant Increase |
| **Forest** | 0.98s | **0.78s** | Maintained 100% |
| **Mountains** | 1.18s | **0.95s** | High Stability |
| **Sea** | 0.89s | **0.65s** | High Confidence |


#### 3. **Performance Comparison**

| Method | Accuracy (Low Res) | Accuracy (High Res) | Latency |
| --- | --- | --- | --- |
| **Baseline** | High | Medium | **Very Fast** |
| **Pyramid/Sliding** | Very High | Very High | Slow |
| **Fast Ensemble** | **Very High** | **High** | **Fast (Optimal)** |


#### 4. **Hardware & Optimization Note: CPU Inference**

All benchmarks were conducted using **CPU-only inference**.

* **The Latency Secret:** For lightweight models like MobileNetV2 (~3.4M parameters), the "Overhead" of moving data to a GPU often exceeds the actual computation time. By staying on the CPU, the **Fast Ensemble** stream (OpenCV) and the **Deep Learning** stream (TensorFlow) could share memory buffers efficiently.
* **OpenCV Advantage:** OpenCV's `CLAHE` and `Bilateral Filtering` are highly optimized for CPU instructions. By pre-processing the image into a more "feature-rich" state, the neural network could often reach a confident classification faster during the forward pass.
* **Cost Efficiency:** This demonstrates that high-accuracy scene classification can be achieved in production without the high cost of cloud GPUs, making it ideal for scalable, budget-conscious deployments.


## Phase 2-4: Production & MLOps

1. **FastAPI Integration:** Implemented an `async` lifespan pattern to load the **25MB** model weights efficiently.
2. **Containerization:** Optimized `Dockerfile` to handle the specific shared library requirements of OpenCV in a Linux environment.
3. **CI/CD Pipeline:** Automated via **GitHub Actions** (`deploy.yml`) to trigger on every push to `main`.
4. **Vertex AI Hosting:** Served via a scalable endpoint, achieving 100% success rates in production monitoring.


## Technical Stack

* **Language:** Python 
* **ML Frameworks:** TensorFlow/Keras, OpenCV
* **API Framework:** FastAPI, Uvicorn
* **DevOps:** Docker, GitHub Actions (CI/CD)
* **Cloud Platform:** Google Cloud Platform (Vertex AI, Artifact Registry)

### Project Conclusion
This project successfully bridged the gap between a Machine Learning Notebook and a Scalable Cloud API, delivering high-accuracy predictions on high-resolution data using a lightweight, CPU-optimized ensemble architecture.

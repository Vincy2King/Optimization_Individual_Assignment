## An Empirical Study on Total Variation Regularization for Image Denoising
Individual Assignment for ADVANCED TOPICS IN OPTIMIZATION
### 1. Overview
This project is an individual assignment for the "Advanced Topics in Optimization" course. It addresses the image denoising problem using Total Variation (TV) regularization, modeling the task as a non-smooth convex optimization problem. The project implements and evaluates three classical optimization algorithms: ISTA, FISTA, and ADMM. Through experiments on standard datasets, we conduct an empirical comparison and analysis of their reconstruction quality, convergence dynamics, and computational efficiency.

### 2. Implemented Algorithms
This project implements the following three algorithms for solving the TV image denoising problem:

* Iterative Shrinkage-Thresholding Algorithm (ISTA)
* Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
* Alternating Direction Method of Multipliers (ADMM)

### 3. Getting Started
#### 3.1. Dataset Preparation
This project uses two public benchmark datasets for evaluation. Please download the data from the links below and place the test images into the dataset/ folder in the project's root directory.
* BSDS500: The Berkeley Segmentation Dataset and Benchmark.
> Download Link: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
* DIV2K: The DIVerse 2K resolution image dataset.
> Download Link: https://data.vision.ee.ethz.ch/cvl/DIV2K/

#### 3.2. Running the Code
To run the experiments, execute the main script from the terminal (example):
> python main.py

### 4. Summary of Experimental Results
Our experiments reveal distinct trade-offs in performance among the different algorithms:

* ISTA: Performs best in terms of pixel-level fidelity (PSNR) and computational efficiency. It produces the highest PSNR scores and has the shortest running time.
* ADMM: Excels at preserving structural similarity (SSIM), especially in high-noise environments. It is better at retaining textures and geometric structures but is the most computationally expensive.
* FISTA: As an accelerated version of ISTA, it converges fastest in the initial iterations. However, in our experiments, its final image fidelity did not surpass that of the standard ISTA.

### 5. Conclusion
There is no universally optimal algorithm. The choice of algorithm depends on the specific application needs: ISTA is the ideal choice when speed and pixel accuracy are prioritized, whereas ADMM is more suitable for tasks where preserving structural integrity is critical.

"""
BSDS500 Dataset Image Denoising Experiment
Implementation of ISTA, FISTA, and ADMM algorithms with quantitative and qualitative analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from typing import Tuple, List, Dict, Optional
from scipy import ndimage
from scipy.sparse.linalg import LinearOperator, cg
import time
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import warnings
warnings.filterwarnings('ignore')


class ImageDenoiser:
    """Base class for image denoisers"""
    
    def __init__(self, lambda_reg: float = 0.1, max_iter: int = 100, 
                 tol: float = 1e-6) -> None:
        """
        Initialize denoiser
        
        Args:
            lambda_reg: Regularization parameter
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
        """
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.history: Dict[str, List[float]] = {
            'objective': [],
            'psnr': [],
            'iter_time': []
        }
    
    def soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """
        Soft thresholding function
        
        Args:
            x: Input array
            threshold: Threshold value
            
        Returns:
            Soft-thresholded array
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def compute_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradient (for total variation)
        
        Args:
            x: Current estimate
            y: Observed image
            
        Returns:
            Gradient
        """
        grad_x = np.gradient(x, axis=0)
        grad_y = np.gradient(x, axis=1)
        return grad_x, grad_y
    
    def compute_tv_norm(self, x: np.ndarray) -> float:
        """
        Compute total variation norm
        
        Args:
            x: Input image
            
        Returns:
            TV norm value
        """
        grad_x = np.gradient(x, axis=0)
        grad_y = np.gradient(x, axis=1)
        return np.sum(np.sqrt(grad_x**2 + grad_y**2))
    
    def objective_function(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Objective function: 0.5 * ||x - y||^2 + lambda * TV(x)
        
        Args:
            x: Current estimate
            y: Observed image
            
        Returns:
            Objective function value
        """
        data_fidelity = 0.5 * np.sum((x - y)**2)
        tv_penalty = self.lambda_reg * self.compute_tv_norm(x)
        return data_fidelity + tv_penalty


class ISTA(ImageDenoiser):
    """ISTA (Iterative Shrinkage-Thresholding Algorithm)"""
    
    def __init__(self, lambda_reg: float = 0.1, max_iter: int = 100,
                 tol: float = 1e-6, step_size: float = 0.01) -> None:
        """
        Initialize ISTA algorithm
        
        Args:
            lambda_reg: Regularization parameter
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            step_size: Step size
        """
        super().__init__(lambda_reg, max_iter, tol)
        self.step_size = step_size
    
    def denoise(self, noisy_image: np.ndarray, 
                ground_truth: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Denoise using ISTA algorithm
        
        Args:
            noisy_image: Noisy image
            ground_truth: Ground truth image (for PSNR calculation, optional)
            
        Returns:
            Denoised image
        """
        x = noisy_image.copy()
        y = noisy_image.copy()
        
        self.history = {'objective': [], 'psnr': [], 'iter_time': []}
        
        for i in range(self.max_iter):
            start_time = time.time()
            
            # Compute gradient
            grad_x, grad_y = self.compute_gradient(x, y)
            
            # Compute gradient of total variation
            # Use approximation: div(grad/|grad|) ≈ div(grad) / (|grad| + epsilon)
            epsilon = 1e-8
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2) + epsilon
            
            # Compute TV gradient (simplified version)
            div_x = np.gradient(grad_x / grad_magnitude, axis=0)
            div_y = np.gradient(grad_y / grad_magnitude, axis=1)
            tv_grad = div_x + div_y
            
            # Gradient descent step
            gradient = (x - y) + self.lambda_reg * tv_grad
            x_new = x - self.step_size * gradient
            
            # Project to [0, 1]
            x_new = np.clip(x_new, 0, 1)
            
            # Compute objective function value
            obj_val = self.objective_function(x_new, y)
            self.history['objective'].append(obj_val)
            
            # Compute PSNR (if ground truth available)
            if ground_truth is not None:
                psnr = peak_signal_noise_ratio(
                    ground_truth, x_new, data_range=1.0
                )
                self.history['psnr'].append(psnr)
            
            iter_time = time.time() - start_time
            self.history['iter_time'].append(iter_time)
            
            # Check convergence
            if i > 0:
                relative_change = np.abs(obj_val - self.history['objective'][-2]) / (
                    self.history['objective'][-2] + 1e-10
                )
                if relative_change < self.tol:
                    break
            
            x = x_new
        
        return x


class FISTA(ImageDenoiser):
    """FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)"""
    
    def __init__(self, lambda_reg: float = 0.1, max_iter: int = 100,
                 tol: float = 1e-6, step_size: float = 0.01) -> None:
        """
        Initialize FISTA algorithm
        
        Args:
            lambda_reg: Regularization parameter
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            step_size: Step size
        """
        super().__init__(lambda_reg, max_iter, tol)
        self.step_size = step_size
    
    def denoise(self, noisy_image: np.ndarray,
                ground_truth: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Denoise using FISTA algorithm
        
        Args:
            noisy_image: Noisy image
            ground_truth: Ground truth image (for PSNR calculation, optional)
            
        Returns:
            Denoised image
        """
        x = noisy_image.copy()
        y = noisy_image.copy()
        z = x.copy()
        t = 1.0
        
        self.history = {'objective': [], 'psnr': [], 'iter_time': []}
        
        for i in range(self.max_iter):
            start_time = time.time()
            
            # Compute gradient on z
            grad_x, grad_y = self.compute_gradient(z, y)
            epsilon = 1e-8
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2) + epsilon
            
            div_x = np.gradient(grad_x / grad_magnitude, axis=0)
            div_y = np.gradient(grad_y / grad_magnitude, axis=1)
            tv_grad = div_x + div_y
            
            # Gradient descent step
            gradient = (z - y) + self.lambda_reg * tv_grad
            x_new = z - self.step_size * gradient
            x_new = np.clip(x_new, 0, 1)
            
            # FISTA update
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
            z = x_new + ((t - 1) / t_new) * (x_new - x)
            
            # Compute objective function value
            obj_val = self.objective_function(x_new, y)
            self.history['objective'].append(obj_val)
            
            # Compute PSNR
            if ground_truth is not None:
                psnr = peak_signal_noise_ratio(
                    ground_truth, x_new, data_range=1.0
                )
                self.history['psnr'].append(psnr)
            
            iter_time = time.time() - start_time
            self.history['iter_time'].append(iter_time)
            
            # Check convergence
            if i > 0:
                relative_change = np.abs(obj_val - self.history['objective'][-2]) / (
                    self.history['objective'][-2] + 1e-10
                )
                if relative_change < self.tol:
                    break
            
            x = x_new
            t = t_new
        
        return x


class ADMM(ImageDenoiser):
    """ADMM (Alternating Direction Method of Multipliers)"""
    
    def __init__(self, lambda_reg: float = 0.1, max_iter: int = 100,
                 tol: float = 1e-6, rho: float = 1.0) -> None:
        """
        Initialize ADMM algorithm
        
        Args:
            lambda_reg: Regularization parameter
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            rho: ADMM penalty parameter
        """
        super().__init__(lambda_reg, max_iter, tol)
        self.rho = rho
    
    def denoise(self, noisy_image: np.ndarray,
                ground_truth: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Denoise using ADMM algorithm
        
        Args:
            noisy_image: Noisy image
            ground_truth: Ground truth image (for PSNR calculation, optional)
            
        Returns:
            Denoised image
        """
        y = noisy_image.copy()
        x = y.copy()
        
        # Initialize auxiliary and dual variables (for TV term)
        # Decompose TV term as Dx = z, where D is gradient operator
        grad_x_init, grad_y_init = self.compute_gradient(x, y)
        z_x = grad_x_init.copy()
        z_y = grad_y_init.copy()
        u_x = np.zeros_like(z_x)
        u_y = np.zeros_like(z_y)
        
        self.history = {'objective': [], 'psnr': [], 'iter_time': []}
        
        for i in range(self.max_iter):
            start_time = time.time()
            
            # x-update: solve (I + rho*D^T*D) * x = y + rho*D^T*(z - u)
            # Compute D^T*(z - u), where D^T is divergence operator (negative transpose of gradient)
            diff_x = z_x - u_x
            diff_y = z_y - u_y
            
            # Compute divergence: div = -∂/∂x(diff_x) - ∂/∂y(diff_y)
            # Use finite difference approximation
            h, w = diff_x.shape
            div_x = np.zeros_like(diff_x)
            div_y = np.zeros_like(diff_y)
            
            # Divergence in x direction
            div_x[1:, :] = diff_x[1:, :] - diff_x[:-1, :]
            div_x[0, :] = diff_x[0, :]
            
            # Divergence in y direction
            div_y[:, 1:] = diff_y[:, 1:] - diff_y[:, :-1]
            div_y[:, 0] = diff_y[:, 0]
            
            div_term = -(div_x + div_y)
            
            # x-update
            x = (y + self.rho * div_term) / (1 + self.rho)
            x = np.clip(x, 0, 1)
            
            # z-update: soft thresholding (on gradient)
            grad_x_new, grad_y_new = self.compute_gradient(x, y)
            grad_with_dual_x = grad_x_new + u_x
            grad_with_dual_y = grad_y_new + u_y
            
            grad_magnitude = np.sqrt(grad_with_dual_x**2 + grad_with_dual_y**2)
            threshold = self.lambda_reg / self.rho
            
            # Soft thresholding operation
            shrink_factor = np.maximum(1 - threshold / (grad_magnitude + 1e-10), 0)
            z_x = shrink_factor * grad_with_dual_x
            z_y = shrink_factor * grad_with_dual_y
            
            # u-update: dual variable update
            u_x = u_x + (grad_x_new - z_x)
            u_y = u_y + (grad_y_new - z_y)
            
            # Compute objective function value
            obj_val = self.objective_function(x, y)
            self.history['objective'].append(obj_val)
            
            # Compute PSNR
            if ground_truth is not None:
                psnr = peak_signal_noise_ratio(
                    ground_truth, x, data_range=1.0
                )
                self.history['psnr'].append(psnr)
            
            iter_time = time.time() - start_time
            self.history['iter_time'].append(iter_time)
            
            # Check convergence
            if i > 0:
                relative_change = np.abs(obj_val - self.history['objective'][-2]) / (
                    self.history['objective'][-2] + 1e-10
                )
                if relative_change < self.tol:
                    break
        
        return x


def load_image(image_path: str) -> np.ndarray:
    """
    Load image and convert to grayscale, normalize to [0, 1]
    
    Args:
        image_path: Image file path
        
    Returns:
        Normalized grayscale image array
    """
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float64) / 255.0
    return img_array


def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Add Gaussian white noise
    
    Args:
        image: Clean image
        sigma: Noise standard deviation
        
    Returns:
        Noisy image
    """
    noise = np.random.normal(0, sigma / 255.0, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image


def evaluate_image(ground_truth: np.ndarray, denoised: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate denoised image quality
    
    Args:
        ground_truth: Ground truth image
        denoised: Denoised image
        
    Returns:
        (PSNR, SSIM) tuple
    """
    psnr = peak_signal_noise_ratio(ground_truth, denoised, data_range=1.0)
    ssim = structural_similarity(ground_truth, denoised, data_range=1.0)
    return psnr, ssim


def run_experiments(dataset_path: str, noise_levels: List[float],
                   num_images: int = 5) -> Dict:
    """
    Run complete experiments
    
    Args:
        dataset_path: Dataset path
        noise_levels: List of noise levels (sigma values)
        num_images: Number of images to use
        
    Returns:
        Experiment results dictionary
    """
    # Get test image list
    test_dir = os.path.join(dataset_path, 'images', 'test')
    image_files = sorted([f for f in os.listdir(test_dir) 
                         if f.endswith(('.jpg', '.png'))])[:num_images]
    
    results = {
        'noise_levels': noise_levels,
        'algorithms': ['ISTA', 'FISTA', 'ADMM'],
        'psnr_results': {alg: {sigma: [] for sigma in noise_levels} 
                        for alg in ['ISTA', 'FISTA', 'ADMM']},
        'ssim_results': {alg: {sigma: [] for sigma in noise_levels} 
                        for alg in ['ISTA', 'FISTA', 'ADMM']},
        'time_results': {alg: {sigma: [] for sigma in noise_levels} 
                        for alg in ['ISTA', 'FISTA', 'ADMM']},
        'convergence_history': {alg: {} for alg in ['ISTA', 'FISTA', 'ADMM']},
        'image_results': []
    }
    
    # Algorithm parameter settings
    algorithm_params = {
        'ISTA': {'lambda_reg': 0.1, 'step_size': 0.01, 'max_iter': 100},
        'FISTA': {'lambda_reg': 0.1, 'step_size': 0.01, 'max_iter': 100},
        'ADMM': {'lambda_reg': 0.1, 'rho': 1.0, 'max_iter': 100}
    }
    
    print("=" * 60)
    print("Starting BSDS500 Image Denoising Experiment")
    print("=" * 60)
    
    for img_idx, img_file in enumerate(image_files):
        print(f"\nProcessing image {img_idx + 1}/{len(image_files)}: {img_file}")
        
        # Load image
        img_path = os.path.join(test_dir, img_file)
        ground_truth = load_image(img_path)
        
        for sigma in noise_levels:
            print(f"  Noise level σ = {sigma}")
            
            # Add noise
            np.random.seed(42)  # For reproducibility
            noisy_image = add_gaussian_noise(ground_truth, sigma)
            
            # Run each algorithm
            for alg_name in ['ISTA', 'FISTA', 'ADMM']:
                params = algorithm_params[alg_name]
                
                if alg_name == 'ISTA':
                    denoiser = ISTA(**params)
                elif alg_name == 'FISTA':
                    denoiser = FISTA(**params)
                else:  # ADMM
                    denoiser = ADMM(**params)
                
                # Denoise
                start_time = time.time()
                denoised = denoiser.denoise(noisy_image, ground_truth)
                total_time = time.time() - start_time
                
                # Evaluate
                psnr, ssim = evaluate_image(ground_truth, denoised)
                
                # Save results
                results['psnr_results'][alg_name][sigma].append(psnr)
                results['ssim_results'][alg_name][sigma].append(ssim)
                results['time_results'][alg_name][sigma].append(total_time)
                
                # Save convergence history of first image (for plotting)
                if img_idx == 0 and sigma == noise_levels[0]:
                    results['convergence_history'][alg_name] = denoiser.history.copy()
                
                print(f"    {alg_name}: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}, "
                      f"Time={total_time:.2f}s")
        
        # Save first image results for visualization
        if img_idx == 0:
            results['image_results'] = {
                'ground_truth': ground_truth,
                'noisy_images': {},
                'denoised_images': {alg: {} for alg in ['ISTA', 'FISTA', 'ADMM']}
            }
            for sigma in noise_levels:
                np.random.seed(42)
                noisy = add_gaussian_noise(ground_truth, sigma)
                results['image_results']['noisy_images'][sigma] = noisy
                
                for alg_name in ['ISTA', 'FISTA', 'ADMM']:
                    params = algorithm_params[alg_name]
                    if alg_name == 'ISTA':
                        denoiser = ISTA(**params)
                    elif alg_name == 'FISTA':
                        denoiser = FISTA(**params)
                    else:
                        denoiser = ADMM(**params)
                    denoised = denoiser.denoise(noisy, ground_truth)
                    results['image_results']['denoised_images'][alg_name][sigma] = denoised
    
    return results


def print_quantitative_results(results: Dict) -> None:
    """
    Print quantitative results table
    
    Args:
        results: Experiment results dictionary
    """
    print("\n" + "=" * 80)
    print("Quantitative Results Analysis")
    print("=" * 80)
    
    noise_levels = results['noise_levels']
    algorithms = results['algorithms']
    
    # PSNR table
    print("\nAverage PSNR (dB):")
    print("-" * 80)
    print(f"{'Algorithm':<10}", end="")
    for sigma in noise_levels:
        print(f"σ={sigma:<6}", end="")
    print()
    print("-" * 80)
    
    for alg in algorithms:
        print(f"{alg:<10}", end="")
        for sigma in noise_levels:
            avg_psnr = np.mean(results['psnr_results'][alg][sigma])
            print(f"{avg_psnr:>8.2f}", end="")
        print()
    
    # SSIM table
    print("\nAverage SSIM:")
    print("-" * 80)
    print(f"{'Algorithm':<10}", end="")
    for sigma in noise_levels:
        print(f"σ={sigma:<6}", end="")
    print()
    print("-" * 80)
    
    for alg in algorithms:
        print(f"{alg:<10}", end="")
        for sigma in noise_levels:
            avg_ssim = np.mean(results['ssim_results'][alg][sigma])
            print(f"{avg_ssim:>8.4f}", end="")
        print()
    
    # Runtime table
    print("\nAverage Runtime (seconds):")
    print("-" * 80)
    print(f"{'Algorithm':<10}", end="")
    for sigma in noise_levels:
        print(f"σ={sigma:<6}", end="")
    print()
    print("-" * 80)
    
    for alg in algorithms:
        print(f"{alg:<10}", end="")
        for sigma in noise_levels:
            avg_time = np.mean(results['time_results'][alg][sigma])
            print(f"{avg_time:>8.2f}", end="")
        print()


def plot_convergence_analysis(results: Dict, save_path: str = None) -> None:
    """
    Plot convergence analysis
    
    Args:
        results: Experiment results dictionary
        save_path: Save path (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    algorithms = results['algorithms']
    colors = {'ISTA': 'blue', 'FISTA': 'red', 'ADMM': 'green'}
    
    # PSNR vs iterations
    ax1 = axes[0]
    for alg in algorithms:
        if alg in results['convergence_history'] and 'psnr' in results['convergence_history'][alg]:
            psnr_history = results['convergence_history'][alg]['psnr']
            iterations = range(1, len(psnr_history) + 1)
            ax1.plot(iterations, psnr_history, color=colors[alg], 
                    label=alg, linewidth=2)
    
    ax1.set_xlabel('Iterations', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('PSNR vs Iterations', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Objective function value vs iterations
    ax2 = axes[1]
    for alg in algorithms:
        if alg in results['convergence_history'] and 'objective' in results['convergence_history'][alg]:
            obj_history = results['convergence_history'][alg]['objective']
            iterations = range(1, len(obj_history) + 1)
            ax2.semilogy(iterations, obj_history, color=colors[alg], 
                        label=alg, linewidth=2)
    
    ax2.set_xlabel('Iterations', fontsize=12)
    ax2.set_ylabel('Objective Value (log scale)', fontsize=12)
    ax2.set_title('Objective Value vs Iterations', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConvergence analysis plot saved to: {save_path}")
    plt.show()


def plot_visual_comparison(results: Dict, sigma: float = 25, 
                          save_path: str = None) -> None:
    """
    Plot visual comparison
    
    Args:
        results: Experiment results dictionary
        sigma: Noise level to display
        save_path: Save path (optional)
    """
    if 'image_results' not in results or not results['image_results']:
        print("No image results available for visualization")
        return
    
    img_results = results['image_results']
    ground_truth = img_results['ground_truth']
    noisy = img_results['noisy_images'][sigma]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # First row: Original, Noisy, ISTA
    axes[0, 0].imshow(ground_truth, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title(f'Noisy Image (σ={sigma})', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    ista_result = img_results['denoised_images']['ISTA'][sigma]
    axes[0, 2].imshow(ista_result, cmap='gray')
    psnr, ssim = evaluate_image(ground_truth, ista_result)
    axes[0, 2].set_title(f'ISTA (PSNR={psnr:.2f}dB, SSIM={ssim:.4f})', 
                         fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Second row: FISTA, ADMM, zoomed region comparison
    fista_result = img_results['denoised_images']['FISTA'][sigma]
    axes[1, 0].imshow(fista_result, cmap='gray')
    psnr, ssim = evaluate_image(ground_truth, fista_result)
    axes[1, 0].set_title(f'FISTA (PSNR={psnr:.2f}dB, SSIM={ssim:.4f})', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    admm_result = img_results['denoised_images']['ADMM'][sigma]
    axes[1, 1].imshow(admm_result, cmap='gray')
    psnr, ssim = evaluate_image(ground_truth, admm_result)
    axes[1, 1].set_title(f'ADMM (PSNR={psnr:.2f}dB, SSIM={ssim:.4f})', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Zoomed region comparison (select center region of image)
    h, w = ground_truth.shape
    crop_size = min(h, w) // 3
    start_h, start_w = h // 2 - crop_size // 2, w // 2 - crop_size // 2
    
    # Create zoomed comparison plot
    axes[1, 2].axis('off')
    # Can add detailed zoomed images here, simplified for now
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisual comparison plot saved to: {save_path}")
    plt.show()


def plot_efficiency_comparison(results: Dict, save_path: str = None) -> None:
    """
    Plot efficiency comparison
    
    Args:
        results: Experiment results dictionary
        save_path: Save path (optional)
    """
    noise_levels = results['noise_levels']
    algorithms = results['algorithms']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Average PSNR vs average runtime
    ax1 = axes[0]
    colors = {'ISTA': 'blue', 'FISTA': 'red', 'ADMM': 'green'}
    markers = {'ISTA': 'o', 'FISTA': 's', 'ADMM': '^'}
    
    for alg in algorithms:
        avg_psnr = [np.mean(results['psnr_results'][alg][sigma]) 
                   for sigma in noise_levels]
        avg_time = [np.mean(results['time_results'][alg][sigma]) 
                   for sigma in noise_levels]
        ax1.scatter(avg_time, avg_psnr, color=colors[alg], 
                   marker=markers[alg], s=100, label=alg, alpha=0.7)
    
    ax1.set_xlabel('Average Runtime (seconds)', fontsize=12)
    ax1.set_ylabel('Average PSNR (dB)', fontsize=12)
    ax1.set_title('Efficiency Comparison: PSNR vs Runtime', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Runtime comparison across different noise levels
    ax2 = axes[1]
    x = np.arange(len(noise_levels))
    width = 0.25
    
    for i, alg in enumerate(algorithms):
        avg_times = [np.mean(results['time_results'][alg][sigma]) 
                    for sigma in noise_levels]
        ax2.bar(x + i * width, avg_times, width, label=alg, 
               color=colors[alg], alpha=0.7)
    
    ax2.set_xlabel('Noise Level σ', fontsize=12)
    ax2.set_ylabel('Average Runtime (seconds)', fontsize=12)
    ax2.set_title('Runtime Across Different Noise Levels', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f'σ={sigma}' for sigma in noise_levels])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nEfficiency comparison plot saved to: {save_path}")
    plt.show()


def main() -> None:
    """Main function"""
    # Experimental setup
    dataset_path = '/data16T_2/wangbingbing/PolyU-Optimization/BSDS500'
    noise_levels = [15, 25, 50]  # Noise levels
    num_images = 5  # Number of images to use
    
    print("Experimental Setup:")
    print(f"  Dataset: BSDS500")
    print(f"  Noise Model: Additive Gaussian White Noise")
    print(f"  Noise Levels: σ = {noise_levels}")
    print(f"  Evaluation Metrics: PSNR, SSIM, Runtime")
    print(f"  Algorithms: ISTA, FISTA, ADMM")
    print(f"  Number of Images: {num_images}")
    
    # Run experiments
    results = run_experiments(dataset_path, noise_levels, num_images)
    
    # Print quantitative results
    print_quantitative_results(results)
    
    # Plot convergence analysis
    print("\nPlotting convergence analysis...")
    plot_convergence_analysis(results, 
                             save_path='convergence_analysis.png')
    
    # Plot efficiency comparison
    print("\nPlotting efficiency comparison...")
    plot_efficiency_comparison(results, 
                              save_path='efficiency_comparison.png')
    
    # Plot visual comparison
    print("\nPlotting visual comparison...")
    for sigma in noise_levels:
        plot_visual_comparison(results, sigma=sigma, 
                             save_path=f'visual_comparison_sigma_{sigma}.png')
    
    print("\n" + "=" * 80)
    print("Experiment Completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()


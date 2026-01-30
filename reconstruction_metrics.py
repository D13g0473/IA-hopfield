import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class HopfieldNetwork:
    """Simple Hopfield Network."""
    
    def __init__(self, num_neurons, normalize_weights=True):
        self.num_neurons = num_neurons
        self.normalize_weights = normalize_weights
        self.weights = np.zeros((num_neurons, num_neurons))
    
    def train(self, patterns):
        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        for pattern in patterns:
            pattern = pattern.reshape(-1, 1)
            if self.normalize_weights:
                self.weights += np.dot(pattern, pattern.T) / self.num_neurons
            else:
                self.weights += np.dot(pattern, pattern.T)
        np.fill_diagonal(self.weights, 0)
    
    def update(self, pattern, steps=20):
        pattern = pattern.reshape(-1, 1)
        for _ in range(steps):
            activation = np.dot(self.weights, pattern)
            pattern = np.where(activation >= 0, 1, -1)
        return pattern.flatten()
    
    def energy(self, pattern):
        pattern = pattern.reshape(-1, 1)
        return -0.5 * pattern.T @ self.weights @ pattern

def hamming_distance(p1, p2):
    """Number of differing bits."""
    return np.sum(p1 != p2)

def bit_accuracy(p1, p2):
    """Percentage of matching bits."""
    return np.sum(p1 == p2) / len(p1) * 100

def mse(p1, p2):
    """Mean squared error."""
    return np.mean((p1 - p2) ** 2)

def rmse(p1, p2):
    """Root mean squared error."""
    return np.sqrt(mse(p1, p2))

def peak_signal_to_noise_ratio(p1, p2):
    """PSNR in dB."""
    max_val = 2  # Range is -1 to 1
    mse_val = mse(p1, p2)
    if mse_val == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse_val))

def structural_similarity(p1, p2):
    """Simplified SSIM for binary patterns."""
    # Mean and variance
    mu1, mu2 = np.mean(p1), np.mean(p2)
    var1, var2 = np.var(p1), np.var(p2)
    cov = np.mean((p1 - mu1) * (p2 - mu2))
    
    # Constants
    k1, k2 = 0.01, 0.03
    L = 2  # Dynamic range
    
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    
    numerator = (2 * mu1 * mu2 + c1) * (2 * cov + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)
    
    if denominator == 0:
        return 0
    return numerator / denominator

def load_patterns(csv_files):
    patterns = []
    labels = []
    for file in csv_files:
        data = pd.read_csv(file, header=None)
        matriz = data.values
        matriz = np.where(matriz == 0, -1, matriz)
        patterns.append(matriz.flatten())
        labels.append(file.split('.')[0])
    return np.array(patterns), labels

def load_noisy_examples(data_folder):
    examples = []
    true_labels = []
    for letra in ['A', 'B', 'C', 'D', 'E']:
        folder = os.path.join(data_folder, f'letra_{letra}')
        for i in range(1, 11):
            path = os.path.join(folder, f'Example_{i}.csv')
            if os.path.exists(path):
                data = pd.read_csv(path, header=None)
                matriz = data.values
                matriz = np.where(matriz == 0, -1, matriz)
                examples.append(matriz.flatten())
                true_labels.append(letra)
    return np.array(examples), true_labels

def evaluate_reconstruction(original_patterns, original_labels, noisy_examples, true_labels, hopfield_net):
    """Evaluate ONLY reconstruction metrics."""
    results = {
        'hamming_distance': [],
        'bit_accuracy': [],
        'mse': [],
        'rmse': [],
        'psnr': [],
        'ssim': [],
        'energy': [],
        'energy_reduction': [],
    }
    
    original_energies = [hopfield_net.energy(p) for p in original_patterns]
    
    for i, noisy in enumerate(noisy_examples):
        reconstructed = hopfield_net.update(noisy, steps=20)
        true_letter = true_labels[i]
        true_idx = original_labels.index(true_letter)
        original = original_patterns[true_idx]
        
        # Compute metrics
        results['hamming_distance'].append(hamming_distance(original, reconstructed))
        results['bit_accuracy'].append(bit_accuracy(original, reconstructed))
        results['mse'].append(mse(original, reconstructed))
        results['rmse'].append(rmse(original, reconstructed))
        results['psnr'].append(peak_signal_to_noise_ratio(original, reconstructed))
        results['ssim'].append(structural_similarity(original, reconstructed))
        
        final_energy = hopfield_net.energy(reconstructed)
        results['energy'].append(final_energy)
        results['energy_reduction'].append(original_energies[true_idx] - final_energy)
    
    return results

def print_reconstruction_metrics(results, dataset_name):
    """Print reconstruction metrics."""
    print(f"\n{'='*60}")
    print(f"MÉTRICAS DE RECONSTRUCCIÓN - {dataset_name}")
    print(f"{'='*60}")
    
    metrics_summary = {
        'Distancia Hamming': (results['hamming_distance'], 'bits'),
        'Exactitud de Bits': (results['bit_accuracy'], '%'),
        'MSE': (results['mse'], ''),
        'RMSE': (results['rmse'], ''),
        'PSNR': (results['psnr'], 'dB'),
        'SSIM': (results['ssim'], ''),
        'Energía Final': (results['energy'], ''),
        'Reducción de Energía': (results['energy_reduction'], ''),
    }
    
    print(f"{'Métrica':<25} {'Media':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 70)
    
    for name, (values, unit) in metrics_summary.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        if unit:
            print(f"{name:<25} {mean_val:<12.4f} {std_val:<12.4f} {min_val:<12.4f} {max_val:<12.4f}")
        else:
            print(f"{name:<25} {mean_val:<12.4f} {std_val:<12.4f} {min_val:<12.4f} {max_val:<12.4f}")

def plot_reconstruction_comparison(results_dict, output_file='reconstruction_metrics.png'):
    """Plot comparison of reconstruction metrics across datasets."""
    metrics_to_plot = ['bit_accuracy', 'hamming_distance', 'ssim', 'psnr']
    metric_labels = ['Exactitud de Bits (%)', 'Distancia Hamming', 'SSIM', 'PSNR (dB)']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
        data = [results_dict[ds][metric] for ds in results_dict]
        labels = list(results_dict.keys())
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = ['#3498db', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(label)
        ax.set_title(f'{label} por Dataset')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"\nGráfico guardado en: {output_file}")

def main():
    csv_files = ['A.csv', 'B.csv', 'C.csv', 'D.csv', 'E.csv']
    original_patterns, original_labels = load_patterns(csv_files)
    
    # Train network
    hopfield_net = HopfieldNetwork(100, normalize_weights=True)
    hopfield_net.train(original_patterns)
    
    # Evaluate on both datasets
    results_dict = {}
    
    for dataset in ['data_set', 'data_set_low_noise']:
        if not os.path.exists(dataset):
            continue
            
        noisy_examples, true_labels = load_noisy_examples(dataset)
        results = evaluate_reconstruction(original_patterns, original_labels, noisy_examples, true_labels, hopfield_net)
        results_dict[dataset] = results
        
        print_reconstruction_metrics(results, dataset.upper())
    
    # Plot comparison
    if len(results_dict) > 1:
        plot_reconstruction_comparison(results_dict)
    
    # Summary table
    print(f"\n{'='*60}")
    print("COMPARACIÓN FINAL")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'Bit Accuracy':<15} {'Hamming':<12} {'SSIM':<12} {'PSNR':<12}")
    print("-" * 70)
    for ds, res in results_dict.items():
        print(f"{ds:<20} {np.mean(res['bit_accuracy']):<15.2f} {np.mean(res['hamming_distance']):<12.2f} {np.mean(res['ssim']):<12.4f} {np.mean(res['psnr']):<12.2f}")

if __name__ == "__main__":
    main()
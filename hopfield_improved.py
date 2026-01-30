import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class HopfieldNetworkImproved:
    """Improved Hopfield Network with weight normalization, async updates, and energy tracking."""
    
    def __init__(self, num_neurons, normalize_weights=True, use_async=False):
        self.num_neurons = num_neurons
        self.normalize_weights = normalize_weights
        self.use_async = use_async
        self.weights = np.zeros((num_neurons, num_neurons))
    
    def train(self, patterns):
        """Train with Hebb rule, optionally normalized."""
        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        for pattern in patterns:
            pattern = pattern.reshape(-1, 1)
            if self.normalize_weights:
                self.weights += np.dot(pattern, pattern.T) / self.num_neurons
            else:
                self.weights += np.dot(pattern, pattern.T)
        np.fill_diagonal(self.weights, 0)
    
    def energy(self, pattern):
        """Calculate Hopfield energy function."""
        pattern = pattern.reshape(-1, 1)
        return -0.5 * pattern.T @ self.weights @ pattern
    
    def update_sync(self, pattern, steps=10):
        """Synchronous update (original method)."""
        pattern = pattern.reshape(-1, 1)
        for _ in range(steps):
            activation = np.dot(self.weights, pattern)
            pattern = np.where(activation >= 0, 1, -1)
        return pattern.flatten()
    
    def update_async(self, pattern, steps=None):
        """Asynchronous update (improved method)."""
        pattern = pattern.copy().reshape(-1, 1)
        if steps is None:
            steps = self.num_neurons  # One full sweep through all neurons
        
        energy_history = [self.energy(pattern)]
        for _ in range(steps):
            # Random order of neuron updates
            indices = np.random.permutation(self.num_neurons)
            for i in indices:
                activation = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if activation >= 0 else -1
            energy_history.append(self.energy(pattern))
        return pattern.flatten(), energy_history
    
    def update(self, pattern, steps=10):
        """Unified update method based on use_async flag."""
        if self.use_async:
            result, _ = self.update_async(pattern, steps)
            return result
        else:
            return self.update_sync(pattern, steps)

def center_patterns(patterns):
    """Center patterns by subtracting mean (reduces correlation)."""
    return patterns - np.mean(patterns, axis=1, keepdims=True)

def hamming_distance(pattern1, pattern2):
    """Calculate Hamming distance between two patterns."""
    return np.sum(pattern1 != pattern2)

def load_patterns_from_csv(csv_files):
    """Load patterns from CSV files."""
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
    """Load noisy examples from dataset folder."""
    examples = []
    true_labels = []
    for letra in ['A', 'B', 'C', 'D', 'E']:
        letra_folder = os.path.join(data_folder, f'letra_{letra}')
        for i in range(1, 11):
            file_path = os.path.join(letra_folder, f'Example_{i}.csv')
            if os.path.exists(file_path):
                data = pd.read_csv(file_path, header=None)
                matriz = data.values
                matriz = np.where(matriz == 0, -1, matriz)
                examples.append(matriz.flatten())
                true_labels.append(letra)
    return np.array(examples), true_labels

def evaluate_network(original_patterns, original_labels, noisy_examples, true_labels, hopfield_net, use_energy_class=False):
    """Evaluate network on noisy examples with optional energy-based classification."""
    predictions = []
    hamming_distances = []
    reconstruction_accuracies = []
    energy_values = []
    
    for noisy in noisy_examples:
        if hopfield_net.use_async:
            reconstructed, energy_history = hopfield_net.update_async(noisy, steps=20)
            energy_values.append(energy_history[-1])  # Final energy
        else:
            reconstructed = hopfield_net.update(noisy, steps=20)
            energy_values.append(hopfield_net.energy(reconstructed))
        
        # Classification method: Hamming distance or Energy minimization
        if use_energy_class:
            # Energy-based classification: find pattern with lowest energy
            energies = [hopfield_net.energy(orig) for orig in original_patterns]
            pred_idx = np.argmin(energies)
            predictions.append(original_labels[pred_idx])
        else:
            # Hamming distance based classification
            distances = [hamming_distance(reconstructed, orig) for orig in original_patterns]
            pred_idx = np.argmin(distances)
            predictions.append(original_labels[pred_idx])
        
        # Hamming distance between noisy and reconstructed
        hamming_distances.append(hamming_distance(noisy, reconstructed))
        
        # Reconstruction accuracy
        true_idx = original_labels.index(true_labels[len(predictions)-1])
        true_pattern = original_patterns[true_idx]
        correct_bits = np.sum(reconstructed == true_pattern)
        reconstruction_accuracies.append((correct_bits / len(true_pattern)) * 100)
    
    return predictions, hamming_distances, reconstruction_accuracies, energy_values

def compute_metrics(true_labels, predictions):
    """Compute classification metrics."""
    cm = confusion_matrix(true_labels, predictions, labels=['A', 'B', 'C', 'D', 'E'])
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
    return cm, accuracy, precision, recall, f1

def run_experiment(data_folder='data_set', output_prefix='experiment', center_patterns_flag=False, use_energy_class=False):
    """Run comprehensive experiment comparing different configurations."""
    csv_files = ['A.csv', 'B.csv', 'C.csv', 'D.csv', 'E.csv']
    original_patterns, original_labels = load_patterns_from_csv(csv_files)
    
    # Optionally center patterns to reduce correlation
    if center_patterns_flag:
        original_patterns = center_patterns(original_patterns)
    
    noisy_examples, true_labels = load_noisy_examples(data_folder)
    
    configurations = [
        {'name': 'Original', 'normalize': False, 'use_async': False},
        {'name': 'Normalized', 'normalize': True, 'use_async': False},
        {'name': 'Async', 'normalize': True, 'use_async': True},
    ]
    
    results = {}
    
    for config in configurations:
        num_neurons = original_patterns[0].size
        hopfield_net = HopfieldNetworkImproved(
            num_neurons, 
            normalize_weights=config['normalize'],
            use_async=config['use_async']
        )
        
        # Train
        hopfield_net.train(original_patterns)
        
        # Evaluate
        predictions, hamming_dist, recon_acc, energy_vals = evaluate_network(
            original_patterns, original_labels, noisy_examples, true_labels, 
            hopfield_net, use_energy_class=use_energy_class
        )
        
        cm, accuracy, precision, recall, f1 = compute_metrics(true_labels, predictions)
        
        config_name = config['name']
        if center_patterns_flag:
            config_name += ' (centered)'
        if use_energy_class:
            config_name += ' (energy)'
        
        results[config_name] = {
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'hamming_distance': np.mean(hamming_dist),
            'reconstruction_accuracy': np.mean(recon_acc),
            'energy': np.mean(energy_vals),
        }
    
    return results

def plot_results(results, data_folder, output_prefix='experiment'):
    """Plot comparison results."""
    labels = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(labels))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        values = [results[label][metric] for label in labels]
        ax.bar(x + i*width, values, width, label=metric)
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Score')
    ax.set_title(f'Performance Comparison - {data_folder}')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_{data_folder}.png')
    plt.close()
    
    # Print results table
    print(f"\n=== Results for {data_folder} ===")
    for label, config in results.items():
        print(f"\n{label}:")
        print(f"  Accuracy: {config['accuracy']:.4f}")
        print(f"  Precision: {config['precision']:.4f}")
        print(f"  Recall: {config['recall']:.4f}")
        print(f"  F1: {config['f1']:.4f}")
        print(f"  Hamming Distance: {config['hamming_distance']:.2f}")
        print(f"  Reconstruction Accuracy: {config['reconstruction_accuracy']:.2f}%")
        print(f"  Energy: {config['energy']:.2f}")

if __name__ == "__main__":
    # Run comprehensive experiments
    print("="*60)
    print("EXPERIMENT 1: Original patterns (data_set)")
    print("="*60)
    for data_folder in ['data_set', 'data_set_low_noise']:
        if os.path.exists(data_folder):
            print(f"\n--- {data_folder} ---")
            results = run_experiment(data_folder=data_folder, center_patterns_flag=False, use_energy_class=False)
            plot_results(results, data_folder, output_prefix='experiment')
    
    print("\n" + "="*60)
    print("EXPERIMENT 2: Centered patterns (reduced correlation)")
    print("="*60)
    for data_folder in ['data_set', 'data_set_low_noise']:
        if os.path.exists(data_folder):
            print(f"\n--- {data_folder} ---")
            results = run_experiment(data_folder=data_folder, center_patterns_flag=True, use_energy_class=False)
            plot_results(results, f'{data_folder}_centered', output_prefix='experiment')
    
    print("\n" + "="*60)
    print("EXPERIMENT 3: Energy-based classification")
    print("="*60)
    for data_folder in ['data_set', 'data_set_low_noise']:
        if os.path.exists(data_folder):
            print(f"\n--- {data_folder} ---")
            results = run_experiment(data_folder=data_folder, center_patterns_flag=False, use_energy_class=True)
            plot_results(results, f'{data_folder}_energy', output_prefix='experiment')

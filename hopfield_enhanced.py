import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class HopfieldNetworkEnhanced:
    """Enhanced Hopfield Network with multiple improvements."""
    
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
        """Synchronous update."""
        pattern = pattern.reshape(-1, 1)
        for _ in range(steps):
            activation = np.dot(self.weights, pattern)
            pattern = np.where(activation >= 0, 1, -1)
        return pattern.flatten()
    
    def update_async(self, pattern, steps=None):
        """Asynchronous update."""
        pattern = pattern.copy().reshape(-1, 1)
        if steps is None:
            steps = self.num_neurons
        
        energy_history = [self.energy(pattern)]
        for _ in range(steps):
            indices = np.random.permutation(self.num_neurons)
            for i in indices:
                activation = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if activation >= 0 else -1
            energy_history.append(self.energy(pattern))
        return pattern.flatten(), energy_history
    
    def update(self, pattern, steps=10):
        """Unified update method."""
        if self.use_async:
            result, _ = self.update_async(pattern, steps)
            return result
        else:
            return self.update_sync(pattern, steps)

def hamming_distance(pattern1, pattern2):
    """Calculate Hamming distance."""
    return np.sum(pattern1 != pattern2)

def correlation(pattern1, pattern2):
    """Calculate Pearson correlation coefficient."""
    return np.corrcoef(pattern1, pattern2)[0, 1]

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

def evaluate_with_correlation(original_patterns, original_labels, noisy_examples, true_labels, hopfield_net, threshold=0.7):
    """Evaluate network using correlation for classification."""
    predictions = []
    reconstruction_accuracies = []
    
    for noisy in noisy_examples:
        reconstructed = hopfield_net.update(noisy, steps=20)
        
        # Classify based on highest correlation with original patterns
        correlations = [correlation(reconstructed, orig) for orig in original_patterns]
        max_corr_idx = np.argmax(correlations)
        
        # Only accept prediction if correlation is above threshold
        if correlations[max_corr_idx] >= threshold:
            predictions.append(original_labels[max_corr_idx])
        else:
            predictions.append('Unknown')  # Low confidence
        
        # Reconstruction accuracy
        true_idx = original_labels.index(true_labels[len(predictions)-1])
        true_pattern = original_patterns[true_idx]
        correct_bits = np.sum(reconstructed == true_pattern)
        reconstruction_accuracies.append((correct_bits / len(true_pattern)) * 100)
    
    return predictions, reconstruction_accuracies

def compute_metrics_with_unknown(true_labels, predictions):
    """Compute metrics including 'Unknown' category."""
    all_labels = ['A', 'B', 'C', 'D', 'E', 'Unknown']
    cm = confusion_matrix(true_labels, predictions, labels=all_labels)
    
    # Filter out 'Unknown' for main metrics
    known_mask = [p != 'Unknown' for p in predictions]
    known_true = [t for t, m in zip(true_labels, known_mask) if m]
    known_pred = [p for p, m in zip(predictions, known_mask) if m]
    
    accuracy = accuracy_score(true_labels, predictions)
    if known_pred:
        precision = precision_score(known_true, known_pred, average='macro', zero_division=0)
        recall = recall_score(known_true, known_pred, average='macro', zero_division=0)
        f1 = f1_score(known_true, known_pred, average='macro', zero_division=0)
    else:
        precision, recall, f1 = 0, 0, 0
    
    return cm, accuracy, precision, recall, f1

def evaluate_subset(original_patterns, original_labels, noisy_examples, true_labels, subset_indices):
    """Evaluate using only a subset of training patterns."""
    subset_patterns = original_patterns[subset_indices]
    subset_labels = [original_labels[i] for i in subset_indices]
    
    # Create a mapping from subset label to its pattern index in subset
    subset_label_to_idx = {label: idx for idx, label in enumerate(subset_labels)}
    
    num_neurons = subset_patterns[0].size
    hopfield_net = HopfieldNetworkEnhanced(num_neurons, normalize_weights=True, use_async=True)
    hopfield_net.train(subset_patterns)
    
    predictions = []
    reconstruction_accuracies = []
    
    for i, noisy in enumerate(noisy_examples):
        reconstructed = hopfield_net.update(noisy, steps=20)
        
        # Classify based on highest correlation with subset patterns
        correlations = [correlation(reconstructed, orig) for orig in subset_patterns]
        max_corr_idx = np.argmax(correlations)
        
        predictions.append(subset_labels[max_corr_idx])
        
        # Reconstruction accuracy (compared to true original pattern, if it's in subset)
        true_letter = true_labels[i]
        if true_letter in subset_label_to_idx:
            true_idx_in_subset = subset_label_to_idx[true_letter]
            true_pattern = subset_patterns[true_idx_in_subset]
            correct_bits = np.sum(reconstructed == true_pattern)
            reconstruction_accuracies.append((correct_bits / len(true_pattern)) * 100)
    
    cm, accuracy, precision, recall, f1 = compute_metrics_with_unknown(true_labels, predictions)
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'reconstruction_accuracy': np.mean(reconstruction_accuracies) if reconstruction_accuracies else 0,
        'predictions': predictions
    }

def run_comparison_experiments(data_folder='data_set_low_noise'):
    """Run experiments with different approaches."""
    csv_files = ['A.csv', 'B.csv', 'C.csv', 'D.csv', 'E.csv']
    original_patterns, original_labels = load_patterns_from_csv(csv_files)
    noisy_examples, true_labels = load_noisy_examples(data_folder)
    
    print(f"Dataset: {data_folder}")
    print(f"Total test examples: {len(noisy_examples)}")
    
    results = {}
    
    # Experiment 1: All patterns with Hamming distance (baseline)
    print("\n=== Experiment 1: Baseline (All patterns, Hamming distance) ===")
    hopfield_net = HopfieldNetworkEnhanced(100, normalize_weights=True, use_async=True)
    hopfield_net.train(original_patterns)
    
    predictions_base = []
    for noisy in noisy_examples:
        reconstructed = hopfield_net.update(noisy, steps=20)
        distances = [hamming_distance(reconstructed, orig) for orig in original_patterns]
        predictions_base.append(original_labels[np.argmin(distances)])
    
    cm, acc, prec, rec, f1 = compute_metrics_with_unknown(true_labels, predictions_base)
    results['Baseline'] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    # Experiment 2: All patterns with correlation
    print("\n=== Experiment 2: All patterns, Correlation-based classification ===")
    predictions_corr, _ = evaluate_with_correlation(original_patterns, original_labels, noisy_examples, true_labels, hopfield_net)
    cm, acc, prec, rec, f1 = compute_metrics_with_unknown(true_labels, predictions_corr)
    results['Correlation'] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    # Experiment 3: Subset of most distinct patterns (A, C, E)
    print("\n=== Experiment 3: Subset (A, C, E - more distinct) ===")
    subset_indices = [0, 2, 4]  # A, C, E
    result_subset = evaluate_subset(original_patterns, original_labels, noisy_examples, true_labels, subset_indices)
    results['Subset (A,C,E)'] = result_subset
    print(f"Accuracy: {result_subset['accuracy']:.4f}, Precision: {result_subset['precision']:.4f}, Recall: {result_subset['recall']:.4f}, F1: {result_subset['f1']:.4f}")
    
    # Experiment 4: Subset (A, B, D - less overlapping)
    print("\n=== Experiment 4: Subset (A, B, D) ===")
    subset_indices = [0, 1, 3]  # A, B, D
    result_subset2 = evaluate_subset(original_patterns, original_labels, noisy_examples, true_labels, subset_indices)
    results['Subset (A,B,D)'] = result_subset2
    print(f"Accuracy: {result_subset2['accuracy']:.4f}, Precision: {result_subset2['precision']:.4f}, Recall: {result_subset2['recall']:.4f}, F1: {result_subset2['f1']:.4f}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(f"{'Method':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 60)
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
    
    return results

if __name__ == "__main__":
    # Run on low noise dataset first (should be easier)
    results = run_comparison_experiments('data_set_low_noise')
    
    # Also run on full noise dataset
    print("\n" + "="*60)
    results_full = run_comparison_experiments('data_set')

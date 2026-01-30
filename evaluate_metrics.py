import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from Nnhopfield import HopfieldNetwork

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
        labels.append(file.split('.')[0])  # e.g., 'A' from 'A.csv'
    return np.array(patterns), labels

def load_noisy_examples(data_folder):
    """Load all noisy examples from data_set folder."""
    examples = []
    true_labels = []
    for letra in ['A', 'B', 'C', 'D', 'E']:
        letra_folder = os.path.join(data_folder, f'letra_{letra}')
        for i in range(1, 11):  # Example_1 to Example_10
            file_path = os.path.join(letra_folder, f'Example_{i}.csv')
            if os.path.exists(file_path):
                data = pd.read_csv(file_path, header=None)
                matriz = data.values
                matriz = np.where(matriz == 0, -1, matriz)
                examples.append(matriz.flatten())
                true_labels.append(letra)
    return np.array(examples), true_labels

def evaluate_network(original_patterns, original_labels, noisy_examples, true_labels, hopfield_net):
    """Evaluate the network on noisy examples."""
    predictions = []
    hamming_noisy_reconstructed = []
    reconstruction_accuracies = []

    for noisy in noisy_examples:
        reconstructed = hopfield_net.update(noisy, steps=10)

        # Find predicted label by minimum Hamming distance to originals
        distances = [hamming_distance(reconstructed, orig) for orig in original_patterns]
        pred_idx = np.argmin(distances)
        pred_label = original_labels[pred_idx]
        predictions.append(pred_label)

        # Hamming distance between noisy and reconstructed
        hamming_noisy_reconstructed.append(hamming_distance(noisy, reconstructed))

        # Reconstruction accuracy: percentage of bits correct compared to original (true label)
        true_idx = original_labels.index(true_labels[len(predictions)-1])
        true_pattern = original_patterns[true_idx]
        correct_bits = np.sum(reconstructed == true_pattern)
        reconstruction_accuracies.append((correct_bits / len(true_pattern)) * 100)

    return predictions, hamming_noisy_reconstructed, reconstruction_accuracies

def compute_metrics(true_labels, predictions):
    """Compute classification metrics."""
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=['A', 'B', 'C', 'D', 'E'])

    # Overall metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')

    return cm, accuracy, precision, recall, f1

def main():
    # Load original patterns
    csv_files = ['A.csv', 'B.csv', 'C.csv', 'D.csv', 'E.csv']
    original_patterns, original_labels = load_patterns_from_csv(csv_files)

    # Train the network
    num_neurons = original_patterns[0].size
    hopfield_net = HopfieldNetwork(num_neurons)
    hopfield_net.train(original_patterns)

    # Load noisy examples
    data_folder = 'data_set_low_noise'
    noisy_examples, true_labels = load_noisy_examples(data_folder)

    # Evaluate
    predictions, hamming_noisy_reconstructed, reconstruction_accuracies = evaluate_network(
        original_patterns, original_labels, noisy_examples, true_labels, hopfield_net
    )

    # Compute metrics
    cm, accuracy, precision, recall, f1 = compute_metrics(true_labels, predictions)

    # Print results
    print("Confusion Matrix:")
    print(cm)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-Score (macro): {f1:.4f}")

    print(f"\nAverage Hamming Distance (Noisy to Reconstructed): {np.mean(hamming_noisy_reconstructed):.2f}")
    print(f"Average Reconstruction Accuracy (%): {np.mean(reconstruction_accuracies):.2f}%")

if __name__ == "__main__":
    main()
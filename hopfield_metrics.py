"""
Métricas para Redes de Hopfield
- Curva de Degradación
- Bit Accuracy
- Hamming Distance
- Pasos hasta Convergencia
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from Nnhopfield import HopfieldNetwork, aplicar_ruido

def load_original_patterns(letter_files):
    """Carga patrones originales desde archivos CSV."""
    patrones = []
    labels = []
    for file in letter_files:
        if os.path.exists(file):
            data = pd.read_csv(file, header=None)
            matriz = data.values
            matriz = np.where(matriz == 0, -1, matriz)
            patrones.append(matriz.flatten())
            labels.append(file.split('.')[0])
    return np.array(patrones), labels

def load_noisy_examples(data_folder):
    """Carga ejemplos ruidosos del dataset."""
    ejemplos = []
    true_labels = []
    for letra in ['E', 'M', 'Z', 'X', 'W']:
        carpeta = os.path.join(data_folder, f'letra_{letra}')
        for i in range(1, 11):
            archivo = os.path.join(carpeta, f'Example_{i}.csv')
            if os.path.exists(archivo):
                datos = pd.read_csv(archivo, header=None)
                matriz = datos.values
                ejemplos.append(matriz.flatten())
                true_labels.append(letra)
    return np.array(ejemplos), true_labels

def bit_accuracy(original, reconstructed):
    """Calcula el porcentaje de bits correctos."""
    return np.sum(original == reconstructed) / len(original) * 100

def hamming_distance(p1, p2):
    """Calcula la distancia Hamming."""
    return np.sum(p1 != p2)

def calculate_convergence_steps(pattern, net, max_steps=100):
    """Calcula los pasos hasta convergencia."""
    pattern = pattern.copy().reshape(-1, 1)
    prev_pattern = pattern.copy()
    
    for step in range(1, max_steps + 1):
        activation = np.dot(net.weights, pattern)
        pattern = np.where(activation >= 0, 1, -1)
        
        if np.allclose(pattern, prev_pattern):
            return step
        prev_pattern = pattern.copy()
    
    return max_steps

def calculate_energy(pattern, net):
    """Calcula la energía de Hopfield."""
    pattern = pattern.reshape(-1, 1)
    return -0.5 * pattern.T @ net.weights @ pattern

def run_degradation_analysis(letter_files, data_folder, output_prefix='hopfield_metrics'):
    """
    Ejecuta el análisis completo de métricas.
    
    Returns:
    - results: Diccionario con todas las métricas
    """
    # Cargar patrones originales
    patrones_originales, labels_originales = load_original_patterns(letter_files)
    
    # Entrenar red
    num_neurons = patrones_originales[0].size
    net = HopfieldNetwork(num_neurons)
    net.train(patrones_originales)
    
    # Cargar ejemplos ruidosos
    ejemplos_ruidosos, true_labels = load_noisy_examples(data_folder)
    
    # Niveles de ruido a analizar
    noise_levels = list(range(0, 51, 5))  # 0%, 5%, 10%, ..., 50%
    
    # Métricas por nivel de ruido
    results = {
        'noise_level': [],
        'bit_accuracy_mean': [],
        'bit_accuracy_std': [],
        'hamming_distance_mean': [],
        'hamming_distance_std': [],
        'convergence_steps_mean': [],
        'convergence_steps_std': [],
        'energy_mean': [],
        'accuracy_mean': [],  # Clasificación correcta
    }
    
    print("="*70)
    print("ANÁLISIS DE MÉTRICAS - RED DE HOPFIELD")
    print("="*70)
    print(f"Patrones de entrenamiento: {len(patrones_originales)}")
    print(f"Neuronas: {num_neurons}")
    print(f"Ejemplos de prueba: {len(ejemplos_ruidosos)}")
    print()
    
    for noise in noise_levels:
        bit_accs = []
        hamming_dists = []
        conv_steps = []
        energies = []
        correct_classifications = 0
        
        for i, ejemplo in enumerate(ejemplos_ruidosos):
            # Aplicar ruido controlado
            noisy = aplicar_ruido(ejemplo, noise)
            
            # Reconstruir
            reconstructed = net.update(noisy, steps=100)
            
            # Métricas de reconstrucción
            true_label = true_labels[i]
            true_idx = labels_originales.index(true_label)
            original = patrones_originales[true_idx]
            
            bit_accs.append(bit_accuracy(original, reconstructed))
            hamming_dists.append(hamming_distance(original, reconstructed))
            
            # Pasos de convergencia
            steps = calculate_convergence_steps(noisy, net)
            conv_steps.append(steps)
            
            # Energía
            energies.append(calculate_energy(reconstructed, net))
            
            # Clasificación
            distances = [hamming_distance(reconstructed, p) for p in patrones_originales]
            pred_idx = np.argmin(distances)
            if labels_originales[pred_idx] == true_label:
                correct_classifications += 1
        
        # Promediar métricas
        results['noise_level'].append(noise)
        results['bit_accuracy_mean'].append(np.mean(bit_accs))
        results['bit_accuracy_std'].append(np.std(bit_accs))
        results['hamming_distance_mean'].append(np.mean(hamming_dists))
        results['hamming_distance_std'].append(np.std(hamming_dists))
        results['convergence_steps_mean'].append(np.mean(conv_steps))
        results['convergence_steps_std'].append(np.std(conv_steps))
        results['energy_mean'].append(np.mean(energies))
        results['accuracy_mean'].append(correct_classifications / len(ejemplos_ruidosos) * 100)
        
        print(f"Ruido: {noise:3d}% | Bit Acc: {np.mean(bit_accs):5.1f}% | "
              f"Hamming: {np.mean(hamming_dists):5.1f} | Conv Steps: {np.mean(conv_steps):5.1f} | "
              f"Accuracy: {correct_classifications/len(ejemplos_ruidosos)*100:5.1f}%")
    
    return results, net

def plot_degradation_curve(results, output_file='degradation_curve.png'):
    """Genera gráfico de curva de degradación."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    noise = np.array(results['noise_level'])
    
    # 1. Bit Accuracy vs Ruido
    ax1 = axes[0, 0]
    ax1.errorbar(noise, results['bit_accuracy_mean'], 
                 yerr=results['bit_accuracy_std'], 
                 marker='o', capsize=3, color='blue')
    ax1.set_xlabel('Nivel de Ruido (%)')
    ax1.set_ylabel('Bit Accuracy (%)')
    ax1.set_title('Curva de Degradación - Bit Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # 2. Hamming Distance vs Ruido
    ax2 = axes[0, 1]
    ax2.errorbar(noise, results['hamming_distance_mean'],
                 yerr=results['hamming_distance_std'],
                 marker='s', capsize=3, color='red')
    ax2.set_xlabel('Nivel de Ruido (%)')
    ax2.set_ylabel('Distancia Hamming (bits)')
    ax2.set_title('Curva de Degradación - Distancia Hamming')
    ax2.grid(True, alpha=0.3)
    
    # 3. Pasos de Convergencia vs Ruido
    ax3 = axes[1, 0]
    ax3.errorbar(noise, results['convergence_steps_mean'],
                 yerr=results['convergence_steps_std'],
                 marker='^', capsize=3, color='green')
    ax3.set_xlabel('Nivel de Ruido (%)')
    ax3.set_ylabel('Pasos hasta Convergencia')
    ax3.set_title('Pasos de Convergencia vs Ruido')
    ax3.grid(True, alpha=0.3)
    
    # 4. Accuracy de Clasificación vs Ruido
    ax4 = axes[1, 1]
    ax4.plot(noise, results['accuracy_mean'], marker='d', linewidth=2, color='purple')
    ax4.fill_between(noise, results['accuracy_mean'], alpha=0.3, color='purple')
    ax4.set_xlabel('Nivel de Ruido (%)')
    ax4.set_ylabel('Accuracy de Clasificación (%)')
    ax4.set_title('Accuracy de Clasificación vs Ruido')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)
    
    plt.suptitle('Métricas de la Red de Hopfield', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"\nGráfico guardado: {output_file}")

def print_summary(results):
    """Imprime resumen de métricas."""
    print("\n" + "="*70)
    print("RESUMEN DE MÉTRICAS")
    print("="*70)
    
    # Encontrar punto donde accuracy cae bajo 50%
    accuracy = np.array(results['accuracy_mean'])
    noise = np.array(results['noise_level'])
    
    below_50 = np.where(accuracy < 50)[0]
    if len(below_50) > 0:
        failure_point = noise[below_50[0]]
        print(f"Punto de fallo (accuracy < 50%): {failure_point}% de ruido")
    else:
        print("No se alcanzó accuracy < 50% en el rango probado")
    
    # Encontrar índice para 10% de ruido
    noise_10_idx = None
    for i, n in enumerate(noise):
        if n == 10:
            noise_10_idx = i
            break
    
    # Mejor caso (0% ruido)
    print(f"\n📌 CASO 0% DE RUIDO (Sin ruido - Baseline):")
    print(f"  - Bit Accuracy: {results['bit_accuracy_mean'][0]:.1f}%")
    print(f"  - Hamming Distance: {results['hamming_distance_mean'][0]:.1f} bits")
    print(f"  - Pasos Convergencia: {results['convergence_steps_mean'][0]:.1f}")
    print(f"  - Accuracy: {results['accuracy_mean'][0]:.1f}%")
    
    # Caso intermedio (10% ruido)
    if noise_10_idx is not None:
        print(f"\n📌 CASO 10% DE RUIDO (Ruido bajo - Típico):")
        print(f"  - Bit Accuracy: {results['bit_accuracy_mean'][noise_10_idx]:.1f}%")
        print(f"  - Hamming Distance: {results['hamming_distance_mean'][noise_10_idx]:.1f} bits")
        print(f"  - Pasos Convergencia: {results['convergence_steps_mean'][noise_10_idx]:.1f}")
        print(f"  - Accuracy: {results['accuracy_mean'][noise_10_idx]:.1f}%")
    
    # Peor caso (50% ruido)
    print(f"\n📌 CASO 50% DE RUIDO (Ruido extremo - Fallo):")
    print(f"  - Bit Accuracy: {results['bit_accuracy_mean'][-1]:.1f}%")
    print(f"  - Hamming Distance: {results['hamming_distance_mean'][-1]:.1f} bits")
    print(f"  - Pasos Convergencia: {results['convergence_steps_mean'][-1]:.1f}")
    print(f"  - Accuracy: {results['accuracy_mean'][-1]:.1f}%")

def save_results_csv(results, filename='hopfield_metrics.csv'):
    """Guarda resultados en CSV."""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Resultados guardados: {filename}")

if __name__ == "__main__":
    # Archivos de letras
    letter_files = ['M.csv', 'W.csv', 'X.csv', 'E.csv', 'Z.csv']
    
    # Dataset de prueba
    data_folder = 'data_set_low_noise'
    
    if not os.path.exists(data_folder):
        print(f"Error: No se encontró el dataset '{data_folder}'")
        print("Generando dataset...")
        from generate_dataset import generate_dataset
        generate_dataset(10)
    
    # Ejecutar análisis
    results, net = run_degradation_analysis(letter_files, data_folder)
    
    # Generar gráficos
    plot_degradation_curve(results)
    
    # Imprimir resumen
    print_summary(results)
    
    # Guardar resultados
    save_results_csv(results)

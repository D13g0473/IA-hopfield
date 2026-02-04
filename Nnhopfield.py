import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def list_available_csv_files():
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    return csv_files

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        for pattern in patterns:
            pattern = pattern.reshape(-1, 1)
            self.weights += np.dot(pattern, pattern.T)
        np.fill_diagonal(self.weights, 0)

    def update(self, pattern, steps=5):
        pattern = pattern.reshape(-1, 1)
        for _ in range(steps):
            activation = np.dot(self.weights, pattern)
            pattern = np.where(activation >= 0, 1, -1)
        return pattern.flatten()

def create_sample_csv():
    letra_A = np.array([
        [-1,-1, 1, 1, 1, 1, 1, 1,-1,-1],
        [-1, 1,-1,-1,-1,-1,-1,-1, 1,-1],
        [ 1,-1,-1,-1,-1,-1,-1,-1,-1, 1],
        [ 1,-1,-1,-1,-1,-1,-1,-1,-1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1,-1,-1,-1,-1,-1,-1,-1,-1, 1],
        [ 1,-1,-1,-1,-1,-1,-1,-1,-1, 1],
        [ 1,-1,-1,-1,-1,-1,-1,-1,-1, 1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    ])

    letra_B = np.array([
        [ 1, 1, 1, 1, 1, 1,-1,-1,-1,-1],
        [ 1,-1,-1,-1,-1,-1, 1,-1,-1,-1],
        [ 1,-1,-1,-1,-1,-1, 1,-1,-1,-1],
        [ 1, 1, 1, 1, 1, 1,-1,-1,-1,-1],
        [ 1,-1,-1,-1,-1,-1, 1,-1,-1,-1],
        [ 1,-1,-1,-1,-1,-1, 1,-1,-1,-1],
        [ 1,-1,-1,-1,-1,-1, 1,-1,-1,-1],
        [ 1, 1, 1, 1, 1, 1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    ])

    letra_C = np.array([
        [-1,-1, 1, 1, 1, 1, 1, 1,-1,-1],
        [-1, 1,-1,-1,-1,-1,-1,-1, 1,-1],
        [ 1,-1,-1,-1,-1,-1,-1,-1,-1, 1],
        [ 1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [ 1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [ 1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1, 1,-1,-1,-1,-1,-1,-1, 1,-1],
        [-1,-1, 1, 1, 1, 1, 1, 1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    ])

    letra_D = np.array([
        [ 1, 1, 1, 1, 1,-1,-1,-1,-1,-1],
        [ 1,-1,-1,-1,-1, 1,-1,-1,-1,-1],
        [ 1,-1,-1,-1,-1,-1, 1,-1,-1,-1],
        [ 1,-1,-1,-1,-1,-1,-1, 1,-1,-1],
        [ 1,-1,-1,-1,-1,-1,-1, 1,-1,-1],
        [ 1,-1,-1,-1,-1,-1,-1, 1,-1,-1],
        [ 1,-1,-1,-1,-1,-1, 1,-1,-1,-1],
        [ 1,-1,-1,-1,-1, 1,-1,-1,-1,-1],
        [ 1, 1, 1, 1, 1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    ])

    letra_E = np.array([
        [ 1, 1, 1, 1, 1, 1, 1,-1,-1,-1],
        [ 1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [ 1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [ 1, 1, 1, 1, 1,-1,-1,-1,-1,-1],
        [ 1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [ 1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [ 1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [ 1, 1, 1, 1, 1, 1, 1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    ])

    letras = {
        "A.csv": letra_A,
        "B.csv": letra_B,
        "C.csv": letra_C,
        "D.csv": letra_D,
        "E.csv": letra_E,
    }

    for nombre, matriz in letras.items():
        df = pd.DataFrame(matriz)
        df.replace(0, -1, inplace=True)
        df.to_csv(nombre, index=False, header=False)

    print("Archivos CSV de ejemplo (A.csv, B.csv, C.csv, D.csv, E.csv) creados en el directorio actual.")

def mostrar_comparacion(original, ruidoso, reconstruido, titulo="Comparación patrones"):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis('off')

    axs[1].imshow(ruidoso, cmap='gray')
    axs[1].set_title("Con ruido")
    axs[1].axis('off')

    axs[2].imshow(reconstruido, cmap='gray')
    axs[2].set_title("Reconstruido")
    axs[2].axis('off')

    plt.suptitle(titulo)
    plt.show()

def aplicar_ruido(patron, porcentaje_ruido):
    """Invierte aleatoriamente un porcentaje de bits del patrón."""
    ruidoso = patron.copy()
    n_cambios = int(len(patron) * porcentaje_ruido / 100)
    indices = np.random.choice(len(patron), n_cambios, replace=False)
    for idx in indices:
        ruidoso[idx] = -ruidoso[idx]
    return ruidoso

def calculate_similarity(original, reconstructed):
    """Calcula el porcentaje de similitud entre dos patrones."""
    matches = np.sum(original == reconstructed)
    total = len(original)
    return (matches / total) * 100

def calculate_reconstruction_difference(noisy, reconstructed):
    """
    Calcula métricas de diferencia entre el patrón ruidoso (entrada) y el reconstruido.
    
    Returns:
    - hamming_diff: Número de bits que cambiaron durante la reconstrucción
    - change_percentage: Porcentaje de bits que cambiaron
    - flipped_indices: Índices de los bits que cambiaron
    """
    flipped_mask = noisy != reconstructed
    hamming_diff = np.sum(flipped_mask)
    change_percentage = (hamming_diff / len(noisy)) * 100
    flipped_indices = np.where(flipped_mask)[0]
    return hamming_diff, change_percentage, flipped_indices

def create_noisy_examples():
    """Genera ejemplos ruidosos de cada letra y los guarda en CSV en la carpeta test/."""
    test_folder = 'test'
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    csv_files = ['A.csv', 'B.csv', 'C.csv', 'D.csv', 'E.csv']
    noise_levels = [0, 10, 20, 30, 40, 50]  # Porcentajes de ruido

    for csv_file in csv_files:
        if os.path.exists(csv_file):
            data = pd.read_csv(csv_file, header=None)
            matriz = data.values
            matriz = np.where(matriz == 0, -1, matriz)
            original_pattern = matriz.flatten()

            for noise in noise_levels:
                noisy_pattern = aplicar_ruido(original_pattern, noise)
                noisy_matrix = noisy_pattern.reshape(10, 10)
                output_filename = os.path.join(test_folder, f'{csv_file.split(".")[0]}_ruido_{noise}.csv')
                pd.DataFrame(noisy_matrix).to_csv(output_filename, index=False, header=False)
                print(f"Ejemplo ruidoso guardado: {output_filename}")
        else:
            print(f"Archivo {csv_file} no encontrado.")

    print("Generación de ejemplos ruidosos completada.")

if __name__ == "__main__":
    print("Bienvenido al programa de la red de Hopfield")
    create_sample_csv()
    create_noisy_examples()

    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    while True:
        choice = input("¿Desea ejecutar el programa con la red de Hopfield? (si/no): ").strip().lower()
        if choice == "si":
            csv_files = list_available_csv_files()
            if not csv_files:
                print("No se encontraron archivos CSV en el directorio actual.")
                continue

            print("Archivos CSV disponibles:")
            for i, filename in enumerate(csv_files):
                print(f"{i + 1}. {filename}")

            seleccion_entrenar = input("Ingrese los números de los archivos CSV para entrenar (separados por coma): ")
            indices = []
            try:
                indices = [int(x.strip())-1 for x in seleccion_entrenar.split(',')]
                if not all(0 <= idx < len(csv_files) for idx in indices):
                    print("Uno o más números están fuera de rango.")
                    continue
            except:
                print("Entrada no válida.")
                continue

            patrones = []
            for idx in indices:
                data = pd.read_csv(csv_files[idx], header=None)
                matriz = data.values
                matriz = np.where(matriz == 0, -1, matriz)
                patrones.append(matriz.flatten())

            num_neurons = patrones[0].size
            hopfield_net = HopfieldNetwork(num_neurons)
            hopfield_net.train(np.array(patrones))
            print("Entrenamiento completado con los patrones seleccionados.")

            while True:
                print("\nPatrones disponibles para prueba:")
                for i, idx in enumerate(indices):
                    print(f"{i+1}. {csv_files[idx]}")
                prueba = input("Seleccione el número del patrón para probar la red o 'salir' para terminar: ").strip().lower()
                if prueba == "salir":
                    break
                try:
                    prueba_idx = int(prueba) - 1
                    if 0 <= prueba_idx < len(indices):
                        input_pattern = patrones[prueba_idx]
                        print("Patrón original:")
                        print(input_pattern.reshape(10,10))

                        porcentaje_ruido = input("Ingrese el porcentaje de ruido a aplicar (ej: 10): ")
                        try:
                            porcentaje_ruido = float(porcentaje_ruido)
                            if not (0 <= porcentaje_ruido <= 100):
                                print("Porcentaje inválido, debe estar entre 0 y 100.")
                                continue
                        except:
                            print("Entrada no válida para el porcentaje de ruido.")
                            continue

                        patron_ruidoso = aplicar_ruido(input_pattern, porcentaje_ruido)
                        print(f"Patrón con {porcentaje_ruido}% de ruido aplicado:")
                        print(patron_ruidoso.reshape(10,10))

                        reconstruido = hopfield_net.update(patron_ruidoso, steps=10)
                        print("Patrón reconstruido:")
                        print(reconstruido.reshape(10,10))
                        
                        # Calcular métricas de diferencia entre entrada (ruidosa) y reconstruida
                        hamming_diff, change_pct, flipped_idx = calculate_reconstruction_difference(patron_ruidoso, reconstruido)
                        similarity_to_original = calculate_similarity(input_pattern, reconstruido)
                        
                        # Mostrar métricas en el título del gráfico
                        title = f"Ruido: {porcentaje_ruido}% | Diferencia: {change_pct:.1f}% | Similitud: {similarity_to_original:.1f}%"
                        mostrar_comparacion(input_pattern.reshape(10,10),
                                            patron_ruidoso.reshape(10,10),
                                            reconstruido.reshape(10,10),
                                            titulo=title)

                        output_filename = os.path.join(results_folder, f'salida_{csv_files[indices[prueba_idx]].split(".")[0]}_ruido{int(porcentaje_ruido)}.csv')
                        pd.DataFrame(reconstruido.reshape(10,10)).to_csv(output_filename, index=False, header=False)
                        print(f"Resultado guardado en '{output_filename}'")
                    else:
                        print("Número fuera de rango.")
                except Exception as e:
                    print("Entrada no válida o error:", e)
        elif choice == "no":
            print("Muchas Gracias")
            break
        else:
            print("Opción no válida. Por favor, elija 'si' o 'no'.")

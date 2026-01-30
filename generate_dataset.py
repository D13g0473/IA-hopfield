import numpy as np
import pandas as pd
import os

def aplicar_ruido(patron, porcentaje_ruido):
    """Invierte aleatoriamente un porcentaje de bits del patrón."""
    ruidoso = patron.copy()
    n_cambios = int(len(patron) * porcentaje_ruido / 100)
    indices = np.random.choice(len(patron), n_cambios, replace=False)
    for idx in indices:
        ruidoso[idx] = -ruidoso[idx]
    return ruidoso

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
        "A": letra_A,
        "B": letra_B,
        "C": letra_C,
        "D": letra_D,
        "E": letra_E,
    }

    for nombre, matriz in letras.items():
        df = pd.DataFrame(matriz)
        df.to_csv(f"{nombre}.csv", index=False, header=False)

    print("Archivos CSV de ejemplo (A.csv, B.csv, C.csv, D.csv, E.csv) creados en el directorio actual.")

def generate_dataset(num_examples=10):
    """Genera un dataset con ejemplos ruidosos de cada letra en carpetas separadas."""
    # Asegurarse de que los archivos originales existan
    if not os.path.exists('A.csv'):
        create_sample_csv()

    letras = ['A', 'B', 'C', 'D', 'E']
    base_folder = 'data_set_low_noise'

    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    for letra in letras:
        letra_folder = os.path.join(base_folder, f'letra_{letra}')
        if not os.path.exists(letra_folder):
            os.makedirs(letra_folder)

        csv_file = f"{letra}.csv"
        data = pd.read_csv(csv_file, header=None)
        matriz = data.values
        matriz = np.where(matriz == 0, -1, matriz)
        original_pattern = matriz.flatten()

        for i in range(1, num_examples + 1):
            ruido = np.random.uniform(0, 10)  # Porcentaje aleatorio entre 0 y 10
            noisy_pattern = aplicar_ruido(original_pattern, ruido)
            noisy_matrix = noisy_pattern.reshape(10, 10)
            output_filename = os.path.join(letra_folder, f'Example_{i}.csv')
            pd.DataFrame(noisy_matrix).to_csv(output_filename, index=False, header=False)
            print(f"Ejemplo {i} para letra {letra} guardado en '{output_filename}' con {ruido:.2f}% de ruido.")

    print(f"Dataset generado con {num_examples} ejemplos por letra en la carpeta '{base_folder}'.")

if __name__ == "__main__":
    generate_dataset(10)  # Genera 10 ejemplos por letra
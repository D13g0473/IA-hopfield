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
    letras = {"A.csv": letra_A}
    for nombre, matriz in letras.items():
        df = pd.DataFrame(matriz)
        df.replace(0, -1, inplace=True)
        df.to_csv(nombre, index=False, header=False)

    print("Archivo CSV de ejemplo (A.csv) creado en el directorio actual.")

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

if __name__ == "__main__":
    print("Bienvenido al programa de la red de Hopfield")
    create_sample_csv()
    
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
                print("\nOpciones de prueba:")
               ## print("1. Probar con patrones entrenados")
                print("1. Probar con un archivo CSV externo")
                print("3. Salir")

                opcion = input("Seleccione una opción: ").strip().lower()

               ## if opcion == "x":
               ##     print("\nPatrones disponibles para prueba:")
               ##     for i, idx in enumerate(indices):
               ##         print(f"{i+1}. {csv_files[idx]}")
               ##     prueba = input("Seleccione el número del patrón para probar: ").strip().lower()
               ##     try:
               ##         prueba_idx = int(prueba) - 1
               ##         if 0 <= prueba_idx < len(indices):
               ##             input_pattern = patrones[prueba_idx]
               ##         else:
               ##             print("Número fuera de rango.")
               ##             continue
               ##     except:
               ##         print("Entrada no válida.")
               ##         continue
               ##
               ## el
                if opcion == "1":
                    externo = input("Ingrese el nombre del archivo CSV externo (ej: prueba.csv): ").strip()
                    if not os.path.exists(externo):
                        print("El archivo no existe en el directorio actual.")
                        continue
                    data = pd.read_csv(externo, header=None)
                    matriz = data.values
                    matriz = np.where(matriz == 0, -1, matriz)
                    input_pattern = matriz.flatten()
                    print(f"Archivo {externo} cargado correctamente.")

                elif opcion == "3":
                    break
                else:
                    print("Opción no válida.")
                    continue

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
                reconstruido = hopfield_net.update(patron_ruidoso, steps=10)

                mostrar_comparacion(input_pattern.reshape(10,10),
                                    patron_ruidoso.reshape(10,10),
                                    reconstruido.reshape(10,10),
                                    titulo=f"Original vs Ruido {porcentaje_ruido}% vs Reconstruido")

                output_filename = os.path.join(results_folder, f'salida_prueba_ruido{int(porcentaje_ruido)}.csv')
                pd.DataFrame(reconstruido.reshape(10,10)).to_csv(output_filename, index=False, header=False)
                print(f"Resultado guardado en '{output_filename}'")

        elif choice == "no":
            print("Muchas Gracias")
            break
        else:
            print("Opción no válida. Por favor, elija 'si' o 'no'.")

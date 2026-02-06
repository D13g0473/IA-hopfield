import numpy as np
import pandas as pd
import os

def crear_patron_10x10(letra):
    """Crea patrones de letras en formato 10x10."""
    
    if letra == 'A':
        patron = [
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
        ]
    elif letra == 'B':
        patron = [
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
        ]
    elif letra == 'C':
        patron = [
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
        ]
    elif letra == 'D':
        patron = [
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
        ]
    elif letra == 'E':
        patron = [
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
        ]
    elif letra == 'M':
        # M: Dos diagonales hacia el centro
        patron = [
            [ 1,-1,-1,-1,-1,-1,-1,-1, 1,-1],
            [ 1, 1,-1,-1,-1,-1,-1, 1, 1,-1],
            [-1, 1, 1,-1,-1,-1, 1, 1,-1,-1],
            [-1,-1, 1, 1,-1,-1, 1, 1,-1,-1],
            [-1,-1,-1, 1, 1, 1, 1,-1,-1,-1],
            [-1,-1,-1, 1, 1, 1, 1,-1,-1,-1],
            [-1,-1,-1, 1, 1, 1, 1,-1,-1,-1],
            [-1,-1,-1,-1, 1, 1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        ]
    elif letra == 'N':
        # N: Dos verticales con diagonal
        patron = [
            [ 1,-1,-1,-1,-1,-1,-1,-1, 1,-1],
            [ 1, 1,-1,-1,-1,-1,-1,-1, 1,-1],
            [ 1, 1, 1,-1,-1,-1,-1, 1, 1,-1],
            [-1, 1, 1, 1,-1,-1,-1, 1, 1,-1],
            [-1,-1, 1, 1, 1,-1,-1, 1, 1,-1],
            [-1,-1,-1, 1, 1, 1,-1, 1, 1,-1],
            [-1,-1,-1,-1, 1, 1, 1, 1, 1,-1],
            [-1,-1,-1,-1,-1, 1, 1, 1, 1,-1],
            [-1,-1,-1,-1,-1,-1, 1, 1, 1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        ]
    elif letra == 'O':
        # O: Círculo perfecto
        patron = [
            [-1,-1, 1, 1, 1, 1, 1, 1,-1,-1],
            [-1, 1,-1,-1,-1,-1,-1,-1, 1,-1],
            [ 1,-1,-1,-1,-1,-1,-1,-1,-1, 1],
            [ 1,-1,-1,-1,-1,-1,-1,-1,-1, 1],
            [ 1,-1,-1,-1,-1,-1,-1,-1,-1, 1],
            [ 1,-1,-1,-1,-1,-1,-1,-1,-1, 1],
            [ 1,-1,-1,-1,-1,-1,-1,-1,-1, 1],
            [-1, 1,-1,-1,-1,-1,-1,-1, 1,-1],
            [-1,-1, 1, 1, 1, 1, 1, 1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        ]
    elif letra == 'W':
        # W: Doble V
        patron = [
            [ 1,-1,-1,-1,-1,-1,-1,-1, 1,-1],
            [ 1,-1,-1,-1,-1,-1,-1,-1, 1,-1],
            [ 1, 1,-1,-1,-1,-1,-1, 1, 1,-1],
            [ 1, 1,-1,-1,-1,-1,-1, 1, 1,-1],
            [-1, 1, 1,-1,-1,-1, 1, 1,-1,-1],
            [-1, 1, 1,-1,-1,-1, 1, 1,-1,-1],
            [-1,-1, 1, 1,-1, 1, 1,-1,-1,-1],
            [-1,-1, 1, 1,-1, 1, 1,-1,-1,-1],
            [-1,-1,-1, 1, 1, 1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        ]
    elif letra == 'X':
        # X: Dos diagonales cruzando
        patron = [
            [ 1,-1,-1,-1,-1,-1,-1,-1, 1,-1],
            [-1, 1,-1,-1,-1,-1,-1, 1,-1,-1],
            [-1,-1, 1,-1,-1,-1, 1,-1,-1,-1],
            [-1,-1,-1, 1, 1, 1,-1,-1,-1,-1],
            [-1,-1,-1, 1, 1, 1,-1,-1,-1,-1],
            [-1,-1,-1, 1, 1, 1,-1,-1,-1,-1],
            [-1,-1,-1, 1, 1, 1,-1,-1,-1,-1],
            [-1,-1, 1,-1,-1,-1, 1,-1,-1,-1],
            [-1, 1,-1,-1,-1,-1,-1, 1,-1,-1],
            [ 1,-1,-1,-1,-1,-1,-1,-1, 1,-1],
        ]
    elif letra == 'Z':
        # Z: Zigzag horizontal
        patron = [
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [-1,-1,-1,-1,-1,-1,-1,-1, 1,-1],
            [-1,-1,-1,-1,-1,-1,-1, 1,-1,-1],
            [-1,-1,-1,-1,-1,-1, 1,-1,-1,-1],
            [-1,-1,-1,-1,-1, 1,-1,-1,-1,-1],
            [-1,-1,-1,-1, 1,-1,-1,-1,-1,-1],
            [-1,-1,-1, 1,-1,-1,-1,-1,-1,-1],
            [-1,-1, 1,-1,-1,-1,-1,-1,-1,-1],
            [ 1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    else:
        raise ValueError(f"Letra '{letra}' no soportada")
    
    return np.array(patron)

def aplicar_ruido(patron, porcentaje_ruido):
    """Invierte aleatoriamente un porcentaje de bits del patrón."""
    ruidoso = patron.copy()
    n_cambios = int(len(patron) * porcentaje_ruido / 100)
    indices = np.random.choice(len(patron), n_cambios, replace=False)
    for idx in indices:
        ruidoso[idx] = -ruidoso[idx]
    return ruidoso

class HopfieldNetwork:
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
    return np.sum(p1 != p2)

def correlation_matrix(patterns):
    """Calcula matriz de correlación entre patrones."""
    return np.corrcoef(patterns)

def evaluar_red(patrones, labels, ejemplos, labels_true):
    """Evalúa la red."""
    num_neurons = patrones[0].size
    red = HopfieldNetwork(num_neurons, normalize_weights=True)
    red.train(patrones)
    
    predicciones = []
    for ejemplo in ejemplos:
        reconstruido = red.update(ejemplo, steps=20)
        distancias = [hamming_distance(reconstruido, p) for p in patrones]
        pred_idx = np.argmin(distancias)
        predicciones.append(labels[pred_idx])
    
    accuracy = sum(p == v for p, v in zip(predicciones, labels_true)) / len(predicciones)
    return accuracy, predicciones

def main():
    # Letras originales (correlacionadas)
    letras_originales = ['A', 'B', 'C', 'D', 'E']
    
    # Letras más distintivas seleccionadas
    letras_distintas = ['M', 'N', 'O', 'W', 'X', 'Z']
    
    # Combinación balanceada
    letras_combinadas = ['A', 'B', 'M', 'N', 'O', 'W', 'X', 'Z']
    
    print("="*70)
    print("ANÁLISIS DE LETRAS DISTINTIVAS")
    print("="*70)
    
    # Cargar patrones
    patrones_originales = {}
    for letra in letras_originales + letras_distintas:
        patrones_originales[letra] = crear_patron_10x10(letra).flatten()
    
    # Mostrar matriz de correlación
    print("\n--- Correlación entre letras (valores cercanos a 1 = muy similares) ---")
    todas_letras = letras_originales + letras_distintas
    patrones_todos = np.array([patrones_originales[l] for l in todas_letras])
    corr = correlation_matrix(patrones_todos)
    
    print("\nCorrelación promedio por letra:")
    for i, letra in enumerate(todas_letras):
        corr_promedio = np.mean([abs(corr[i,j]) for j in range(len(todas_letras)) if i != j])
        print(f"  {letra}: {corr_promedio:.3f}")
    
    # Generar dataset de prueba
    def generar_dataset(letras, num_ejemplos=5, ruido_max=30):
        ejemplos = []
        labels = []
        for letra in letras:
            patron = patrones_originales[letra].reshape(10, 10)
            for i in range(num_ejemplos):
                ruido = np.random.uniform(0, ruido_max)
                ruidoso = aplicar_ruido(patron.flatten(), ruido)
                ejemplos.append(ruidoso)
                labels.append(letra)
        return np.array(ejemplos), labels
    
    # Evaluar cada conjunto
    print("\n" + "="*70)
    print("RESULTADOS DE CLASIFICACIÓN")
    print("="*70)
    
    for nombre, letras in [("Originales (A-E)", letras_originales), 
                           ("Distintivas (M,N,O,W,X,Z)", letras_distintas),
                           ("Combinadas", letras_combinadas)]:
        print(f"\n--- {nombre} ---")
        patrones = np.array([patrones_originales[l] for l in letras])
        ejemplos, labels_true = generar_dataset(letras, num_ejemplos=5, ruido_max=30)
        accuracy, predicciones = evaluar_red(patrones, letras, ejemplos, labels_true)
        print(f"Letras: {letras}")
        print(f"Accuracy: {accuracy*100:.1f}%")
    
    # Guardar patrones distintivos como CSV
    for letra in letras_distintas:
        matriz = crear_patron_10x10(letra)
        df = pd.DataFrame(matriz)
        df.to_csv(f'{letra}.csv', index=False, header=False)
    
    print(f"\nPatrones distintivos guardados: {', '.join(letras_distintas)}.csv")

if __name__ == "__main__":
    main()

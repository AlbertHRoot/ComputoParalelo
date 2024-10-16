import pandas as pd
import itertools
import multiprocessing
import time
import numpy as np
import logging
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

logging.basicConfig(filename='RedesN_logNucleos.log', level=logging.INFO, format='%(message)s')

# Método para nivelación de cargas
def nivelacion_cargas(D, n_p):
    s = len(D) % n_p
    n_D = D[:s]
    t = int((len(D) - s) / n_p)
    out = []
    temp = []
    for i in D[s:]:
        temp.append(i)
        if len(temp) == t:
            out.append(temp)
            temp = []
    for i in range(len(n_D)):
        out[i].append(n_D[i])
    return out

# Cargar el dataset de vinos
data = pd.read_csv('WineQT.csv')  
X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values   
# Normalizar las características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Parámetros para Redes Neuronales
param_grid_nn = {
    'hidden_layer_sizes': [(32,), (64,), (128,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01]
}

# Generar combinaciones de parámetros para RN
keys_nn, values_nn = zip(*param_grid_nn.items())
combinations_nn = [dict(zip(keys_nn, v)) for v in itertools.product(*values_nn)]

# Función para evaluar los modelos RN con diferentes hiperparámetros
def evaluate_nn_set(hyperparameter_set, lock):
    """ Evaluar un conjunto de hiperparámetros para RN """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    for s in hyperparameter_set:
        model = MLPClassifier(hidden_layer_sizes=s['hidden_layer_sizes'], 
                              activation=s['activation'], 
                              solver=s['solver'], 
                              alpha=s['alpha'], 
                              max_iter=200, 
                              random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        with lock:
            log_message = f"Accuracy con Redes Neuronales y parametros {s}: {accuracy}"
            print(log_message)
            logging.info(log_message)

if __name__ == '__main__':
    # Número de hilos
    N_THREADS = 8
    splits = nivelacion_cargas(combinations_nn, N_THREADS)
    lock = multiprocessing.Lock()
    threads = []
    # Iniciar el proceso para RN
    for i in range(N_THREADS):
        threads.append(multiprocessing.Process(target=evaluate_nn_set, args=(splits[i], lock)))
    # Ejecutar los hilos
    start_time = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    finish_time = time.perf_counter()
    print(f"RN finished in {finish_time - start_time} seconds")
    logging.info(f"RN finished in {finish_time - start_time} seconds")

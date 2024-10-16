import pandas as pd
import itertools
import multiprocessing
import time
import numpy as np
import logging
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Configurar el logger
logging.basicConfig(filename='svm_log.log', level=logging.INFO, format='%(message)s')

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
# Asignar variables de entrada y salida
X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values   
# Normalizar las características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Parámetros para SVM
param_grid_svm = { 
    'C': [0.1, 1, 10, 100], 
    'kernel': ['linear', 'poly', 'rbf'], 
    'gamma': ['scale', 'auto'], 
    'coef0': [0.0, 0.1, 0.5, 1.0]
}

# Generar combinaciones de parámetros para SVM
keys_svm, values_svm = zip(*param_grid_svm.items())
combinations_svm = [dict(zip(keys_svm, v)) for v in itertools.product(*values_svm)]

# Función para evaluar los modelos SVM con diferentes hiperparámetros
def evaluate_svm_set(hyperparameter_set, lock):
    """ Evaluar un conjunto de hiperparámetros para SVM """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    for s in hyperparameter_set:
        clf = SVC()
        clf.set_params(C=s['C'], kernel=s['kernel'], gamma=s['gamma'], coef0=s['coef0'])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        with lock:
            log_message = f"Accuracy con SVM y parametros {s}: {accuracy}"
            print(log_message)
            logging.info(log_message)

if __name__ == '__main__':
    # Número de hilos
    N_THREADS = 8
    splits = nivelacion_cargas(combinations_svm, N_THREADS)
    lock = multiprocessing.Lock()
    threads = []
    # Iniciar el proceso para SVM
    for i in range(N_THREADS):
        threads.append(multiprocessing.Process(target=evaluate_svm_set, args=(splits[i], lock)))
    # Ejecutar los hilos
    start_time = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    finish_time = time.perf_counter()
    print(f"SVM finished in {finish_time - start_time} seconds")
    logging.info(f"SVM finished in {finish_time - start_time} seconds")

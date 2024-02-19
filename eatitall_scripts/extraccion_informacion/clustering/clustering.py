from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans



def inertia_of_kmeans(df,maxClusters=15,n_clusters=4,init='k-means++',n_init = 10,max_iter=500,tol=0.0001,random_state= 111,algorithm='elkan'):
    inertia = []  # Lista para almacenar los valores de inercia
    # La inercia es la suma de las distancias al cuadrado de cada muestra a su centroide más cercano

    # Bucle para probar diferentes valores de k (número de clústeres)
    for n in range(1 , maxClusters):
        algorithm = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                        tol=0.0001, random_state=111, algorithm='elkan')
        algorithm.fit(df)
        # Almacena la inercia del modelo en la lista
        inertia.append(algorithm.inertia_)
    return inertia

def kmeans_algorithm(df,n_clusters=4,init='k-means++',n_init = 10,max_iter=500,tol=0.0001,random_state= 111,algorithm='elkan'):
    algorithm = (KMeans(n_clusters,init,n_init,max_iter,tol,random_state,algorithm)) 
    ## Aplica el algoritmo de k-means al conjunto de datos X1 para realizar el clustering. 
    ## Esto ajusta los centroides de los clústeres y asigna cada punto a un clúster.
    algorithm.fit(df)
    ## Cada etiqueta indica a qué clúster pertenece el punto.
    labels = algorithm.labels_
    ## Después de ajustar el modelo, esta línea guarda las coordenadas de los centroides de los clústeres en centroids1.
    centroids = algorithm.cluster_centers_
    return labels,centroids

def plot_inertia(inertia,maxClusters=15):
    # Crea una figura con un tamaño específico
    plt.figure(1, figsize=(15, 8))  
    # Gráfica de dispersión ('o') y línea ('-') para visualizar la relación entre k y la inercia
    plt.plot(np.arange(1, maxClusters), inertia, 'o')
    plt.plot(np.arange(1, maxClusters), inertia, '-', alpha=0.5)
    # Etiqueta del eje x
    plt.xlabel('Number of Clusters')
    # Etiqueta del eje y
    plt.ylabel('Inertia')
    # Muestra el gráfico
    plt.show()
    return

def plot_clusters(df,x_index,y_index):
    # x_index e y_index son las dos variables para visualizar

    # Crear un mapa de colores basado en las etiquetas de los clusters
    colors = plt.cm.viridis(np.linspace(0, 1, len(centroids)))

    # Gráfico de dispersión de los datos
    for i, color in enumerate(colors):
        plt.scatter(df.loc[labels == i, df.columns[x_index]], 
                    df.loc[labels == i, df.columns[y_index]], 
                    color=color, marker='o', edgecolor='k', label=f'Cluster {i}')

    # Gráfico de los centroides
    plt.scatter(centroids[:, x_index], centroids[:, y_index], c=colors, marker='x', s=100, label='Centroides')

    # Etiquetas y título
    plt.xlabel(df.columns[x_index])
    plt.ylabel(df.columns[y_index])
    plt.title('K-Means Clustering con 2 Variables')
    plt.legend()

    # Mostrar el gráfico
    plt.show()
    return

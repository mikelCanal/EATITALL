import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def buscar_correlaciones(df:pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la matriz de correlación de Pearson para todas las variables en un DataFrame de pandas.

    :param df: DataFrame de pandas.
    :return: DataFrame de pandas con la matriz de correlación.
    """
    # Calcula la matriz de correlación
    correlaciones = df.corr(method='pearson')

    return correlaciones

def encontrar_variables_mas_relacionadas(correlaciones, num_variables, umbral_maximo):
    """
    Encuentra las N variables más relacionadas dentro de un umbral en una matriz de correlación,
    eliminando relaciones simétricas.

    :param correlaciones: DataFrame de pandas con la matriz de correlación.
    :param num_variables: Número de pares de variables a devolver.
    :param umbral_maximo: Umbral máximo de correlación permitido.
    :return: DataFrame con los pares de variables más relacionadas y sus valores de correlación.
    """
    # Crear un DataFrame a partir de la matriz de correlación, excluyendo la diagonal principal y relaciones simétricas
    correlaciones_melt = correlaciones.where(np.triu(np.ones(correlaciones.shape), k=1).astype(np.bool_)).stack().reset_index()
    correlaciones_melt.columns = ['Variable1', 'Variable2', 'Correlacion']

    # Filtrar por el umbral máximo y ordenar
    correlaciones_filtradas = correlaciones_melt[correlaciones_melt['Correlacion'].abs() <= umbral_maximo]
    correlaciones_top_n = correlaciones_filtradas.sort_values(by='Correlacion', ascending=False).head(num_variables)
    return correlaciones_top_n


def visualizar_correlaciones(dataset):
    """
    Visualiza la matriz de correlación de un DataFrame de pandas.

    :param dataset: DataFrame de pandas.
    """
    # Calcular la matriz de correlación
    correlaciones = dataset.corr()

    # Crear un mapa de calor
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlaciones, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Mapa de Calor de Correlaciones')
    plt.show()
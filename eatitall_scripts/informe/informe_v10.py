import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import plot_tree
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import json
from io import BytesIO


# Título de la aplicación
st.title("Análisis de Clustering con K-means")

# Cargar datos
path='/home/eatitall_scripts/archivos/datos_semibuenos_1_completo_v2.csv'
# df = pd.read_csv('./archivos/datos_gpt_con_40_ejemplos_reglas v12.csv')

uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

if uploaded_file is not None:
    # Para leer el archivo CSV
    df = pd.read_csv(uploaded_file)
    # Mostrar el dataframe en la aplicación
else:
    st.write("Por favor, carga un archivo CSV.")

# st.write("Datos cargados (normalizados y pasados a variables numéricas)", df)

### Aplicamos el algoritmo de clustering


def linear_regresion(df,variable='cluster'):
    X = df.drop(variable, axis=1)  # Todas las columnas excepto la variable objetivo
    y = df[variable]  # Variable objetivo

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.coef_

    # Evaluar el modelo
    # print("Coeficientes:", model.coef_)
    # print("Intercepto:", model.intercept_)
    # print("Error cuadrático medio (MSE):", mean_squared_error(y_test, model.predict(X_test)))
    # print("Coeficiente de determinación (R^2):", r2_score(y_test, model.predict(X_test)))

    # Mostrar la importancia de cada característica
    feature_importance = pd.DataFrame(model.coef_, X.columns, columns=['coeficiente'])
    # print(feature_importance.sort_values(by='coeficiente', ascending=False))
    return feature_importance


def kmeans_algorithm(df,n_clusters=4,init='k-means++',n_init = 10,max_iter=500,tol=0.0001,random_state= 111,algorithm='elkan'):
    algorithm = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state, algorithm=algorithm)
    ## Aplica el algoritmo de k-means al conjunto de datos X1 para realizar el clustering. 
    ## Esto ajusta los centroides de los clústeres y asigna cada punto a un clúster.
    algorithm.fit(df)
    ## Cada etiqueta indica a qué clúster pertenece el punto.
    labels = algorithm.labels_
    ## Después de ajustar el modelo, esta línea guarda las coordenadas de los centroides de los clústeres en centroids1.
    centroids = algorithm.cluster_centers_
    return labels,centroids

# K-means clustering
k_kmeans = st.slider("Seleccione el número de clusters", 1, 10, 3)
labels_kmeans,centroids_kmeans=kmeans_algorithm(df,n_clusters=k_kmeans,init='k-means++',n_init = 10,max_iter=500,tol=0.0001,random_state= 111,algorithm='elkan')
df['cluster'] = labels_kmeans
# st.write(f"Resultados del clustering con {k} clusters", df)

# output_directory = './archivos/cluster_outputs'
# os.makedirs(output_directory, exist_ok=True)

st.title("Clústers normalizados")
# st.write("""Un valor de -100 significa que no hay datos.""")

st.write("""Normalizado significa que las variables binarias (o de un conjunto de números) tiene el valor de 0 o 1 (o lo que corresponda). Se ha aproximado el resultado promedio.""")

# Identificamos variables binarias.
def detect_binary_columns(dataframe):
    binary_columns = []
    for column in dataframe.columns:
        unique_values = dataframe[column].unique()
        if set(unique_values).issubset({0, 1}):
            binary_columns.append(column)
    return binary_columns

binary_columns_full_df = detect_binary_columns(df)
binary_columns_full_df.append('sexo')
binary_columns_full_df.append('lugar_nac')
binary_columns_full_df.append('dm')
binary_columns_full_df.append('ic_tipo')
binary_columns_full_df.append('careme_estadio')
binary_columns_full_df.append('rockwood') #Es binaria pero no de 0,1 sino 1,2
# print(f"Las columnas binarias son: {binary_columns_full_df}")

# Cargamos el archivo de configuración
# config_path='./archivos/config_perfiles_clinicos.json'
# with open(config_path, 'r') as archivo:
#     config_perfiles_clinicos = json.load(archivo)

# Aplicamos los valores definidos en el archivo de configuración para cada variable binaria 

# Crear dataframes para cada algoritmo con los centroides y el valor del cluster
def create_centroid_dataframe(df, centroids, algorithm_name):
    df_new=df.drop(columns=['cluster'])
    centroid_df = pd.DataFrame(centroids, columns=df_new.columns)
    centroid_df['cluster'] = range(len(centroids))
    centroid_df['algorithm'] = algorithm_name
    return centroid_df
df_kmeans_centroids = create_centroid_dataframe(df, centroids_kmeans, 'kmeans')
# st.write(f"Centroides (valores promedio) de los cluster",df_kmeans_centroids)

df_kmeans_centroids_v2=df_kmeans_centroids
for columna in binary_columns_full_df:
    # nombre_umbral='umbral_'+columna
    for k in range(0,len(df_kmeans_centroids_v2)):
        if columna=='sexo':
            if df_kmeans_centroids_v2[columna][k]>=1.5:
                df_kmeans_centroids_v2[columna][k]=2
            else:
                df_kmeans_centroids_v2[columna][k]=1
        elif columna=='lugar_nac' or columna=='dm' or columna=='ic_tipo' or columna=='careme_estadio' or columna=='rockwood':
            df_kmeans_centroids_v2[columna][k]=df_kmeans_centroids_v2[columna][k].round(0)
        else:
            if df_kmeans_centroids_v2[columna][k]>=0.5:
                df_kmeans_centroids_v2[columna][k]=1    
            else:
                df_kmeans_centroids_v2[columna][k]=0 

#Redondeando a dos decimales
df_kmeans_centroids_v2 = df_kmeans_centroids_v2.apply(lambda x: round(x, 2) if x.dtype == 'float' else x)

# df_kmeans_centroids_v2.to_excel('perfiles_clinicos_v7.xlsx', index=False)
df_kmeans_centroids_v2['f_dx_dm'] = pd.to_datetime(df_kmeans_centroids_v2['f_dx_dm'], unit='s')
df_kmeans_centroids_v2['f_dx_dm'] = df_kmeans_centroids_v2['f_dx_dm'].dt.strftime('%Y-%m-%d')

# st.write("Datos procesados con umbrales aplicados:")
st.dataframe(df_kmeans_centroids_v2)  

# st.title('Descargar clusters en formato Excel')

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    # writer.save()
    processed_data = output.getvalue()
    return processed_data

val = to_excel(df_kmeans_centroids_v2)
st.download_button(label=f'Descargar clústers normalizados como Excel',
                    data=val,
                    file_name=f'clusters_norm.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# Crear un diccionario para almacenar DataFrames de cada cluster
cluster_dfs = {}

# df['f_dx_dm'] = pd.to_datetime(df['f_dx_dm'], unit='s')
# df['f_dx_dm'] = df['f_dx_dm'].dt.strftime('%Y-%m-%d')

# Iterar sobre cada cluster y crear un DataFrame separado
for cluster_label in range(len(df_kmeans_centroids_v2)):
    cluster_dfs[cluster_label] = df[df['cluster'] == cluster_label]

# Mostrar los DataFrames de cada cluster y guardarlos en local
for cluster_label, cluster_df in cluster_dfs.items():
    # if st.button('Descargar Excel'):
    val = to_excel(cluster_df)
    st.download_button(label=f'Descargar listado de pacientes del clúster (perfil clínico) {cluster_label} como Excel',
                    data=val,
                    file_name=f'cluster_{cluster_label}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    # print(f"DataFrame del Cluster {cluster_label}:\n")
    # print(cluster_df)
    # print("\n\n")
    # Guardar el DataFrame en un archivo Excel
    # output_path = os.path.join(output_directory, f'cluster_{cluster_label}.xlsx')
    # cluster_df.to_excel(output_path, index=False)
    # st.write(f"Resultados del cluster {cluster_label+1}",cluster_df)
    # print(f'El DataFrame del Cluster {cluster_label} ha sido guardado en {output_path}')

# df['f_dx_dm'] = pd.to_datetime(df['f_dx_dm'])
# df['f_dx_dm'] = df['f_dx_dm'].apply(lambda x: x.timestamp())

# ANÁLISIS DE COMBINACIONES EN EL DATASET
st.title("Generación de perfiles clínicos")

def encontrar_combinaciones(df, prefijos_interes, n_cluster, n=4):
    """
    Encuentra las combinaciones más comunes de 0s y 1s en las variables de interés,
    considerando registros que tienen al menos una variable con valor 1 en cada categoría.
    
    Parámetros:
    df (pd.DataFrame): DataFrame con los datos.
    prefijos_interes (dict): Diccionario con los prefijos de las columnas de interés como claves y los nombres amigables como valores.
    n (int): Número de combinaciones más comunes a retornar.
    
    Retorna:
    str: Descripción de las combinaciones más comunes.
    """
    # Filtrar las columnas que pertenecen a los prefijos de interés
    cols_interes = [col for col in df.columns if any(col.startswith(pref) for pref in prefijos_interes.keys())]
    
    # Crear un DataFrame con las columnas seleccionadas
    df_interes = df[cols_interes]
    
    # Crear un DataFrame que combine las columnas de interés en una representación legible
    combinaciones = pd.DataFrame(index=df.index)
    for prefijo, nombre in prefijos_interes.items():
        # Filtrar columnas de la categoría actual
        cols_categoria = [col for col in cols_interes if col.startswith(prefijo)]
        # Crear una columna para la categoría con los nombres de las columnas con valor 1
        combinaciones[nombre] = df[cols_categoria].apply(lambda row: ', '.join(row.index[row == 1].str.replace(prefijo, '')), axis=1)
    
    # Contar las filas que no tienen datos (todas las variables son 0) para cada categoría
    sin_datos = {}
    for prefijo, nombre in prefijos_interes.items():
        cols_categoria = [col for col in cols_interes if col.startswith(prefijo)]
        sin_datos[nombre] = (df[cols_categoria].sum(axis=1) == 0).mean() * 100

    # Filtrar filas donde al menos una columna por categoría no sea ''
    combinaciones = combinaciones[(combinaciones != '').all(axis=1)]
    
    # Contar combinaciones únicas
    combinaciones_contadas = combinaciones.value_counts().head(n).reset_index()
    combinaciones_contadas.columns = list(combinaciones.columns) + ['Frecuencia']
    
    # Calcular porcentaje
    combinaciones_contadas['Porcentaje'] = (combinaciones_contadas['Frecuencia'] / len(df)) * 100
    
    # Generar descripciones de las combinaciones más comunes
    descripciones = []
    pc=1
    for _, row in combinaciones_contadas.iterrows():
        descripcion = f"PC_{pc}: el {row['Porcentaje']:.2f}% de los registros tienen " + ", ".join(
            [f"{cat} {row[cat]}" for cat in prefijos_interes.values() if row[cat]]
        )
        descripciones.append(descripcion)
        pc+=1
    
    # Agregar descripciones de categorías sin datos si no es 0%
    for nombre, porcentaje in sin_datos.items():
        if porcentaje > 0:
            descripciones.append(f"el {porcentaje:.2f}% de los registros no tienen datos sobre la categoría {nombre}")
    
    # st.dataframe(combinaciones)
    # st.dataframe(combinaciones_contadas)

    # Crear botones para descargar las filas correspondientes a cada combinación
    combinaciones_df_list = []
    df['subcluster']=0
    df_resultado = pd.DataFrame()
    for i, row in combinaciones_contadas.iterrows():
        # Filtrar las filas que corresponden a la combinación actual
        # df['subcluster']=n*n_cluster+i
        combinacion_actual = (combinaciones == row[list(prefijos_interes.values())]).all(axis=1)
        df_comb = df[combinacion_actual]
        df_comb['subcluster']=n*n_cluster+i
        combinaciones_df_list.append(df_comb)
        df_resultado = pd.concat([df_resultado, df_comb])
        # Convertir a CSV
        csv = df_comb.to_csv(index=False)
        b64 = convert_df_to_bytes(csv)

        # Crear el botón de descarga para esta combinación
        st.download_button(
            label=f"Descargar perfil clínico {i + 1} (PC_{i+1})",
            data=b64,
            file_name=f"combinacion_{i + 1}.csv",
            mime='text/csv',
            key=f"download_button_{i}_{n_cluster}"
        )

    feature_importance=linear_regresion(df_resultado,variable='subcluster')
    feature_importance['abs_coeficiente'] = feature_importance['coeficiente'].abs()

    total_abs_coeficiente = feature_importance['abs_coeficiente'].sum()
    feature_importance['porcentaje'] = ((feature_importance['abs_coeficiente'] / total_abs_coeficiente) * 100).round(2)

    feature_sorted=feature_importance.sort_values(by='abs_coeficiente', ascending=False).drop(columns=['abs_coeficiente','coeficiente'])

    # Añadir columnas de media y desviación estándar por cluster para cada característica
    for i in range(n):
        subcluster_data = df_resultado[df_resultado['subcluster'] == n*n_cluster+i]  # Datos solo de este cluster
        subcluster_means = subcluster_data.mean()
        subcluster_stds = subcluster_data.std()
        
        # Añadir las medias y desviaciones estándar al DataFrame feature_sorted
        for col in subcluster_means.index:
            if col in feature_sorted.index:  # Solo añadir si la característica está en feature_sorted
                feature_sorted.loc[col, f'PC_{i+1}_avg'] = subcluster_means[col]
                feature_sorted.loc[col, f'PC_{i+1}_std'] = subcluster_stds[col]

    # Redondear todos los valores numéricos a dos decimales
    feature_sorted_subcluster = feature_sorted.round(2)


    return combinaciones_df_list,feature_sorted_subcluster,"\n".join(descripciones)

def convert_df_to_bytes(df_str):
    """
    Convierte el string de un CSV a un formato adecuado para la descarga en Streamlit.
    """
    return BytesIO(df_str.encode())


def contar_variables_frecuentes(df, prefijo='rec_nutricionales_', N=5):
    """
    Cuenta las variables que comienzan con un prefijo determinado y muestra las N más frecuentes.
    
    Parámetros:
    df (pd.DataFrame): DataFrame con los datos.
    prefijo (str): Prefijo de las columnas de interés.
    N (int): Número de variables más frecuentes a retornar.
    
    Retorna:
    pd.DataFrame: DataFrame con las N variables más frecuentes y sus recuentos.
    """
    # Filtrar las columnas que empiezan con el prefijo especificado
    cols_interes = [col for col in df.columns if col.startswith(prefijo)]
    
    # Hacer un recuento de cuántas veces aparece el valor 1 (o cualquier valor diferente de 0)
    frecuencias = df[cols_interes].sum().sort_values(ascending=False)
    
    # Seleccionar las N más frecuentes
    top_frecuentes = frecuencias.head(N)
    
    # Eliminar el prefijo de los nombres de las variables
    top_frecuentes.index = top_frecuentes.index.str.replace(prefijo, '')

    # Convertir en DataFrame para mostrar
    top_frecuentes_df = top_frecuentes.reset_index()
    top_frecuentes_df.columns = ['Variable', 'Frecuencia']
    
    return top_frecuentes_df

# # Configuración de Streamlit
# st.title('Análisis de Combinaciones en el Dataset')

# Definir prefijos de interés y sus nombres amigables
todos_prefijos_interes = {
    'dx_principal_': 'diagnóstico principal',
    'dx_asociados_': 'diagnósticos asociados',
    'medicamento': 'medicamento',
    'rec_nutri': 'recomendaciones nutricionales',
    'cons_dietetico': 'consejo dietético',
    'reglas ': 'reglas'
}

# Seleccionar categorías
categorias_seleccionadas = st.multiselect(
    "Selecciona con qué categoría/s quieres generar los perfiles clínicos:",
    options=list(todos_prefijos_interes.keys()),
    default=['reglas '],
    format_func=lambda x: todos_prefijos_interes[x]
)

# Filtrar prefijos de interés seleccionados
prefijos_interes = {key: todos_prefijos_interes[key] for key in categorias_seleccionadas}

# Seleccionar número de resultados a mostrar
n_resultados = st.slider("Número de perfiles clínicos por clúster a mostrar:", min_value=1, max_value=10, value=4)


# Mostrar los resultados para cada clúster
for cluster_label, cluster_df in cluster_dfs.items():
    st.subheader(f"Clúster {cluster_label}")
    combinaciones_df_list,feature_sorted_subcluster,descripcion_resultado = encontrar_combinaciones(cluster_df, prefijos_interes,cluster_label, n_resultados)
    st.text(descripcion_resultado)
    st.write("Las variables más significativas para generar cada perfil clínico y su estadística:",feature_sorted_subcluster)
    st.text("Recomendaciones nutricionales y el consejo dietético de estos perfiles son:")
    pc=1
    col1, col2 = st.columns(2)
    for df_i in combinaciones_df_list:
        top_frecuentes_df_rec_nutri=contar_variables_frecuentes(df_i, prefijo='rec_nutricionales_', N=5)
        # st.dataframe(top_frecuentes_df_rec_nutri)
        top_frecuentes_df_cons_diet=contar_variables_frecuentes(df_i, prefijo='cons_dietetico_', N=5)
        # st.dataframe(top_frecuentes_df_cons_diet)
        # En la primera columna colocas el primer DataFrame
        with col1:
            st.text(f"Recomendaciones Nutricionales PC_{pc}:")
            st.dataframe(top_frecuentes_df_rec_nutri)

        # En la segunda columna colocas el segundo DataFrame
        with col2:
            st.text(f"Consejo dietético PC_{pc}:")
            st.dataframe(top_frecuentes_df_cons_diet)
        pc+=1




### Desviación estándar de cada variable para cada clúster

# # Función para calcular la desviación estándar por cluster
# def desviacion_estandar_por_cluster(df, centroids, labels):
#     desviaciones_estandar = pd.DataFrame(columns=df.columns)
    
#     for cluster_id in range(len(centroids)):
#         cluster_data = df[labels == cluster_id]
#         std_dev = cluster_data.std()
#         desviaciones_estandar.loc[cluster_id] = std_dev
    
#     return desviaciones_estandar

# # Función para crear el DataFrame de centroides y desviaciones estándar
# def crear_dataframe_centroides_desviacion(centroids, desviaciones_estandar):
#     cluster_dataframes = {}
#     output = BytesIO()
    
#     with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#         for cluster_id in range(len(centroids)):
#             centroide = centroids[cluster_id]
#             desviacion = desviaciones_estandar.loc[cluster_id].values

#             # Verificación de longitudes y contenido
#             # print(f"Cluster {cluster_id}:")
#             # print(f"Centroide: {centroide}, longitud: {len(centroide)}")
#             # print(f"Desviacion: {desviacion}, longitud: {len(desviacion)}")
#             # print(f"Columnas: {desviaciones_estandar.columns}, longitud: {len(desviaciones_estandar.columns)}")

#             df_cluster = pd.DataFrame({
#                 'Variable': desviaciones_estandar.columns,
#                 'Valor Promedio': centroide,
#                 'Desviacion Estándar': desviacion
#             })
            
#             cluster_dataframes[cluster_id] = df_cluster
#             df_cluster.to_excel(writer, sheet_name=f'Cluster_{cluster_id}', index=False)
        
#         writer.save()
#         processed_data = output.getvalue()
    
#     return cluster_dataframes, processed_data

# st.title("Estadísticas de los perfiles clínicos")

# # Calcular las desviaciones estándar por cluster

# desviaciones_estandar = desviacion_estandar_por_cluster(df, centroids_kmeans, labels_kmeans)
# desviaciones_estandar=desviaciones_estandar.drop(columns=['cluster'])

# # Mostrar el DataFrame de desviaciones estándar
# # st.write("Desviaciones Estándar por Cluster:", desviaciones_estandar)

# # Crear y mostrar los DataFrames de centroides y desviaciones estándar
# # print(centroids_kmeans)
# # print(desviaciones_estandar)    
# cluster_dataframes, excel_data = crear_dataframe_centroides_desviacion(centroids_kmeans, desviaciones_estandar)
# st.write("Desviaciones Estándar y Valores promedio de cada perfil clínico:")
# for cluster_id, df_cluster in cluster_dataframes.items():
#     st.write(f"### Perfil clínico {cluster_id}")
#     st.dataframe(df_cluster)  



#ESTADÍSTICA
### Las variables más significativas para generar cada cluster (regresión lineal)


# k_kmeans = st.slider("Seleccione el número de clusters", 1, 10, 3)
# labels_kmeans,centroids_kmeans=kmeans_algorithm(df,n_clusters=k_kmeans,init='k-means++',n_init = 10,max_iter=500,tol=0.0001,random_state= 111,algorithm='elkan')
# df['cluster'] = labels_kmeans

feature_importance=linear_regresion(df)
feature_importance['abs_coeficiente'] = feature_importance['coeficiente'].abs()

total_abs_coeficiente = feature_importance['abs_coeficiente'].sum()
feature_importance['porcentaje'] = ((feature_importance['abs_coeficiente'] / total_abs_coeficiente) * 100).round(2)

feature_sorted=feature_importance.sort_values(by='abs_coeficiente', ascending=False).drop(columns=['abs_coeficiente','coeficiente'])

# Añadir columnas de media y desviación estándar por cluster para cada característica
for i in range(k_kmeans):
    cluster_data = df[df['cluster'] == i]  # Datos solo de este cluster
    cluster_means = cluster_data.mean()
    cluster_stds = cluster_data.std()
    
    # Añadir las medias y desviaciones estándar al DataFrame feature_sorted
    for col in cluster_means.index:
        if col in feature_sorted.index:  # Solo añadir si la característica está en feature_sorted
            feature_sorted.loc[col, f'{i}_avg'] = cluster_means[col]
            feature_sorted.loc[col, f'{i}_std'] = cluster_stds[col]

# Redondear todos los valores numéricos a dos decimales
feature_sorted = feature_sorted.round(2)

st.title("Peso de las variables para generar los clústers")
st.write("Las variables más significativas para generar cada cluster (regresión lineal):",feature_sorted)

nombres_de_variables = feature_importance['porcentaje'].index.tolist()

lista_corr_dx = [elemento for elemento in nombres_de_variables if elemento.startswith('dx_')]
lista_corr_medicamento = [elemento for elemento in nombres_de_variables if elemento.startswith('medicamento_')]
lista_corr_reglas = [elemento for elemento in nombres_de_variables if elemento.startswith('reglas ')]
lista_corr_nutricion = [elemento for elemento in nombres_de_variables if (elemento.startswith('rec_nutri') or elemento.startswith('cons_dietetico_'))]
# Crear una lista combinada de elementos a eliminar
elementos_a_eliminar = set(lista_corr_dx + lista_corr_medicamento + lista_corr_reglas + lista_corr_nutricion)

# Filtrar feature_importance para excluir los elementos a eliminar
lista_corr_hce = [elemento for elemento in nombres_de_variables if elemento not in elementos_a_eliminar]

#### SELECCCIONAR Nº DE VARIABLES MÁS SIGNIFICATIVAS PARA HACER LOS CLUSTERS
# Streamlit widget para seleccionar el número de variables a mostrar
st.write("Selecciona el número de variables a mostrar")
N1 = st.slider('Selecciona el número de variables del HCE a mostrar:', min_value=0, max_value=len(lista_corr_hce), value=3)  # Puedes ajustar los valores mínimo y máximo según tus necesidades
N2 = st.slider('Selecciona el número de variables de nutrición a mostrar:', min_value=0, max_value=len(lista_corr_nutricion), value=3)  # Puedes ajustar los valores mínimo y máximo según tus necesidades
N3 = st.slider('Selecciona el número de variables de diagnóstico a mostrar:', min_value=0, max_value=len(lista_corr_dx), value=3)  # Puedes ajustar los valores mínimo y máximo según tus necesidades
N4 = st.slider('Selecciona el número de variables de medicamentos a mostrar:', min_value=0, max_value=len(lista_corr_medicamento), value=3)  # Puedes ajustar los valores mínimo y máximo según tus necesidades
N5 = st.slider('Selecciona el número de variables de reglas a mostrar:', min_value=0, max_value=len(lista_corr_reglas), value=3)  # Puedes ajustar los valores mínimo y máximo según tus necesidades

# Mostrar las N primeras variables
st.write("Variables del HCE")
feature_sorted_hce=feature_sorted[feature_sorted['porcentaje'].index.isin(lista_corr_hce)]
st.write(feature_sorted_hce.head(N1))
feature_sorted_nutricion=feature_sorted[feature_sorted['porcentaje'].index.isin(lista_corr_nutricion)]
st.write(feature_sorted_nutricion.head(N2))
feature_sorted_dx=feature_sorted[feature_sorted['porcentaje'].index.isin(lista_corr_dx)]
st.write(feature_sorted_dx.head(N3))
feature_sorted_medicamento=feature_sorted[feature_sorted['porcentaje'].index.isin(lista_corr_medicamento)]
st.write(feature_sorted_medicamento.head(N4))
feature_sorted_reglas=feature_sorted[feature_sorted['porcentaje'].index.isin(lista_corr_reglas)]
st.write(feature_sorted_reglas.head(N5))


top_n_features = feature_sorted.head(N1).index.tolist()

  




st.title("Exploración de cada categoría para cada cluster")

### Los diagnósticos y medicamentos de cada cluster

def from_df_to_json(df, variables='diagnostico_principal'):
    if variables == 'diagnostico_principal':
        columnas = df.filter(regex='^dx_principal_').columns.tolist()
    
    if variables == 'diagnostico_asociado':
        columnas = df.filter(regex='^dx_asociados_').columns.tolist()

    if variables =='medicamento':
        columnas = df.filter(regex='^medicamento').columns.tolist()
    
    if variables =='consejo_dietetico':
        columnas = df.filter(regex='^categorias_cons_dietetico_').columns.tolist()

    if variables =='reglas':
        columnas = df.filter(regex='^reglas ').columns.tolist()

    recuento = {columna: int(df[columna].sum()) for columna in columnas}
    
    # Convertir el diccionario a JSON
    recuento_json = json.dumps(recuento)
        
    return recuento_json

# Definir la función para mostrar resultados ordenados
def mostrar_ordenado(diccionario, titulo):
    st.write(f"### {titulo} (de más frecuente a menos frecuente):")
    # Ordenar el diccionario por los valores de forma descendente
    for clave, valor in sorted(diccionario.items(), key=lambda item: item[1], reverse=True):
        st.write(f"{clave}: {valor}")

# Variables disponibles para visualizar
categorias = {
    'diagnostico_principal': 'Diagnósticos Principales',
    'diagnostico_asociado': 'Diagnósticos Asociados',
    'medicamento': 'Medicamentos',
    'consejo_dietetico': 'Consejo dietético',
    'reglas': 'Reglas'
}

# Selección de categorías por el usuario
seleccion = st.selectbox(
    'Seleccione las categorías que desea visualizar:',
    options=list(categorias.keys()),
    format_func=lambda x: categorias[x]
)

if seleccion:
    # Procesar y mostrar resultados para cada cluster
    resultados = {}
    for cluster_id, df_cluster in cluster_dfs.items():
        df_cluster_reset = df_cluster.reset_index(drop=True)

        # Convertir los DataFrames a JSON y luego a diccionarios
        json_data = from_df_to_json(df_cluster_reset, variables=seleccion)
        diccionario = json.loads(json_data)

        resultados[cluster_id] = diccionario

    # Mostrar los resultados ordenados para cada cluster según la selección del usuario
    for cluster_id, diccionario in resultados.items():
        st.write(f"## Resultados para el cluster {cluster_id+1}:")
        mostrar_ordenado(diccionario, categorias[seleccion])
else:
    st.write("Por favor, seleccione al menos una categoría para visualizar.")




# Asumiendo que 'df' es tu DataFrame original y que tiene una columna 'cluster'
def plot_cluster_histogram(df):
    """
    Cuenta los valores de cada cluster y los representa en un histograma.

    :param df: pandas DataFrame, el DataFrame que contiene la columna 'cluster'.
    """
    # Contar los valores de cada cluster
    cluster_counts = df['cluster'].value_counts().sort_index()

    # Crear el histograma
    plt.figure(figsize=(10, 6))
    cluster_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribución de Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Número de registros')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()
    st.pyplot(plt)

# Llamar a la función para representar el histograma
st.title(f"Distribución de clusters")
plot_cluster_histogram(df)

st.title("Visualización de Clusters en función de dos variables")

# Selección de variables para el eje X e Y
x_var = st.selectbox("Selecciona la variable para el eje X:", df.columns[:-1])
y_var = st.selectbox("Selecciona la variable para el eje Y:", df.columns[:-1])

# Comprobar que no se seleccionen las mismas variables para ambos ejes
if x_var == y_var:
    st.error("Selecciona diferentes variables para los ejes X e Y.")
else:
    # Crear el gráfico
    fig, ax = plt.subplots()
    clusters = df['cluster'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

    for cluster, color in zip(clusters, colors):
        cluster_data = df[df['cluster'] == cluster]
        ax.scatter(cluster_data[x_var], cluster_data[y_var], s=50, alpha=0.6, label=f'Cluster {cluster+1}', color=color)

    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_title('Visualización de Clusters')
    ax.legend(title='Clusters')
    ax.grid(True)

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)




# st.title("Decission Tree")

# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.metrics import accuracy_score, confusion_matrix

# # Filtrar las columnas que empiezan con 'dx_principal_'
# columnas = df.filter(regex='^dx_principal_').columns.tolist()

# # Crear una nueva columna para almacenar los resultados
# df['dx_combined'] = 0  # Inicializar con 0

# # Asignar valores a 'dx_combined' basado en las columnas 'dx_principal_'
# for i in range(1, len(columnas) + 1):
#     for j in range(len(df)):
#         if df[columnas[i - 1]][j] == 1:
#             df.at[j, 'dx_combined'] = i

# class_mapping = {i: columnas[i - 1] for i in range(1, len(columnas) + 1)}


# # Añadir 'dx_combined' a la lista de columnas
# columnas.append('dx_combined')

# # Definir características (X) y variable objetivo (y)
# X = df.drop(columns=columnas, axis=1)
# y = df['dx_combined']

# # Dividir los datos en conjuntos de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Crear el modelo de árbol de decisión
# k_decission_tree=st.slider("Seleccione la profundidad del árbol", 1, 10, 3)
# model = DecisionTreeClassifier(max_depth=k_decission_tree)

# # Entrenar el modelo
# model.fit(X_train, y_train)

# # Predecir con el modelo entrenado
# y_pred = model.predict(X_test)

# # Evaluar el modelo
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Mostrar precisión y matriz de confusión
# st.write(f'Accuracy: {accuracy:.2%}')
# # st.write('Confusion Matrix:')
# # st.write(conf_matrix)

# # Calcular la precisión por clase
# true_positives = conf_matrix.diagonal()
# totals_per_class = conf_matrix.sum(axis=1)
# precision_per_leaf = true_positives / totals_per_class

# # Mostrar la precisión por clase
# for i, precision in enumerate(precision_per_leaf):
#     if i + 1 in class_mapping:
#         st.write(f'Precision for {class_mapping[i + 1]}: {precision:.2%}')
#     # else:
#     #     st.write(f'Precision for class {i + 1}: {precision:.2%}')
#     # st.write(f'Precision for {class_mapping.get(i + 1, "Unknown")}: {precision:.2%}')
#     # st.write(f'Precision for class {i}: {precision:.2%}')

# # Visualizar el árbol de decisión
# fig, ax = plt.subplots(figsize=(20, 10))
# #class_names = list(map(str, model.classes_))
# class_names = [class_mapping.get(c, "Unknown") for c in model.classes_]

# plot_tree(model, filled=True, feature_names=X.columns, class_names=class_names, rounded=True, fontsize=12, ax=ax)
# st.pyplot(fig)


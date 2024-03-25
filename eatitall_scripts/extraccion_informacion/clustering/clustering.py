from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def select_dataset(df,template='all',vars=[]):
    """
    Selecciona un subconjunto del DataFrame dado según la plantilla especificada.

    Parámetros:
        - df (pandas.DataFrame): El DataFrame de entrada.
        - template (str, opcional): La plantilla para seleccionar el conjunto de datos. El valor predeterminado es 'all'.
            - 'all': Devuelve todo el DataFrame.
            - 'personalized': Devuelve un subconjunto del DataFrame basado en las variables especificadas.
            - 'compact': Devuelve una versión compacta del DataFrame.
            - 'HCER': Devuelve un subconjunto del DataFrame relacionado con eventos de atención médica (HCE) + Reglas extraídas.
            - 'HCE sin reglas': Devuelve un subconjunto del DataFrame relacionado con eventos de atención médica (HCE)
        - vars (lista, opcional): Una lista de nombres de variables para incluir en el conjunto de datos personalizado.
            Solo aplicable cuando template='personalized'.

    Devuelve:
        pandas.DataFrame: El subconjunto seleccionado del DataFrame de entrada.

    Notas:
        - Esta función depende del orden en que se extrajeron las variables.
        - Para template='compact', combina variables con sufijos '_v1_' y '_v2_' en una sola variable y depende 
          del nombre de la última variable correspondiente a una regla
        - Para template='HCER', selecciona variables hasta la última variable de eventos de atención médica (HCE) + las reglas extraídas, 
          motivo por el cual se requiere el nombre de esa última variable.
        - Para template='HCE sin reglas', selecciona variables hasta la última variable de eventos de atención médica (HCE), 
          motivo por el cual se requiere el nombre de esa última variable.

    Ejemplo de uso:
        >>> df = pd.read_csv('data.csv')
        >>> selected_df = select_dataset(df, template='personalized', vars=['edad', 'colesterol'])
        >>> print(selected_df.head())
    """
    if template=='all':
        return df
    if template=='personalized':
        if vars==[]:
            print("se deben añadir las variables que quieres en el dataset de la siguiente forma: clustering.select_dataset(df,template='personalized',vars=['edad','tg','estatina'])")
        else:
            return df[vars]
    if template=='compact':
        indice_ultima_var_HCE_y_reglas=df.columns.get_loc('diabetes_mayores_de_65_y_salud_muy_compleja') #AQUÍ SE COGE EL ÍNDICE DE LA *ÚLTIMA VARIABLE*
                                                                                                         #DE LA ÚTLIMA REGLA AÑADIDA. EN LA VERSIÓN 0.0.1 
                                                                                                         #CORRESPONDE A 'diabetes_mayores_de_65_y_salud_muy_compleja'
        df_compacto=df.iloc[:,:indice_ultima_var_HCE_y_reglas+1]
        indice_ultima_var_HCE_y_reglas
        for i in range(indice_ultima_var_HCE_y_reglas+1,len(df.iloc[0])):
            nombre_var_v1=df.columns[i]
            if "_v1_" in nombre_var_v1:
                nombre_var_v2=nombre_var_v1.replace("_v1_","_v2_")
                nombre_nueva_var=nombre_var_v1.replace("_v1_","_")
                df_compacto[nombre_nueva_var]=0
                for j in range(0,len(df)):
                    df_compacto[nombre_nueva_var][j]=df[nombre_var_v1][j]+df[nombre_var_v2][j]
        return df_compacto
    if template=='HCER':
        indice_ultima_var_HCE_y_reglas=df.columns.get_loc('diabetes_mayores_de_65_y_salud_muy_compleja') #AQUÍ SE COGE EL ÍNDICE DE LA *ÚLTIMA VARIABLE*
                                                                                                         #DE LA ÚTLIMA REGLA AÑADIDA. EN LA VERSIÓN 0.0.1 
                                                                                                         #CORRESPONDE A 'diabetes_mayores_de_65_y_salud_muy_compleja'
        df_compacto=df.iloc[:,:indice_ultima_var_HCE_y_reglas+1]
        df_hce=df.iloc[:,:indice_ultima_var_HCE_y_reglas+1]
        return df_hce
    if template=='HCE sin reglas':
        indice_ultima_var_HCE_sin_reglas=df.columns.get_loc('HbA1cbasal7') #AQUÍ SE COGE EL ÍNDICE DE LA *ÚLTIMA VARIABLE*
                                                                                #DEL ÚLTIMO DATO DEL HCE. EN LA VERSIÓN 0.0.1 
                                                                                #CORRESPONDE A 'HbA1cbasal7'
        df_compacto=df.iloc[:,:indice_ultima_var_HCE_sin_reglas+1]
        df_hce_sin_reglas=df.iloc[:,:indice_ultima_var_HCE_sin_reglas+1]
        return df_hce_sin_reglas
    else:
        print("Escribe un nombre de template válido: all, compact, HCER, HCE sin reglas")

def replace_nan_values(df):
    # Comprobar si hay valores NaN en el DataFrame y en qué columnas
    hay_nans = df.isnull().values.any()
    if hay_nans:
        print("Hay valores NaN en el DataFrame.")
        # Mostrar cuántos NaN hay por columna y listar las columnas que contienen NaN
        nans_por_columna = df.isnull().sum()
        print("\nNúmero de NaN por columna:")
        print(nans_por_columna[nans_por_columna > 0])  # Filtrar para mostrar solo las columnas con NaN
        # Identificar y listar explícitamente las columnas que tienen NaN
        columnas_con_nans = nans_por_columna[nans_por_columna > 0].index.tolist()
        print("\nColumnas que contienen NaNs:", columnas_con_nans)
        df=df.fillna(-100)
        print("Se han convertido cada NaN en un -100")
    else:
        print("No hay valores NaN en el DataFrame.")
    return df

def manage_string_values(df):
    columnas_string = [col for col in df.columns if df[col].dtype == 'object']
    df=df.drop(columnas_string,axis=1)
    print("Se han eliminado las siguientes columnas: ",columnas_string)
    return df

def inertia_of_kmeans(df,maxClusters=10,n_clusters=5,init='k-means++',n_init = 10,max_iter=500,tol=0.0001,random_state= 111,algorithm='elkan'):
    inertia = []  # Lista para almacenar los valores de inercia
    # La inercia es la suma de las distancias al cuadrado de cada muestra a su centroide más cercano

    # Bucle para probar diferentes valores de k (número de clústeres)
    for n in range(2 , maxClusters):
        algorithm = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                        tol=0.0001, random_state=111, algorithm='elkan')
        algorithm.fit(df)
        # Almacena la inercia del modelo en la lista
        inertia.append(algorithm.inertia_)
    plt.figure(1, figsize=(15, 8))  
    # Gráfica de dispersión ('o') y línea ('-') para visualizar la relación entre k y la inercia
    plt.plot(np.arange(2, maxClusters), inertia, 'o')
    plt.plot(np.arange(2, maxClusters), inertia, '-', alpha=0.5)
    # Etiqueta del eje x
    plt.xlabel('Number of Clusters')
    # Etiqueta del eje y
    plt.ylabel('Inertia')
    # Muestra el gráfico
    plt.show() 
    return inertia

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

def plot_clusters(df,centroids,labels,x_index=0,y_index=1):
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

def obtener_desvEst_distPromedio_distPromedioResto(variable,df,centroids,labels,indice_cluster_i):
    indices_df_cluster_i = np.where(labels == indice_cluster_i)[0]
    df_cluster_i=df.iloc[indices_df_cluster_i]
    variable_desv_est_cluster_i=np.std(np.array(df_cluster_i[variable].tolist()))
    indice_variable=df.columns.get_loc(variable)
    variable_dist_centroide=np.sqrt(((np.array(df_cluster_i[variable].tolist())-centroids[indice_cluster_i][indice_variable])**2).sum(axis=0))
    variable_distancia_promedio = np.mean(variable_dist_centroide)

    indices_df_cluster_resto_i = np.where(labels != indice_cluster_i)[0]
    df_cluster_resto_i=df.iloc[indices_df_cluster_resto_i]
    variable_dist_centroide_resto=np.sqrt(((np.array(df_cluster_resto_i[variable].tolist())-centroids[indice_cluster_i][indice_variable])**2).sum(axis=0))
    variable_distancia_resto_promedio = np.mean(variable_dist_centroide_resto)
    return variable_desv_est_cluster_i,variable_distancia_promedio,variable_distancia_resto_promedio

def extraer_indices_de_columnas_con_prefijo(df,prefijo):
    columnas_con_prefijo = [col for col in df.columns if col.startswith(prefijo)]

    # Extraer los índices de estas columnas
    indices = [df.columns.get_loc(col) for col in columnas_con_prefijo]
    return indices

def centroid_dictionary(df,centroids,labels):
    indices_al=extraer_indices_de_columnas_con_prefijo(df,'al_')
    indices_far=extraer_indices_de_columnas_con_prefijo(df,'far_')
    indices_sin=extraer_indices_de_columnas_con_prefijo(df,'sin_')
    indices_pc=extraer_indices_de_columnas_con_prefijo(df,'pc_')

    centroides_dict = {}

    for i in range(len(centroids)):
        # Crear un diccionario para los alimentos del centroide actual
        num_pacientes=np.sum(labels==i)
        indices_df_cluster_i = np.where(labels == i)[0]
        df_cluster_i=df.iloc[indices_df_cluster_i]

        imc_desv_est_cluster_i,imc_distancia_promedio,imc_distancia_resto_promedio=obtener_desvEst_distPromedio_distPromedioResto("imc",df,centroids,labels,i)
        tas_desv_est_cluster_i,tas_distancia_promedio,tas_distancia_resto_promedio=obtener_desvEst_distPromedio_distPromedioResto("tas",df,centroids,labels,i)
        tad_desv_est_cluster_i,tad_distancia_promedio,tad_distancia_resto_promedio=obtener_desvEst_distPromedio_distPromedioResto("tad",df,centroids,labels,i)
        tg_desv_est_cluster_i,tg_distancia_promedio,tg_distancia_resto_promedio=obtener_desvEst_distPromedio_distPromedioResto("tg",df,centroids,labels,i)
        hba1c_desv_est_cluster_i,hba1c_distancia_promedio,hba1c_distancia_resto_promedio=obtener_desvEst_distPromedio_distPromedioResto("hba1c",df,centroids,labels,i)
        fg_desv_est_cluster_i,fg_distancia_promedio,fg_distancia_resto_promedio=obtener_desvEst_distPromedio_distPromedioResto("fg",df,centroids,labels,i)
        edad_desv_est_cluster_i,edad_distancia_promedio,edad_distancia_resto_promedio=obtener_desvEst_distPromedio_distPromedioResto("edad",df,centroids,labels,i)
        acv_desv_est_cluster_i,acv_distancia_promedio,acv_distancia_resto_promedio=obtener_desvEst_distPromedio_distPromedioResto("acv",df,centroids,labels,i)

        alimentos_dict = {}
        farmacos_dict={}
        sintomas_dict={}
        pruebas_clinicas_dict={}
        for k in range(len(centroids[i])):
            if indices_al[0] <= k <= indices_al[-1]:  # Verificar si el índice corresponde a un alimento
                if centroids[i][k] != 0:  # Considerar solo los alimentos con frecuencia no nula
                    alimentos_dict[f"alimento{k}"] = {"nombre": df.columns[k], "frecuencia": centroids[i][k]}
            if indices_far[0] <= k <= indices_far[-1]:  # Verificar si el índice corresponde a un alimento
                if centroids[i][k] != 0:  # Considerar solo los alimentos con frecuencia no nula
                    farmacos_dict[f"farmaco{k}"] = {"nombre": df.columns[k], "frecuencia": centroids[i][k]}
            if indices_sin[0] <= k <= indices_sin[-1]:  # Verificar si el índice corresponde a un alimento
                if centroids[i][k] != 0:  # Considerar solo los alimentos con frecuencia no nula
                    sintomas_dict[f"sintoma{k}"] = {"nombre": df.columns[k], "frecuencia": centroids[i][k]}
            if indices_pc[0] <= k <= indices_pc[-1]:  # Verificar si el índice corresponde a un alimento
                if centroids[i][k] != 0:  # Considerar solo los alimentos con frecuencia no nula
                    pruebas_clinicas_dict[f"prueba_clinica{k}"] = {"nombre": df.columns[k], "frecuencia": centroids[i][k]}
            if k==df.columns.get_loc("imc"):
                imc_value={"valor": centroids[i][k]}
            if k==df.columns.get_loc("tas"):
                tas_value={"valor": centroids[i][k]}
            if k==df.columns.get_loc("tad"):
                tad_value={"valor": centroids[i][k]}
            if k==df.columns.get_loc("tg"):
                tg_value={"valor": centroids[i][k]}
            if k==df.columns.get_loc("hba1c"):
                hba1c_value={"valor": centroids[i][k]}
            if k==df.columns.get_loc("fg"):
                fg_value={"valor": centroids[i][k]}
            if k==df.columns.get_loc("edad"):
                edad_value={"valor": centroids[i][k]}
            if k==df.columns.get_loc("acv"):
                acv_value={"valor": centroids[i][k]}
        # Asignar el diccionario de alimentos al cluster correspondiente
        centroides_dict[f"cluster{i}"] = {"nun. paceintes":num_pacientes,
                                            "imc":{"valor":imc_value,
                                                    "desv_est":imc_desv_est_cluster_i,
                                                    "dist promedio":imc_distancia_promedio,
                                                    "dist promedio resto clusters":imc_distancia_resto_promedio,},
                                            "tas":{"valor":tas_value,
                                                    "desv_est":tas_desv_est_cluster_i,
                                                    "dist promedio":tas_distancia_promedio,
                                                    "dist promedio resto clusters":tas_distancia_resto_promedio,},
                                            "tad":{"valor":tad_value,
                                                    "desv_est":tad_desv_est_cluster_i,
                                                    "dist promedio":tad_distancia_promedio,
                                                    "dist promedio resto clusters":tad_distancia_resto_promedio,},
                                            "tg":{"valor":tg_value,
                                                    "desv_est":tg_desv_est_cluster_i,
                                                    "dist promedio":tg_distancia_promedio,
                                                    "dist promedio resto clusters":tg_distancia_resto_promedio,},
                                            "hba1c":{"valor":hba1c_value,
                                                    "desv_est":hba1c_desv_est_cluster_i,
                                                    "dist promedio":hba1c_distancia_promedio,
                                                    "dist promedio resto clusters":hba1c_distancia_resto_promedio,},
                                            "fg":{"valor":fg_value,
                                                    "desv_est":fg_desv_est_cluster_i,
                                                    "dist promedio":fg_distancia_promedio,
                                                    "dist promedio resto clusters":fg_distancia_resto_promedio,},
                                            "edad":{"valor":edad_value,
                                                    "desv_est":edad_desv_est_cluster_i,
                                                    "dist promedio":edad_distancia_promedio,
                                                    "dist promedio resto clusters":edad_distancia_resto_promedio,},
                                            "acv":{"valor":acv_value,
                                                    "desv_est":acv_desv_est_cluster_i,
                                                    "dist promedio":acv_distancia_promedio,
                                                    "dist promedio resto clusters":acv_distancia_resto_promedio,},
                                            "alimentos": alimentos_dict,
                                            "farmacos":farmacos_dict,
                                            "sintomas":sintomas_dict,
                                            "pruebas clinicas":pruebas_clinicas_dict}
    return centroides_dict



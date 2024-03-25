import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def preparar_dataset_para_correlaciones(df,centroids='',labels='',template='all',cluster_i=-100,variables=[]):
    if template=='cluster':
        if cluster_i!=-100:
            df['label'] = labels
            df=df[df['label'] == cluster_i]
        if cluster_i==-100:
            print("Indica un valor del cluster correcto")
    if template=='eliminar voacbulario que no aparece':
        # Lista de prefijos a buscar
        prefijos = ['al_', 'far_', 'sin_', 'pc_']
        # Identificar las columnas que empiezan con los prefijos indicados
        columnas_a_eliminar = [col for col in df.columns if col.startswith(tuple(prefijos))]
        # Revisar si las columnas identificadas están llenas de 0s y eliminarlas
        for columna in columnas_a_eliminar:
            if df[columna].eq(0).all():  # .eq(0).all() comprueba si todos los valores son 0
                df.drop(columna, axis=1, inplace=True)
    if template=='eliminar columnas con strings':
        df=df.select_dtypes(exclude=['object'])
    if template=='eliminar columnas con 0s':
        df = df.loc[:, (df != 0).any(axis=0)]
    if template=='eliminar reglas':
        pass
    if template=='solo reglas':
        pass
    if template=='seleccionar variables':
        if variables!=[]:
            pass
        else:
            print("Indica en una lista las variables que quieres en el dataframe")
    return df

def encontrar_N_variables_mas_correlacionadas_y_correlaciones(df, num_variables, umbral_maximo):
    # Calcular la matriz de correlación para las variables numéricas
    corr_matrix = df.corr().abs()
    
    # Convertir la matriz de correlación en un DataFrame más manejable
    corr_unstack = corr_matrix.unstack()
    sorted_corr = corr_unstack.sort_values(kind="quicksort", ascending=False)
    
    # Eliminar duplicados y la correlación perfecta de una variable consigo misma
    sorted_corr = sorted_corr[sorted_corr < 1].drop_duplicates()

    # Filtrar las correlaciones que superan el umbral_maximo
    filtered_corr = sorted_corr[sorted_corr < umbral_maximo]
    
    # Seleccionar las top N variables más correlacionadas (en pares) según el filtro de umbral
    top_n_corr = filtered_corr.head(num_variables * 2)  # Multiplicado por 2 para asegurar la cobertura de N variables
    
    # Identificar las variables únicas implicadas en las correlaciones más altas
    unique_vars = pd.Index(top_n_corr.index.get_level_values(0)).union(top_n_corr.index.get_level_values(1)).unique()

    correlaciones_ordenadas_df = pd.DataFrame(top_n_corr).reset_index()
    correlaciones_ordenadas_df.columns = ['Variable1', 'Variable2', 'Correlacion']
    
    # Limitar a las primeras N variables únicas
    if len(unique_vars) > num_variables:
        unique_vars = unique_vars[:num_variables]
    
    return df[unique_vars],correlaciones_ordenadas_df

def visualizar_correlaciones(dataset,titulo=''):
    """
    Visualiza la matriz de correlación de un DataFrame de pandas.

    :param dataset: DataFrame de pandas.
    """
    # Calcular la matriz de correlación
    correlaciones = dataset.corr()

    # Crear un mapa de calor
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlaciones, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Mapa de Calor de Correlaciones'+titulo)
    plt.show()

def correlacion_una_var_con_el_df(df,var,N=20):
    """
    Calcula la correlación de una variable dada con el resto de las variables en el DataFrame.

    Parámetros:
        - df (pandas.DataFrame): El DataFrame de entrada.
        - variable_name (str): El nombre de la variable para la cual se calculará la correlación.

    Devuelve:
        pandas.DataFrame: Un DataFrame con una fila y N columnas, donde N es el número de las otras variables.
        Cada columna representa el coeficiente de correlación entre la variable especificada y otra variable.

    Ejemplo de uso:
        >>> df = pd.read_csv('data.csv')
        >>> correlation_df = calculate_variable_correlation(df, 'variable_objetivo')
        >>> print(correlation_df)
    """
    # Excluye la variable especificada del DataFrame
    otras_variables = df.drop(columns=[var])
    
    # Calcula los coeficientes de correlación
    correlaciones = otras_variables.corrwith(df[var])

    # Crea un DataFrame con una fila y N columnas
    correlation_df = pd.DataFrame(correlaciones).T
    correlation_df.insert(0,"nombre_var", var)


    # Calcula als N variables más correlacionadas
    top_N_correlations = correlaciones.abs().nlargest(N)
    selected_variables = otras_variables[top_N_correlations.index]
    correlation_df_top_N = pd.DataFrame(selected_variables.corrwith(df[var])).T
    correlation_df_top_N.insert(0,"nombre_var", var)
    
    return correlation_df, correlation_df_top_N

def correlacion_conjunto_de_vars_con_el_df(df,tipo,vars=[]):
    #Esta función solo funciona en alimento, fármacos, síntomas, pruebas clínicas si se coge el dataset 'compact'
    #En caso contrario se debe trabajar con 'personalizado'
    dataframes_agrupados=[]
    longitud_df=len(df.iloc[0])
    indice_ultima_var_HCE_y_reglas=df.columns.get_loc('diabetes_mayores_de_65_y_salud_muy_compleja') #AQUÍ SE COGE EL ÍNDICE DE LA *ÚLTIMA VARIABLE*
                                                                                                         #DE LA ÚTLIMA REGLA AÑADIDA. EN LA VERSIÓN 0.0.1 
                                                                                                         #CORRESPONDE A 'diabetes_mayores_de_65_y_salud_muy_compleja'
    df_hce=df.iloc[:,:indice_ultima_var_HCE_y_reglas+1]
    df_hce_modificado=df_hce
    if tipo!='personalizado':
        if tipo=='alimento':
            prefijo='al_'
        if tipo=='farmacos':
            prefijo='al_'
        if tipo=='sintomas':
            prefijo='al_'
        for k in range(indice_ultima_var_HCE_y_reglas,len(df.iloc[0])):
            longitud_df=len(df.iloc[0])
            nombre_var=df.columns[k]
            #Añadir columna con la nueva variable en el df HCE. Añadir también su valor
            if nombre_var[:3]=='al_':
                if all(df[nombre_var]==0)==False:
                    df_hce_modificado[nombre_var]=df[nombre_var]
                    df_final,_=correlacion_una_var_con_el_df(df_hce_modificado,nombre_var,N=20)
                    df_hce_modificado.drop([nombre_var],axis=1,inplace=True)
                    dataframes_agrupados.append(df_final)
            else:
                pass
        resultado_df = pd.concat(dataframes_agrupados, axis=0)
        return resultado_df
    if tipo=='personalizado':
        if vars==[]:
            print("DEbes introducir las variables en el parámetro vars de la función vars=['edad','acv']")
        else:
            for k in range(0,len(df.iloc[0])):
                longitud_df=len(df.iloc[0])
                nombre_var=df.columns[k]
                #Añadir columna con la nueva variable en el df HCE. Añadir también su valor
                for variable in vars:
                    if nombre_var==variable:
                        if all(df[nombre_var]==0)==False:
                            df_hce_modificado[nombre_var]=df[nombre_var]
                            df_final,_=correlacion_una_var_con_el_df(df_hce_modificado,nombre_var)
                            df_hce_modificado.drop([nombre_var],axis=1,inplace=True)
                            dataframes_agrupados.append(df_final)
                    else:
                        pass
            resultado_df = pd.concat(dataframes_agrupados, axis=0)
            return resultado_df
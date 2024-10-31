import json
import pandas as pd
import ollama
import re

def otras_ecv(df,variables=['otras_ecv']): #dummy vars
    df = pd.get_dummies(df, columns=variables)
    return df

def dx_principal(df,variables=['dx_principal']): #dummy vars
    df = pd.get_dummies(df, columns=variables)
    return df

def dx_secundario(df,variables=['dx_asociados','dx_asociados_2','dx_asociados_3','dx_asociados_4','dx_asociados_5']):
     # Crear un DataFrame vacío para las dummies
    dummy_df = pd.DataFrame(index=df.index)
    
    # Crear un conjunto para mantener un registro de los valores únicos ya procesados
    seen_values = set()
    
    # Procesar cada columna especificada
    for column in variables:
        if column in df.columns:
            for index, value in df[column].iteritems():
                # Formatear el nombre de la columna dummy basada en el valor del texto
                dummy_name = 'Diagnósticos asociados_' + str(value)
                
                # Revisar si el valor ya ha sido visto y usado para crear una columna dummy
                if dummy_name not in seen_values:
                    # Si no está en seen_values, lo añadimos y creamos la columna en dummy_df
                    seen_values.add(dummy_name)
                    dummy_df[dummy_name] = 0
                
                # Asignar 1 a la fila correspondiente en la columna dummy
                dummy_df.at[index, dummy_name] = 1

    # Concatenar el DataFrame original con el de dummies
    result_df = pd.concat([df, dummy_df], axis=1)
    result_df.drop(columns=variables, inplace=True)
    return result_df

def medicamentos(df,variables=['medicamento','medicamento_2','medicamento_3','medicamento_4','medicamento_5','medicamento_6',
                               'medicamento_7','medicamento_8','medicamento_9','medicamento_10','medicamento_11','medicamento_12']):
    # Crear un DataFrame vacío para las dummies
    dummy_df = pd.DataFrame(index=df.index)
    
    # Crear un conjunto para mantener un registro de los valores únicos ya procesados
    seen_values = set()
    
    # Procesar cada columna especificada
    for column in variables:
        if column in df.columns:
            for index, value in df[column].iteritems():
                # Formatear el nombre de la columna dummy basada en el valor del texto
                dummy_name = 'Medicamento_' + str(value)
                
                # Revisar si el valor ya ha sido visto y usado para crear una columna dummy
                if dummy_name not in seen_values:
                    # Si no está en seen_values, lo añadimos y creamos la columna en dummy_df
                    seen_values.add(dummy_name)
                    dummy_df[dummy_name] = 0
                
                # Asignar 1 a la fila correspondiente en la columna dummy
                dummy_df.at[index, dummy_name] = 1

    # Concatenar el DataFrame original con el de dummies
    result_df = pd.concat([df, dummy_df], axis=1)
    result_df.drop(columns=variables, inplace=True)
    return result_df

def rec_nutricionales(df,variable='Recomendaciones nutricionales recibidas'):
    #Hacer un prompt
    #Procesar la salida del prompt para quedarnos con una lista
    #Aplicarlo a todas las filas
    #***igual vale al pena conseguir las listas con GPT4o y Ctrl+C Ctrl+V
    #Aplicar dummy variables dada una lista ['alimento1','alimento2',...]
    # df = pd.get_dummies(df, columns=variables)
    unique_values = set([item for sublist in df[variable+'-Lista'] for item in sublist])

    # 2. Crear columnas dummy para cada valor único
    for value in unique_values:
        df[variable+'_'+value] = df[variable+'-Lista'].apply(lambda x: 1 if value in x else 0)
    df.drop(columns=[variable+'-Lista'], inplace=True)
    df.drop(columns=[variable], inplace=True)
    return df
    return df

def cons_dietetico(df,variable='Consejo dietético'):
    # df['cons_dietetico_1']=0
    # df['cons_dietetico_2']=0
    # df=df.drop('cons_dietetico', axis=1)
    # df = pd.get_dummies(df, columns=variables)
    # 1. Expandir todas las listas en la columna a una sola lista
    unique_values = set([item for sublist in df[variable+'-Lista'] for item in sublist])

    # 2. Crear columnas dummy para cada valor único
    for value in unique_values:
        df[variable+'_'+value] = df[variable+'-Lista'].apply(lambda x: 1 if value in x else 0)
    df.drop(columns=[variable+'-Lista'], inplace=True)
    df.drop(columns=[variable], inplace=True)
    return df

def juicio_cl(df):
    df['Juicio clínico.1']=0
    df['Juicio clínico.2']=0
    df=df.drop('Juicio clínico', axis=1)
    return df

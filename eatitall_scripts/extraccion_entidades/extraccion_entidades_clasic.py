import sys
sys.path.insert(1, '/home/eatitall_scripts')
sys.path.insert(1, '/home/root/pctobs/lib/python3.8/site-packages')
import pandas as pd
import json
import re
import unicodedata

# Función para calcular la distancia de Levenshtein
def distancia_levenshtein(s1, s2):
    if len(s1) < len(s2):
        return distancia_levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    matriz_previa = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        matriz_actual = [i + 1]
        for j, c2 in enumerate(s2):
            inserciones = matriz_previa[j + 1] + 1
            eliminaciones = matriz_actual[j] + 1
            sustituciones = matriz_previa[j] + (c1 != c2)
            matriz_actual.append(min(inserciones, eliminaciones, sustituciones))
        matriz_previa = matriz_actual
    
    return matriz_previa[-1]

# Función mejorada para encontrar alimentos considerando la distancia de Levenshtein
def encontrar_alimentos(df, vocab_alimentos, umbral_levenshtein=0):
    todos_los_alimentos = [item for sublist in vocab_alimentos.values() for item in sublist]
    for k in range(len(df)):
        texto_entrada = df['observaciones'][k].lower()
        texto_entrada=unicodedata.normalize('NFKD', texto_entrada).encode('ASCII', 'ignore').decode('ASCII') #Eliminar acentos
        palabras = re.split(r'[ ,.;]+', texto_entrada)
        alimentos_encontrados = []
        alimentos_levenshtein=[]
        palabras_exactas_pre_levenshtein=[]

        i = 0
        while i < len(palabras):
            max_longitud = 0
            alimento_a_agregar = ''
            for alimento in todos_los_alimentos:
                alimento_partes = re.split(r'[ ,.;]+',alimento.lower())
                longitud = len(alimento_partes)
                # Comprobar si la secuencia de palabras coincide con algún alimento
                if palabras[i:i+longitud] == alimento_partes and longitud > max_longitud:
                    # Guardar el alimento más largo que coincide
                    alimento_a_agregar = ' '.join(palabras[i:i+longitud])
                    max_longitud = longitud
            if max_longitud > 0:
                alimentos_encontrados.append(alimento_a_agregar)
                i += max_longitud
            else:
                for alimento in todos_los_alimentos:
                    alimento_partes = re.split(r'[ ,.;]+', alimento.lower())
                    longitud = len(alimento_partes)
                    secuencia_palabras = ' '.join(palabras[i:i+longitud])
                    distancia=distancia_levenshtein(secuencia_palabras,alimento)
                    if distancia<=2:
                        alimentos_levenshtein.append(alimento)
                        palabras_exactas_pre_levenshtein.append(secuencia_palabras)
                i += 1

        alimentos_string = ', '.join(alimentos_encontrados)
        df.loc[k, 'alimentos_encontrados'] = alimentos_string
        alimentos_levenshtein_string=', '.join(alimentos_levenshtein)
        df.loc[k, 'alimentos_encontrados_con_Levenshtein'] = alimentos_levenshtein_string
        palabras_exactas_pre_levenshtein_string=', '.join(palabras_exactas_pre_levenshtein)
        df.loc[k, 'palabras_exactas_pre_Levenshtein'] = palabras_exactas_pre_levenshtein_string
    return df


def nerc_diccionario(df,diccionario,tipo_entidad):
    if tipo_entidad=="alimentos":
        todos_los_elementos = [item for sublist in diccionario.values() for item in sublist]
        prefijo='al_'
    if tipo_entidad=="farmacos":
        prefijo="far_"
        todos_los_elementos = [elemento['nombre'] for sublist in diccionario.values() for elemento in sublist]
    if tipo_entidad=="sintomas":
        prefijo="sin_"
        todos_los_elementos = [elemento['nombre'] for sublist in diccionario.values() for elemento in sublist]
    if tipo_entidad=="pruebas clinicas":
        prefijo="pc_"
        todos_los_elementos = [elemento['nombre'] for sublist in diccionario.values() for elemento in sublist]

    for elemento in todos_los_elementos:
        elemento=elemento.lower()
        elemento=unicodedata.normalize('NFKD', elemento).encode('ASCII', 'ignore').decode('ASCII') #Eliminar acentos
        nombre_columna_entidad=prefijo+elemento
        df[nombre_columna_entidad]=0

    for k in range(0,len(df)):
        texto_entrada=df['observaciones'][k].lower()
        texto_entrada=unicodedata.normalize('NFKD', texto_entrada).encode('ASCII', 'ignore').decode('ASCII') #Eliminar acentos
        palabras = re.split(r'[ ,.;]+', texto_entrada)
        elementos_encontrados = []
        i = 0
        while i < len(palabras):
            max_longitud = 0
            elemento_a_agregar = ''
            for elemento in todos_los_elementos:
                elemento_partes = re.split(r'[ ,.;]+',elemento.lower())
                longitud = len(elemento_partes)
                # Comprobar si la secuencia de palabras coincide con algún alimento
                if palabras[i:i+longitud] == elemento_partes and longitud > max_longitud:
                    # Guardar el alimento más largo que coincide
                    elemento_a_agregar = ' '.join(palabras[i:i+longitud])
                    max_longitud = longitud
            if max_longitud > 0:
                elementos_encontrados.append(elemento_a_agregar)
                i += max_longitud  # Ajustar el índice según la longitud del alimento encontrado más largo
            else:
                i += 1
        # Generar una columna con las entidades reconocidas separadas por comas
        # elementos_string = ', '.join(elementos_encontrados)
        # nombre_columna='NERC '+tipo_entidad
        # df.loc[k,nombre_columna]=elementos_string
        for elemento2 in elementos_encontrados:
            if df[prefijo+elemento2][k]!=0:
                df[prefijo+elemento2][k]=df[prefijo+elemento2][k]+1
            if df[prefijo+elemento2][k]==0:        
                df[prefijo+elemento2][k]=1
    return df

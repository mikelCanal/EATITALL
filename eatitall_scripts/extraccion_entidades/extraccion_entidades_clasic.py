import sys
sys.path.insert(1, '/home/eatitall_scripts')
sys.path.insert(1, '/home/root/pctobs/lib/python3.8/site-packages')
import pandas as pd
import json
import re

### ESTO DEBE ESTAR EN EL CÓDIGO PRINCIPAL PARA INICIALIZAR LA VARIABLE TODOS LOS ALIMENTOS ###
# # Cargamos los NOMBRES de los paths
# data_path='./../../archivos/datos_con_10_ejemplos_reglas.csv'
# vocab_path='./../../archivos/vocab.json'

# # Cargamos los datos
# df = pd.read_csv(data_path)
# #Cargamos el vocabulario
# with open(vocab_path, 'r') as archivo:
#     vocab = json.load(archivo)

# todos_los_alimentos = [item for sublist in vocab.values() for item in sublist]
###############################################################################################

def encontrar_alimentos(df,vocab_alimentos):
    todos_los_alimentos = [item for sublist in vocab_alimentos.values() for item in sublist]
    for k in range(0,len(df)):
        texto_entrada=df['observaciones'][k]
        texto = texto_entrada.lower()
        palabras = re.split(r'[ ,.;]+', texto)
        alimentos_encontrados = []
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
                i += max_longitud  # Ajustar el índice según la longitud del alimento encontrado más largo
            else:
                i += 1
        alimentos_string = ', '.join(alimentos_encontrados)
        df.loc[k,'alimentos_encontrados']=alimentos_string
    return df

# def encontrar_alimentos_v0(texto_entrada,todos_los_alimentos):
#     texto = texto_entrada.lower()
#     palabras = re.split(r'[ ,.;]+', texto)
#     alimentos_encontrados = []
#     i = 0
#     while i < len(palabras):
#         max_longitud = 0
#         alimento_a_agregar = ''
#         for alimento in todos_los_alimentos:
#             alimento_partes = re.split(r'[ ,.;]+',alimento.lower())
#             longitud = len(alimento_partes)
#             # Comprobar si la secuencia de palabras coincide con algún alimento
#             if palabras[i:i+longitud] == alimento_partes and longitud > max_longitud:
#                 # Guardar el alimento más largo que coincide
#                 alimento_a_agregar = ' '.join(palabras[i:i+longitud])
#                 max_longitud = longitud
#         if max_longitud > 0:
#             alimentos_encontrados.append(alimento_a_agregar)
#             i += max_longitud  # Ajustar el índice según la longitud del alimento encontrado más largo
#         else:
#             i += 1
#     return alimentos_encontrados

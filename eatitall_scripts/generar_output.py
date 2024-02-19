# ASEGURARSE DE QUE EL ARCHIVO config.json ESTÉ ACTUALIZADO.
import sys

sys.path.insert(1, '/home/eatitall_scripts')
sys.path.insert(1, '/home/root/pctobs/lib/python3.8/site-packages')
import json
import pandas as pd
from definir_reglas import reglas_2023
from normalizar_datos import normalizar_datos, nuevas_variables
from extraccion_entidades import extraccion_entidades_clasic

# Cargamos los NOMBRES de los paths
config_path='./definir_reglas/config.json'
input_path='./archivos/datos_con_10_ejemplos.csv'
output_con_reglas_path='./archivos/datos_con_10_ejemplos_reglas.csv'
output_con_reglas_y_extraccion_entidades_path='./archivos/datos_con_10_ejemplos_reglas_y_extraccion_entidades.csv'
vocab_alimentos_path='./archivos/vocab_alimentos.json'

# Cargamos los NOMBRES de las variables
with open(config_path, 'r') as archivo:
    config = json.load(archivo)
# Cargamos el VOCABULARIO DE LOS ALIMENTOS
with open(vocab_alimentos_path, 'r') as archivo:
    vocab_alimentos = json.load(archivo)

#############################
### NORMALIZACIÓN DE DATOS ###
#############################
# Cargamos el archivo CSV
df = pd.read_csv(input_path)
#Añadimos nuevas variables
df=nuevas_variables.imc(df,config)
#Convertimos Si/No a 0/1
df=normalizar_datos.convertir_si_no_a_0_1(df)

#############################
### DEFINICIÓN REGLAS ###
#############################
df=reglas_2023.añadir_reglas(df,config)
# Generamos un csv con los datos generados con las reglas y lo guardamos
df.to_csv(output_con_reglas_path, index=False, decimal='.')

#############################
### EXTRACCIÓN ENTIDADES ###
#############################
df=extraccion_entidades_clasic.encontrar_alimentos(df,vocab_alimentos)
# Generamos un csv con los datos generados con las reglas y lo guardamos
df.to_csv(output_con_reglas_y_extraccion_entidades_path, index=False)
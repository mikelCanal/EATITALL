import pandas as pd
from datetime import datetime
import ollama
import re

def convertir_si_no_a_0_1(df):
    columnas_si_no = []
    for col in df.columns:
        if df[col].dtype == 'object':  # Verificar si la columna es de tipo object (potencialmente string)
            valores_unicos = df[col].dropna().unique()  # Obtener valores únicos sin contar NaN
            if set(valores_unicos) <= {'Sí', 'No'}:  # Verificar si todos los valores son 'Sí' o 'No'
                columnas_si_no.append(col)

    for columna in columnas_si_no:
        df[columna] = df[columna].map({'Sí': 1, 'No': 0})
    return df

def convertir_mujer_hombre_a_1_2(df):
    columnas_12 = []
    for col in df.columns:
        if df[col].dtype == 'object':  # Verificar si la columna es de tipo object (potencialmente string)
            valores_unicos = df[col].dropna().unique()  # Obtener valores únicos sin contar NaN
            if set(valores_unicos) <= {'Mujer', 'Hombre'}:  # Verificar si todos los valores son 'Mujer' o 'Hombre'
                columnas_12.append(col)

    for columna in columnas_12:
        df[columna] = df[columna].map({'Mujer': 1, 'Hombre': 2})
    return df

def convertir_lugar_nac_a_numero(df):
    columnas_1234 = []
    for col in df.columns:
        if df[col].dtype == 'object':  # Verificar si la columna es de tipo object (potencialmente string)
            valores_unicos = df[col].dropna().unique()  # Obtener valores únicos sin contar NaN
            if set(valores_unicos) <= {'Europa', 'África','Sudamérica','Asia'}:  # Verificar si todos los valores son 'Mujer' o 'Hombre'
                columnas_1234.append(col)

    for columna in columnas_1234:
        df[columna] = df[columna].map({'Europa': 1, 'África': 2, 'Sudamérica':3, 'Asia':4})
    return df

def convertir_dm_tipo_a_numero(df):
    columnas_1234 = []
    for col in df.columns:
        if df[col].dtype == 'object':  # Verificar si la columna es de tipo object (potencialmente string)
            valores_unicos = df[col].dropna().unique()  # Obtener valores únicos sin contar NaN
            if set(valores_unicos) <= {'Tipo 1', 'Tipo 2','Otros tipos diabetes','No diabetes'}:  # Verificar si todos los valores son 'Mujer' o 'Hombre'
                columnas_1234.append(col)

    for columna in columnas_1234:
        df[columna] = df[columna].map({'Tipo 1': 1, 'Tipo 2': 2, 'Otros tipos diabetes':3, 'No diabetes':4})
    return df

def convertir_ic_tipo_a_numero(df):
    columnas_123 = []
    for col in df.columns:
        if df[col].dtype == 'object':  # Verificar si la columna es de tipo object (potencialmente string)
            valores_unicos = df[col].dropna().unique()  # Obtener valores únicos sin contar NaN
            if set(valores_unicos) <= {'FE preservada', 'FE reducida','FE ligeramente'}:  # Verificar si todos los valores son 'Mujer' o 'Hombre'
                columnas_123.append(col)

    for columna in columnas_123:
        df[columna] = df[columna].map({'FE preservada': 1, 'FE reducida': 2, 'FE ligeramente':3})
    return df

def convertir_careme_estadio_a_numero(df):
    columnas_12345 = []
    for col in df.columns:
        if df[col].dtype == 'object':  # Verificar si la columna es de tipo object (potencialmente string)
            valores_unicos = df[col].dropna().unique()  # Obtener valores únicos sin contar NaN
            if set(valores_unicos) <= {'ckm 1', 'ckm 2','ckm 3','ckm4a','ckm4b'}:  # Verificar si todos los valores son 'Mujer' o 'Hombre'
                columnas_12345.append(col)

    for columna in columnas_12345:
        df[columna] = df[columna].map({'ckm 1': 1, 'ckm 2': 2, 'ckm 3':3, 'ckm4a':4, 'ckm4b':5})
    return df

def convertir_intolerancias_a_numero(df):
    columnas_12345 = []
    for col in df.columns:
        if df[col].dtype == 'object':  # Verificar si la columna es de tipo object (potencialmente string)
            valores_unicos = df[col].dropna().unique()  # Obtener valores únicos sin contar NaN
            if set(valores_unicos) <= {'Lactosa', 'Fructosa','Sorbitol','Gluten','Alergia a la proteína de la vaca'}:  # Verificar si todos los valores son 'Mujer' o 'Hombre'
                columnas_12345.append(col)

    for columna in columnas_12345:
        df[columna] = df[columna].map({'Lactosa': 1, 'Fructosa': 2, 'Sorbitol':3, 'Gluten':4, 'Alergia a la proteína de la vaca':5})
    return df


def dummie_vars(df,variables_a_excluir=['rec_nutricionales','cons_dietetico','ejer_fisico','juicio_cl']):
    # Identificar las columnas de tipo string
    string_columns = df.select_dtypes(include=['object']).columns
    string_columns_def=[]
    for variable in string_columns:
        if variable not in variables_a_excluir:
            print("variable: ",variable)
            string_columns_def.append(variable)
    # Generar variables dummy para las columnas de tipo string
    df_dummies = pd.get_dummies(df, columns=string_columns_def)#, drop_first=True)
    return df_dummies

def eliminar_variables(df,variables=['record_id','codigo_dig','codigo_pac','pa'],eliminar_nans=True):
    variables_existentes=[]
    for variable in variables:
        if variable in df.columns:
            variables_existentes.append(variable)
    df = df.drop(columns=variables_existentes)
    if eliminar_nans==True:
        df = df.dropna(axis=1, how='all')
    return df


def gestionar_nans(df):
    # df.fillna(-10000, inplace=True)
    for column in df.columns:
        if df[column].isnull().all():  # Check if the column is entirely NaN
            df[column].fillna(-100, inplace=True)
        elif df[column].dtype == 'float64' or df[column].dtype == 'float32':  # Check if the column is decimal
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
        elif df[column].dtype == 'object' or df[column].dtype == 'category':  # Check if the column is categorical
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)
    return df

def comas_a_puntos(df,variables):
    for column in variables:
        if column in df.columns:
            # Convertir cada valor a string, reemplazar comas por puntos y convertir a float
            df[column] = df[column].astype(str).str.replace(',', '.').astype(float)
    return df

def ajustar_decimal_dividir_por_mil(df,variables=['fib4']):
    for column in variables:
        if column in df.columns:
            # Aplicar la transformación solo a los valores que sean mayores que 10
            df[column] = df[column].apply(lambda x: x / 1000 if x > 10 else x)
    return df
                                    
def ajustar_punto_decimal(df,variables):
    for column_name in variables:
        if column_name in df.columns:
            # Iterar sobre cada fila del DataFrame en la columna especificada
            for index, value in df[column_name].iteritems():
                # Comprobar si el valor termina en ".000" (como string)
                if isinstance(value, float) and '{:.3f}'.format(value).endswith('.000'):
                    # Dividir el valor por 1000 si termina en ".000"
                    df.at[index, column_name] = value / 1000
    return df

def fecha_a_timestamp(df,variables):
    for column_name in variables:
        if column_name in df.columns:
            # Convertir las fechas a formato 'aaaammdd' como entero
            df[column_name] = pd.to_datetime(df[column_name])
            df[column_name] = df[column_name].apply(lambda x: x.timestamp())
    return df


def texto_libre_a_categorias(df,variable='Consejo dietético',categorias=""):
    ollama_server = 'https://ollama.tools.gplsi.es'
    ollama_client = ollama.Client(host=ollama_server)
    ollama_model = 'llama3.2'
    ollama_context = 32768
    df[variable+'-Lista']=0
    for k in range(0,len(df)):
        frase=df[variable][k]
        if pd.notnull(frase):  
            messages = [
            {
                'role': 'user',
                'content': f"""
                Eres un modelo experto en la extracción de entidades relacionadas con características nutricionales a partir de historiales clínicos electrónicos. Tu tarea es, dado un conjunto de características y una frase, identificar todas las características relevantes mencionadas en dicha frase. El output debe estar en formato de lista de python "[]", separando las categorías por comas. Si en el texto no identificas nignuna de las categorías de la lista, NO te inventes categorías, en su lugar deja una lista vacía: [].
                A continuación, te proporcionaré un listado de categorías y las palabras clave o descripción que pertenecen a cada una de ellas en el fromato "categoría:descripción". El nombre de cada categoría, formado por palabras separadas por guiones bajos, también es una indicación de su contenido:
                {categorias}.
                En tu respuesta dame solo las categorías, no me des las descripciones.
                Esta es la frase que debes analizar:{frase}
                Esta es la respuesta:
                [
                """
                        ,        
                    },
                ]
            response = ollama_client.chat(model=ollama_model, options={ 'num_ctx': ollama_context }, messages=messages)
            text=response['message']['content']
            print(text)
            lista_texto = re.findall(r'\[([^\]]+)\]', text)

            # Convertir a lista de Python
            c=0
            while True:
                try:
                    if lista_texto:
                        lista_python = [item.strip().strip('"') for item in lista_texto[0].split(',')]
                        df[variable+'-Lista'][k]=lista_python
                        break
                    else:
                        if c==5:
                            df[variable+'-Lista'][k]=[]
                            break
                        print("No se encontró ninguna lista en el texto.")
                        c=c+1
                except:
                    pass
        else:
            df[variable+'-Lista'][k]=[]
    # return df
    return df

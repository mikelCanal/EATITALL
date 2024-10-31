
#############################################################################################################################################
######## Reglas extraídas del documento "6. Glycemic Goals and Hypoglycemia: Standards of Care in Diabetes — 2024" (2.limites glucemia) ########
#############################################################################################################################################

import pandas as pd

def añadir_reglas(df:pd.DataFrame,config):
    df,_=corr_hba1c_fg(df,config, path_variables_ausentes='variables_ausentes.txt')
    df,_=bajo_nivel_insulina(df,config, path_variables_ausentes='variables_ausentes.txt')
    df,_=hipoglucemia(df,config, path_variables_ausentes='variables_ausentes.txt')
    return df

# página 2
def corr_hba1c_fg(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
    # Extracción de las variables y sus umbrales del diccionario de configuración
    hba1c = config['variables']['analisis_sangre']['hba1c']['siglas dataset']
    fg = config['variables']['analisis_sangre']['fg']['siglas dataset']

    # Parámetros de umbrales y porcentajes para diferentes rangos de HbA1c
    porcentaje_hba1c_5 = config['parametros']['corr_hba1c_fg']['porcentaje_hba1c_5']
    hba1c_5_umbral_inferior = config['parametros']['corr_hba1c_fg']['hba1c_5_umbral_inferior']
    hba1c_5_umbral_superior = config['parametros']['corr_hba1c_fg']['hba1c_5_umbral_superior']
    porcentaje_hba1c_6 = config['parametros']['corr_hba1c_fg']['porcentaje_hba1c_6']
    hba1c_6_umbral_inferior = config['parametros']['corr_hba1c_fg']['hba1c_6_umbral_inferior']
    hba1c_6_umbral_superior = config['parametros']['corr_hba1c_fg']['hba1c_6_umbral_superior']
    porcentaje_hba1c_7 = config['parametros']['corr_hba1c_fg']['porcentaje_hba1c_7']
    hba1c_7_umbral_inferior = config['parametros']['corr_hba1c_fg']['hba1c_7_umbral_inferior']
    hba1c_7_umbral_superior = config['parametros']['corr_hba1c_fg']['hba1c_7_umbral_superior']
    porcentaje_hba1c_8 = config['parametros']['corr_hba1c_fg']['porcentaje_hba1c_8']
    hba1c_8_umbral_inferior = config['parametros']['corr_hba1c_fg']['hba1c_8_umbral_inferior']
    hba1c_8_umbral_superior = config['parametros']['corr_hba1c_fg']['hba1c_8_umbral_superior']
    porcentaje_hba1c_9 = config['parametros']['corr_hba1c_fg']['porcentaje_hba1c_9']
    hba1c_9_umbral_inferior = config['parametros']['corr_hba1c_fg']['hba1c_9_umbral_inferior']
    hba1c_9_umbral_superior = config['parametros']['corr_hba1c_fg']['hba1c_9_umbral_superior']
    porcentaje_hba1c_10 = config['parametros']['corr_hba1c_fg']['porcentaje_hba1c_10']
    hba1c_10_umbral_inferior = config['parametros']['corr_hba1c_fg']['hba1c_10_umbral_inferior']
    hba1c_10_umbral_superior = config['parametros']['corr_hba1c_fg']['hba1c_10_umbral_superior']
    porcentaje_hba1c_11 = config['parametros']['corr_hba1c_fg']['porcentaje_hba1c_11']
    hba1c_11_umbral_inferior = config['parametros']['corr_hba1c_fg']['hba1c_11_umbral_inferior']
    hba1c_11_umbral_superior = config['parametros']['corr_hba1c_fg']['hba1c_11_umbral_superior']
    porcentaje_hba1c_12 = config['parametros']['corr_hba1c_fg']['porcentaje_hba1c_12']
    hba1c_12_umbral_inferior = config['parametros']['corr_hba1c_fg']['hba1c_12_umbral_inferior']
    hba1c_12_umbral_superior = config['parametros']['corr_hba1c_fg']['hba1c_12_umbral_superior']

    df['reglas corr hba1c fg'] = 0  # Inicialización de la columna
    
    # Verificar la existencia de las variables antes del bucle
    missing_vars = set()
    available_vars = []

    required_vars = [hba1c, fg]
    
    for var in required_vars:
        if var not in df.columns:
            missing_vars.add(var)
        else:
            available_vars.append(var)
    
    # Si hay variables faltantes, escribirlas en un archivo de texto
    if missing_vars:
        with open(path_variables_ausentes, 'w') as f:
            for var in missing_vars:
                f.write(f"No se encuentra la variable '{var}' en el dataset.\n")
    
    # Procesamiento de las filas
    for k in range(len(df)):
        try:
            hba1c_value = df[hba1c][k]
            fg_value = df[fg][k]
            if (hba1c_value >= porcentaje_hba1c_5 - 0.5 and hba1c_value < porcentaje_hba1c_5 + 0.5 and 
                fg_value >= hba1c_5_umbral_inferior and fg_value <= hba1c_5_umbral_superior):
                df['reglas corr hba1c fg'][k] = 1
            elif (hba1c_value >= porcentaje_hba1c_6 - 0.5 and hba1c_value < porcentaje_hba1c_6 + 0.5 and 
                  fg_value >= hba1c_6_umbral_inferior and fg_value <= hba1c_6_umbral_superior):
                df['reglas corr hba1c fg'][k] = 1
            elif (hba1c_value >= porcentaje_hba1c_7 - 0.5 and hba1c_value < porcentaje_hba1c_7 + 0.5 and 
                  fg_value >= hba1c_7_umbral_inferior and fg_value <= hba1c_7_umbral_superior):
                df['reglas corr hba1c fg'][k] = 1
            elif (hba1c_value >= porcentaje_hba1c_8 - 0.5 and hba1c_value < porcentaje_hba1c_8 + 0.5 and 
                  fg_value >= hba1c_8_umbral_inferior and fg_value <= hba1c_8_umbral_superior):
                df['reglas corr hba1c fg'][k] = 1
            elif (hba1c_value >= porcentaje_hba1c_9 - 0.5 and hba1c_value < porcentaje_hba1c_9 + 0.5 and 
                  fg_value >= hba1c_9_umbral_inferior and fg_value <= hba1c_9_umbral_superior):
                df['reglas corr hba1c fg'][k] = 1
            elif (hba1c_value >= porcentaje_hba1c_10 - 0.5 and hba1c_value < porcentaje_hba1c_10 + 0.5 and 
                  fg_value >= hba1c_10_umbral_inferior and fg_value <= hba1c_10_umbral_superior):
                df['reglas corr hba1c fg'][k] = 1
            elif (hba1c_value >= porcentaje_hba1c_11 - 0.5 and hba1c_value < porcentaje_hba1c_11 + 0.5 and 
                  fg_value >= hba1c_11_umbral_inferior and fg_value <= hba1c_11_umbral_superior):
                df['reglas corr hba1c fg'][k] = 1
            elif (hba1c_value >= porcentaje_hba1c_12 - 0.5 and hba1c_value < porcentaje_hba1c_12 + 0.5 and 
                  fg_value >= hba1c_12_umbral_inferior and fg_value <= hba1c_12_umbral_superior):
                df['reglas corr hba1c fg'][k] = 1
        except KeyError:
            print(f"No se encuentra la variable {e} en el dataset.")
        except Exception as e:
            print(f"Error procesando la fila {k}: {e}")

    # Crear el nuevo DataFrame con las variables identificadas y 'corr_hba1c_fg'
    available_vars.append('reglas corr hba1c fg')
    df_vars_identificadas = df[available_vars]
    
    return df, df_vars_identificadas


# página 2
def bajo_nivel_insulina(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
    insulina = config['variables']['analisis_sangre']['insulina']['siglas dataset']
    insulina_umbral_inferior = config['parametros']['bajo_nivel_insulina']['insulina_umbral_inferior']

    df[f'reglas bajo nivel insulina ({insulina}<{insulina_umbral_inferior})'] = 0

    missing_vars = set()
    available_vars = []

    required_vars = [insulina]
    
    for var in required_vars:
        if var not in df.columns:
            missing_vars.add(var)
        else:
            available_vars.append(var)

    if missing_vars:
        with open(path_variables_ausentes, 'w') as f:
            for var in missing_vars:
                f.write(f"No se encuentra la variable '{var}' en el dataset.\n")

    for k in range(len(df)):
        try:
            if (df[insulina][k] < insulina_umbral_inferior and df[insulina][k]>0)==True:
                df[f'reglas bajo nivel insulina ({insulina}<{insulina_umbral_inferior})'][k] = 1
        except:
            pass

    available_vars.append(f'reglas bajo nivel insulina ({insulina}<{insulina_umbral_inferior})')
    df_vars_identificadas = df[available_vars]

    return df, df_vars_identificadas


# página 8
def hipoglucemia(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
    glucosa = config['variables']['analisis_sangre']['glucosa']['siglas dataset']
    glucosa_nivel1_umbral_inferior = config['parametros']['hipoglucemia']['nivel1_glucosa_umbral_inferior']
    glucosa_nivel1_umbral_superior = config['parametros']['hipoglucemia']['nivel1_glucosa_umbral_superior']
    glucosa_nivel2_umbral_superior = config['parametros']['hipoglucemia']['nivel2_glucosa_umbral_superior']

    df['reglas hipoglucemia'] = 0

    missing_vars = set()
    available_vars = []

    required_vars = [glucosa]
    
    for var in required_vars:
        if var not in df.columns:
            missing_vars.add(var)
        else:
            available_vars.append(var)

    if missing_vars:
        with open(path_variables_ausentes, 'w') as f:
            for var in missing_vars:
                f.write(f"No se encuentra la variable '{var}' en el dataset.\n")

    for k in range(len(df)):
        try:
            if df[glucosa][k] >= glucosa_nivel1_umbral_inferior and df[glucosa][k] < glucosa_nivel1_umbral_superior:
                df['reglas hipoglucemia'][k] = 1
            if df[glucosa][k] < glucosa_nivel2_umbral_superior:
                df['reglas hipoglucemia'][k] = 2
        except:
            pass

    available_vars.append('reglas hipoglucemia')
    df_vars_identificadas = df[available_vars]

    return df, df_vars_identificadas



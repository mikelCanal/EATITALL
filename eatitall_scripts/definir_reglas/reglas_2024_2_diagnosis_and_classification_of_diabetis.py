
#############################################################################################################################################
# Reglas extraídas del documento "2. Diagnosis and Classification of Diabetes: Standards of Care in Diabetes — 2024" (1.diagnosis y classif DM)#
#############################################################################################################################################

import pandas as pd

def añadir_reglas(df:pd.DataFrame,config):
    df,_=hiperglucemia(df,config, path_variables_ausentes='variables_ausentes.txt')
    df,_=diabetes(df,config, path_variables_ausentes='variables_ausentes.txt')
    df,_=prediabetes(df,config, path_variables_ausentes='variables_ausentes.txt')
    df,_=riesgo_prediabetes(df,config, path_variables_ausentes='variables_ausentes.txt')
    return df

# en las página 2
def hiperglucemia(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
    ga = config['variables']['analisis_sangre']['ga']['siglas dataset']  # se asume 1que ahora ga está definido en config
    #ga=random glucemia_ayunas

    ga_umbral_inferior_hiperglucemia = config['parametros']['hiperglucemia']['ga_umbral_inferior']
    
    df[f'reglas hiperglucemia ({ga}>{ga_umbral_inferior_hiperglucemia})'] = 0
    
    # Verificar la existencia de la variable antes del bucle
    missing_vars = set()
    available_vars = []

    required_vars = [ga]

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
    
    # Iterar sobre las filas solo si no hay variables faltantes
    for k in range(len(df)):
        try:
            if round(df[ga][k], 0) > ga_umbral_inferior_hiperglucemia:
                df[f'reglas hiperglucemia ({ga}>{ga_umbral_inferior_hiperglucemia})'][k] = 1
        except:
            pass  # Ya hemos manejado las variables faltantes, no necesitamos hacer nada aquí

    # Crear el nuevo DataFrame con las variables identificadas y 'hiperglucemia'
    available_vars.append(f'reglas hiperglucemia ({ga}>{ga_umbral_inferior_hiperglucemia})')
    df_vars_identificadas = df[available_vars]
    
    return df, df_vars_identificadas

# en la página 2
def diabetes(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
    hba1c = config['variables']['analisis_sangre']['hba1c']['siglas dataset']
    fg = config['variables']['analisis_sangre']['fg']['siglas dataset']
    pg2h_75g_ogtt = config['variables']['analisis_sangre']['pg2h_75g_ogtt']['siglas dataset']
    ga = config['variables']['analisis_sangre']['ga']['siglas dataset']
    hiperglucemia = config['variables']['situacion_clinica']['diagnosticos']['hiperglucemia']['siglas dataset']
    
    hba1c_umbral_inferior_diabetes = config['parametros']['diabetes']['hba1c_umbral_inferior']
    fg_umbral_inferior_diabetes = config['parametros']['diabetes']['fg_umbral_inferior'] # A fecha de 25/04/2024 fg_umbral=110 porque lo dice Kamila
    pg2h_75g_ogtt_umbral_inferior_diabetes = config['parametros']['diabetes']['pg2h_75g_ogtt_umbral_inferior']
    ga_umbral_inferior_diabetes = config['parametros']['diabetes']['ga_umbral_inferior']
    
    df['reglas diabetes'] = 0 # Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    
    missing_vars = set()
    available_vars=[]

    required_vars = [hba1c, fg, pg2h_75g_ogtt, ga, hiperglucemia]
    
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
    
    # Iterar sobre las filas solo si no hay variables faltantes
    for k in range(0, len(df)):
        try:
            if round(df[hba1c][k],1) >= hba1c_umbral_inferior_diabetes:
                df['reglas diabetes'][k] = 1
        except:
            pass
        
        try:
            if round(df[fg][k],0) >= fg_umbral_inferior_diabetes:
                df['reglas diabetes'][k] = 1
        except:
            pass
        
        try:
            if round(df[pg2h_75g_ogtt][k],0) >= pg2h_75g_ogtt_umbral_inferior_diabetes:
                df['reglas diabetes'][k] = 1
        except:
            pass
        
        try:
            if round(df[ga][k],0) >= ga_umbral_inferior_diabetes:
                df['reglas diabetes'][k] = 1
        except:
            pass
        try:
            if df[hiperglucemia][k]==1:
                df['reglas diabetes'][k] = 1
        except:
            pass

        df['reglas diabetes'] = 1 # Todos son diabéticos


    # Crear el nuevo DataFrame con las variables identificadas y 'diabetes'
    available_vars.append('reglas diabetes')
    df_vars_identificadas = df[available_vars]
    
    return df, df_vars_identificadas



#en la página 2
def prediabetes(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
    hba1c = config['variables']['analisis_sangre']['hba1c']['siglas dataset']
    fg = config['variables']['analisis_sangre']['fg']['siglas dataset']
    pg2h_75g_ogtt = config['variables']['analisis_sangre']['pg2h_75g_ogtt']['siglas dataset']
    diabetes = config['variables']['situacion_clinica']['diagnosticos']['diabetes']['siglas dataset']
    
    hba1c_umbral_inferior = config['parametros']['prediabetes']['hba1c_umbral_inferior']
    hba1c_umbral_superior = config['parametros']['prediabetes']['hba1c_umbral_superior']
    fg_umbral_inferior_prediabetes = config['parametros']['prediabetes']['fg_umbral_inferior']
    fg_umbral_superior_prediabetes = config['parametros']['prediabetes']['fg_umbral_superior']
    pg2h_75g_ogtt_umbral_inferior_prediabetes = config['parametros']['prediabetes']['pg2h_75g_ogtt_umbral_inferior']
    pg2h_75g_ogtt_umbral_superior_prediabetes = config['parametros']['prediabetes']['pg2h_75g_ogtt_umbral_superior']
    
    df['reglas prediabetes'] = 0
    
    missing_vars = set()
    available_vars = []

    required_vars = [hba1c, fg, pg2h_75g_ogtt, diabetes]
    
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
    
    # Iterar sobre las filas solo si no hay variables faltantes
    for k in range(len(df)):
        try:
            if df[diabetes][k]==0:
                try:
                    if round(df[hba1c][k], 1) >= hba1c_umbral_inferior and round(df[hba1c][k], 1) <= hba1c_umbral_superior:
                        df['reglas prediabetes'][k] = 1
                except:
                    pass  # Ya hemos manejado las variables faltantes, no necesitamos hacer nada aquí
                try:
                    if round(df[fg][k], 0) >= fg_umbral_inferior_prediabetes and round(df[fg][k], 0) <= fg_umbral_superior_prediabetes:
                        df['reglas prediabetes'][k] = 1
                except:
                    pass
                try:
                    if round(df[pg2h_75g_ogtt][k], 0) >= pg2h_75g_ogtt_umbral_inferior_prediabetes and round(df[pg2h_75g_ogtt][k], 0) <= pg2h_75g_ogtt_umbral_superior_prediabetes:
                        df['reglas prediabetes'][k] = 1
                except:
                    pass
        except:
            pass
    
    # Crear el nuevo DataFrame con las variables identificadas y 'prediabetes'
    available_vars.append('reglas prediabetes')
    df_vars_identificadas = df[available_vars]
    
    return df, df_vars_identificadas


# en la página 8
def riesgo_prediabetes(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt') -> pd.DataFrame:
    # Extracción de variables específicas de la configuración
    imc = config['variables']['mediciones_generales']['imc']['siglas dataset']
    tas = config['variables']['mediciones_generales']['tas']['siglas dataset']
    tad = config['variables']['mediciones_generales']['tad']['siglas dataset']
    tg = config['variables']['analisis_sangre']['tg']['siglas dataset']
    hba1c = config['variables']['analisis_sangre']['hba1c']['siglas dataset']
    hdl = config['variables']['analisis_sangre']['hdl']['siglas dataset']
    gdm = config['variables']['situacion_clinica']['diagnosticos']['gdm']['siglas dataset']
    vih = config['variables']['situacion_clinica']['diagnosticos']['vih']['siglas dataset']
    pancreatitis = config['variables']['situacion_clinica']['diagnosticos']['pancreatitis']['siglas dataset']
    acanthosis_nigricans = config['variables']['situacion_clinica']['diagnosticos']['acanthosis_nigricans']['siglas dataset']
    acv = config['variables']['situacion_clinica']['diagnosticos']['acv']['siglas dataset']#-->
    polycystic_ovary_sindrome = config['variables']['situacion_clinica']['diagnosticos']['polycystic_ovary_sindrom']['siglas dataset']
    physical_inactivity = config['variables']['informacion_basica']['phisical_inactivity']['siglas dataset']
    diabetes = config['variables']['situacion_clinica']['diagnosticos']['diabetes']['siglas dataset']

    # Umbral y identificadores de riesgo
    imc_umbral_inferior = config['parametros']['riesgo_prediabetes']['imc_umbral_inferior']
    tas_umbral_inferior = config['parametros']['riesgo_prediabetes']['tas_umbral_inferior']
    tad_umbral_inferior = config['parametros']['riesgo_prediabetes']['tad_umbral_inferior']
    tg_umbral_inferior = config['parametros']['riesgo_prediabetes']['tg_umbral_inferior']
    hba1c_umbral_inferior = config['parametros']['riesgo_prediabetes']['hba1c_umbral_inferior']
    hdl_umbral_superior = config['parametros']['riesgo_prediabetes']['hdl_umbral_superior']
    acv_identificador = config['parametros']['riesgo_prediabetes']['acv_identificador']
    gdm_identificador = config['parametros']['riesgo_prediabetes']['gdm_identificador']
    vih_identificador = config['parametros']['riesgo_prediabetes']['vih_identificador']
    pancreatitis_identificador = config['parametros']['riesgo_prediabetes']['pancreatitis_identificador']
    acanthosis_nigricans_identificador = config['parametros']['riesgo_prediabetes']['acanthosis_nigricans_identificador']
    polycystic_ovary_sindrome_identificador = config['parametros']['riesgo_prediabetes']['polycystic_ovary_sindrome_identificador']
    physical_inactivity_identificador = config['parametros']['riesgo_prediabetes']['phisical_inactivity_identificador']

    df['reglas riesgo prediabetes'] = 0

    # Verificación inicial de las variables requeridas en el DataFrame
    missing_vars = set()
    available_vars = []

    if diabetes not in df.columns:
        missing_vars.add(diabetes)
    else:
        available_vars.append(diabetes)

    for variable, alias in {'imc': imc, 'tas': tas, 'tad': tad, 'tg': tg, 'hba1c': hba1c, 'hdl': hdl,
                            'gdm': gdm, 'vih': vih, 'pancreatitis': pancreatitis, 'acanthosis_nigricans': acanthosis_nigricans,
                            'acv': acv, 'polycystic_ovary_sindrome': polycystic_ovary_sindrome,
                            'physical_inactivity': physical_inactivity}.items():
        if alias not in df.columns:
            missing_vars.add(alias)
        else:
            available_vars.append(alias)

    # Escritura de variables faltantes a un archivo
    if missing_vars:
        with open(path_variables_ausentes, 'w') as f:
            for var in missing_vars:
                f.write(f"No se encuentra la variable '{var}' en el dataset.\n")
    
    # Procesamiento condicional solo si todas las variables están presentes
    for k in range(len(df)):
        try:
            if df[diabetes][k]==0 and df['prediabetes'][k]==0:
                try:
                    if round(df[imc][k], 0) >= imc_umbral_inferior:
                        df['reglas riesgo prediabetes'][k] = 1
                except:
                    pass
                try:
                    if tas in df.columns and round(df[tas][k], 0) >= tas_umbral_inferior:
                        df['reglas riesgo prediabetes'][k] = 1
                except:
                    pass
                try:
                    if tad in df.columns and round(df[tad][k], 0) >= tad_umbral_inferior:
                        df['reglas riesgo prediabetes'][k] = 1
                except:
                    pass
                try:
                    if tg in df.columns and round(df[tg][k], 0) >= tg_umbral_inferior:
                        df['reglas riesgo prediabetes'][k] = 1
                except:
                    pass
                try:
                    if hba1c in df.columns and round(df[hba1c][k], 1) >= hba1c_umbral_inferior:
                        df['reglas riesgo prediabetes'][k] = 1
                except:
                    pass
                try:
                    if hdl in df.columns and round(df[hdl][k], 1) <= hdl_umbral_superior:
                        df['reglas riesgo prediabetes'][k] = 1
                except:
                    pass
                try:
                    if acv in df.columns and df[acv][k] == acv_identificador:
                        df['reglas riesgo prediabetes'][k] = 1
                except:
                    pass
                try:
                    if gdm in df.columns and df[gdm][k] == gdm_identificador:
                        df['reglas riesgo prediabetes'][k] = 1
                except:
                    pass
                try:
                    if vih in df.columns and df[vih][k] == vih_identificador:
                        df['reglas riesgo prediabetes'][k] = 1
                except:
                    pass
                try:
                    if pancreatitis in df.columns and df[pancreatitis][k] == pancreatitis_identificador:
                        df['reglas riesgo prediabetes'][k] = 1
                except:
                    pass
                try:
                    if acanthosis_nigricans in df.columns and df[acanthosis_nigricans][k] == acanthosis_nigricans_identificador:
                        df['reglas riesgo prediabetes'][k] = 1
                except:
                    pass
                try:
                    if polycystic_ovary_sindrome in df.columns and df[polycystic_ovary_sindrome][k] == polycystic_ovary_sindrome_identificador:
                        df['reglas riesgo prediabetes'][k] = 1
                except:
                    pass
                try:
                    if physical_inactivity in df.columns and df[physical_inactivity][k] == physical_inactivity_identificador:
                        df['reglas riesgo prediabetes'][k] = 1
                except:
                    pass
        except:
            pass

    # Crear un nuevo DataFrame con las variables identificadas y 'riesgo_prediabetes'
    available_vars.append('reglas riesgo prediabetes')
    df_vars_identificadas = df[available_vars]
    
    return df, df_vars_identificadas


# en las página 11
# def riesgo_diabetes_con_vih(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
#     # Extracción de la variable VIH y su identificador del diccionario de configuración
#     vih = config['variables']['situacion_clinica']['diagnosticos']['vih']['siglas dataset']
#     diabetes=config['variables']['situacion_clinica']['diagnosticos']['diabetes']['siglas dataset']
    
#     vih_identificador = config['parametros']['vih']['identificador']

#     # Inicialización de la columna 'diabetes_con_vih' en el DataFrame
#     df['reglas diabetes_con_vih'] = 0

#     # Verificación de la existencia de la variable VIH en el DataFrame
#     missing_vars = set()
#     available_vars = []

#     required_vars = [vih, diabetes]
    
#     for var in required_vars:
#         if var not in df.columns:
#             missing_vars.add(var)
#         else:
#             available_vars.append(var)

#     # Si la variable está ausente, se escribe en un archivo de texto
#     if missing_vars:
#         with open(path_variables_ausentes, 'w') as f:
#             for var in missing_vars:
#                 f.write(f"No se encuentra la variable '{var}' en el dataset.\n")

#     # Procesamiento de las filas para marcar 'diabetes_con_vih'
#     for k in range(len(df)):
#         try:
#             # Solo considerar los casos donde el paciente tiene diabetes
#             if df[diabetes][k] == 1 and df[vih][k] == vih_identificador:
#                 df['reglas diabetes_con_vih'][k] = 1
#         except:
#             pass

#     # Crear el nuevo DataFrame con las variables identificadas y 'diabetes_con_vih'
#     available_vars.append('reglas diabetes_con_vih')
#     df_vars_identificadas = df[available_vars]

#     return df, df_vars_identificadas


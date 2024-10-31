#################################################################################################################################################################
#### Reglas extraídas del documento "10. Cardiovascular Disease and Risk Management: Standards of Care in Diabetes — 2024" (4. Section10-diabetes con ECV) #####
################################################################################################################################################################

import pandas as pd

def añadir_reglas(df:pd.DataFrame,config):
    df,_=tension_alta(df,config, path_variables_ausentes='variables_ausentes.txt')
    df,_=hipertension(df,config, path_variables_ausentes='variables_ausentes.txt')
    # df,_=diabetes_con_hipertension(df,config, path_variables_ausentes='variables_ausentes.txt')
    # df,_=tratamiento_hipertension(df,config, path_variables_ausentes='variables_ausentes.txt')
    df,_=presion_arterial_ligeramente_elevada_hipertension(df,config, path_variables_ausentes='variables_ausentes.txt')
    # df,_=medicacion_hipertension(df,config, path_variables_ausentes='variables_ausentes.txt')
    df,_=hipertension_resistente(df,config, path_variables_ausentes='variables_ausentes.txt')
    df,_=medicacion_hipertension_con_enfermedad_renal(df,config, path_variables_ausentes='variables_ausentes.txt')
    # df,_=dislipemia(df,config, path_variables_ausentes='variables_ausentes.txt')
    # df,_=dislipemia_con_diabetes(df,config, path_variables_ausentes='variables_ausentes.txt')
    # df,_=medicacion_estatinas_prevencion_primaria(df,config, path_variables_ausentes='variables_ausentes.txt')
    # df,_=medicacion_estatinas_prevencion_secundaria(df,config, path_variables_ausentes='variables_ausentes.txt')
    df,_=hipertriglicidemia(df,config, path_variables_ausentes='variables_ausentes.txt')
    df,_=baja_lipoproteina_hdl(df,config, path_variables_ausentes='variables_ausentes.txt')
    df,_=hipercolesterolemia(df,config, path_variables_ausentes='variables_ausentes.txt')

    return df


# página 2
# TA a lo largo del texto = Tensión Alta?

def tension_alta(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
    tas = config['variables']['mediciones_generales']['tas']['siglas dataset']
    tas_umbral_inferior = config['parametros']['tension_alta']['tas_umbral_inferior']
    tas_umbral_superior = config['parametros']['tension_alta']['tas_umbral_superior']

    df[f'reglas tension alta ({tas_umbral_inferior}<{tas}<{tas_umbral_superior})'] = 0

    missing_vars = set()
    available_vars = []

    required_vars = [tas]
    
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
            if tas in df.columns and df[tas][k] > tas_umbral_inferior and df[tas][k] < tas_umbral_superior:
                df[f'reglas tension alta ({tas_umbral_inferior}<{tas}<{tas_umbral_superior})'][k] = 1
        except KeyError:
            pass

    available_vars.append(f'reglas tension alta ({tas_umbral_inferior}<{tas}<{tas_umbral_superior})')
    df_vars_identificadas = df[available_vars]

    return df, df_vars_identificadas


def hipertension(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
    tas = config['variables']['mediciones_generales']['tas']['siglas dataset']
    tad = config['variables']['mediciones_generales']['tad']['siglas dataset']
    tas_umbral_inferior = config['parametros']['hipertension']['tas_umbral_inferior']
    tad_umbral_inferior = config['parametros']['hipertension']['tad_umbral_inferior']

    df[f'reglas hipertension ({tas}>{tas_umbral_inferior} o {tad}>{tad_umbral_inferior})'] = 0

    missing_vars = set()
    available_vars = []

    required_vars = [tas,tad]
    
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
            if tas in df.columns and df[tas][k] > tas_umbral_inferior:
                df[f'reglas hipertension ({tas}>{tas_umbral_inferior} o {tad}>{tad_umbral_inferior})'][k] = 1
        except KeyError:
            pass

        try:
            if tad in df.columns and df[tad][k] > tad_umbral_inferior:
                df[f'reglas hipertension ({tas}>{tas_umbral_inferior} o {tad}>{tad_umbral_inferior})'][k] = 1
        except KeyError:
            pass
    # Estas condiciones de arriba son para el promedio de dos o más medidas. Si en una sola medida se observa
    # de un paciente con problemas CV y un tas/tad>180/110, es suficiente para diagnosticar hipertensión.

    available_vars.append(f'reglas hipertension ({tas}>{tas_umbral_inferior} o {tad}>{tad_umbral_inferior})')
    df_vars_identificadas = df[available_vars]

    return df, df_vars_identificadas


# página 2
# def diabetes_con_hipertension(df:pd.DataFrame,config,path_variables_ausentes='variables_ausentes.txt'):
#     diabetes = config['variables']['situacion_clinica']['diagnosticos']['diabetes']['siglas dataset']
#     hipertension = config['variables']['situacion_clinica']['diagnosticos']['hipertension']['siglas dataset']
    
#     df['reglas diabetes con hipertension']=0
    
#     missing_vars = set()
#     available_vars = []

#     required_vars = [diabetes,hipertension]
    
#     for var in required_vars:
#         if var not in df.columns:
#             missing_vars.add(var)
#         else:
#             available_vars.append(var)

#     if missing_vars:
#         with open(path_variables_ausentes, 'w') as f:
#             for var in missing_vars:
#                 f.write(f"No se encuentra la variable '{var}' en el dataset.\n")

#     for k in range(len(df)):
#         try:
#             if df['reglas_'+hipertension][k] == 1 and df['reglas_'+diabetes][k] == 1:
#                 df['reglas diabetes con hipertension'][k] = 1
#         except:
#             print("Problema en la función diabetes_con_hipertension()")

#     available_vars.append('reglas diabetes con hipertension')
#     df_vars_identificadas = df[available_vars]

#     return df, df_vars_identificadas

# página 4
# def tratamiento_hipertension(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
#     tas = config['variables']['mediciones_generales']['tas']['siglas dataset']
#     tad = config['variables']['mediciones_generales']['tad']['siglas dataset']
#     tas_umbral_inferior = config['parametros']['tratamiento_hipertension']['tas_umbral_inferior']
#     tad_umbral_inferior = config['parametros']['tratamiento_hipertension']['tad_umbral_inferior']

#     df['reglas tratamiento hipertension'] = 0

#     missing_vars = set()
#     available_vars = []

#     required_vars = [tas,tad]
    
#     for var in required_vars:
#         if var not in df.columns:
#             missing_vars.add(var)
#         else:
#             available_vars.append(var)

#     if missing_vars:
#         with open(path_variables_ausentes, 'w') as f:
#             for var in missing_vars:
#                 f.write(f"No se encuentra la variable '{var}' en el dataset.\n")

#     for k in range(len(df)):
#         try:
#             if tas in df.columns and df[tas][k] > tas_umbral_inferior:
#                 df['reglas tratamiento hipertension'][k] = 1
#         except KeyError:
#             pass

#         try:
#             if tad in df.columns and df[tad][k] > tad_umbral_inferior:
#                 df['reglas tratamiento hipertension'][k] = 1
#         except KeyError:
#             pass

#     available_vars.append('reglas tratamiento hipertension')
#     df_vars_identificadas = df[available_vars]

#     return df, df_vars_identificadas

# página 6. Primer subrayado para texto libre. Segundo subrayado también. Tercero igual. 

# página 6
def presion_arterial_ligeramente_elevada_hipertension(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
    tas = config['variables']['mediciones_generales']['tas']['siglas dataset']
    tad = config['variables']['mediciones_generales']['tad']['siglas dataset']
    tas_umbral_inferior = config['parametros']['presion_arterial_ligeramente_elevada']['tas_umbral_inferior']
    tad_umbral_inferior = config['parametros']['presion_arterial_ligeramente_elevada']['tad_umbral_inferior']

    df[f'reglas presion arterial ligeramente elevada ({tas}>{tas_umbral_inferior} o {tad}>{tad_umbral_inferior})'] = 0

    missing_vars = set()
    available_vars = []

    required_vars = [tas,tad]
    
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
            if tas in df.columns and tad in df.columns:
                if df[tas][k] > tas_umbral_inferior or df[tad][k] > tad_umbral_inferior:
                    df[f'reglas presion arterial ligeramente elevada ({tas}>{tas_umbral_inferior} o {tad}>{tad_umbral_inferior})'][k] = 1
        except KeyError:
            print("No se encuentra la variable 'tas' o 'tad' en el dataset.")
        except:
            print("Problema en la función presion_arterial_ligeramente_elevada_hipertension()")

    available_vars.append(f'reglas presion arterial ligeramente elevada ({tas}>{tas_umbral_inferior} o {tad}>{tad_umbral_inferior})')
    df_vars_identificadas = df[available_vars]

    return df, df_vars_identificadas


# páginas 7 y 8. (Incluye los dos primeros párrafos del rectángulo de la parte inferior derecha de la página 6)
# def medicacion_hipertension(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
#     tas = config['variables']['mediciones_generales']['tas']['siglas dataset']
#     tad = config['variables']['mediciones_generales']['tad']['siglas dataset']
#     albumina_creatinina = config['variables']['analisis_orina']['albumina_creatinina']['siglas dataset']
#     cad = config['variables']['situacion_clinica']['diagnosticos']['cad']['siglas dataset']
#     enfermedad_renal = config['variables']['situacion_clinica']['diagnosticos']['enfermedad_renal']['siglas dataset']

#     tas_medicacion_hipertension_start_one_umbral_inferior = config['parametros']['recomendacion_medicacion_hipertension_start_one_drug']['tas_umbral_inferior']
#     tas_medicacion_hipertension_start_one_umbral_superior = config['parametros']['recomendacion_medicacion_hipertension_start_one_drug']['tas_umbral_superior']
#     tad_medicacion_hipertension_start_one_umbral_inferior = config['parametros']['recomendacion_medicacion_hipertension_start_one_drug']['tad_umbral_inferior']
#     tad_medicacion_hipertension_start_one_umbral_superior = config['parametros']['recomendacion_medicacion_hipertension_start_one_drug']['tad_umbral_superior']
    
#     tas_medicacion_hipertension_ACEI_or_ARB_umbral_inferior = config['parametros']['recomendacion_medicacion_hipertension_ACEI_or_ARB']['tas_umbral_inferior']
#     tas_medicacion_hipertension_ACEI_or_ARB_umbral_superior = config['parametros']['recomendacion_medicacion_hipertension_ACEI_or_ARB']['tas_umbral_superior']
#     tad_medicacion_hipertension_ACEI_or_ARB_umbral_inferior = config['parametros']['recomendacion_medicacion_hipertension_ACEI_or_ARB']['tad_umbral_inferior']
#     tad_medicacion_hipertension_ACEI_or_ARB_umbral_superior = config['parametros']['recomendacion_medicacion_hipertension_ACEI_or_ARB']['tad_umbral_superior']
    
#     tas_medicacion_hipertension_start_from_2_umbral_inferior = config['parametros']['recomendacion_medicacion_hipertension_start_from_2']['tas_umbral_inferior']
#     tad_medicacion_hipertension_start_from_2_umbral_inferior = config['parametros']['recomendacion_medicacion_hipertension_start_from_2']['tad_umbral_inferior']
    
#     tas_medicacion_hipertension_ACEI_or_ARB_and_CCB_or_Diuretic_umbral_inferior = config['parametros']['recomendacion_medicacion_hipertension_ACEI_or_ARB_and_CCB_or_Diuretic']['tas_umbral_inferior']
#     tas_medicacion_hipertension_ACEI_or_ARB_and_CCB_or_Diuretic_umbral_superior = config['parametros']['recomendacion_medicacion_hipertension_ACEI_or_ARB_and_CCB_or_Diuretic']['tas_umbral_superior']
#     tad_medicacion_hipertension_ACEI_or_ARB_and_CCB_or_Diuretic_umbral_inferior = config['parametros']['recomendacion_medicacion_hipertension_ACEI_or_ARB_and_CCB_or_Diuretic']['tad_umbral_inferior']
#     tad_medicacion_hipertension_ACEI_or_ARB_and_CCB_or_Diuretic_umbral_superior = config['parametros']['recomendacion_medicacion_hipertension_ACEI_or_ARB_and_CCB_or_Diuretic']['tad_umbral_superior']
    
#     df['recomendacion_medicacion_hipertension_start_one_drug'] = 0
#     df['recomendacion_medicacion_hipertension_ACEI_or_ARB'] = 0
#     df['recomendacion_medicacion_hipertension_start_from_2'] = 0
#     df['recomendacion_medicacion_hipertension_ACEI_or_ARB_and_CCB_or_Diuretic'] = 0
#     df['recomendacion_medicacion_hipertension_con_enfermedad_renal'] = 0

#     missing_vars = set()
#     available_vars = []

#     required_vars = [tas, tad, albumina_creatinina, cad, enfermedad_renal]
    
#     for var in required_vars:
#         if var not in df.columns:
#             missing_vars.add(var)
#         else:
#             available_vars.append(var)

#     if missing_vars:
#         with open(path_variables_ausentes, 'w') as f:
#             for var in missing_vars:
#                 f.write(f"No se encuentra la variable '{var}' en el dataset.\n")

#     for k in range(len(df)):
#         try:
#             if (df[tas][k] > tas_medicacion_hipertension_start_one_umbral_inferior and 
#                 df[tas][k] < tas_medicacion_hipertension_start_one_umbral_superior and 
#                 df[tad][k] > tad_medicacion_hipertension_start_one_umbral_inferior and 
#                 df[tad][k] < tad_medicacion_hipertension_start_one_umbral_superior and 
#                 (df[albumina_creatinina][k] < 300 or df[cad][k] == 0)):
#                 df['recomendacion_medicacion_hipertension_start_one_drug'][k] = 1
#             if (df[tas][k] > tas_medicacion_hipertension_ACEI_or_ARB_umbral_inferior and 
#                 df[tas][k] < tas_medicacion_hipertension_ACEI_or_ARB_umbral_superior and 
#                 df[tad][k] > tad_medicacion_hipertension_ACEI_or_ARB_umbral_inferior and 
#                 df[tad][k] < tad_medicacion_hipertension_ACEI_or_ARB_umbral_superior and 
#                 (df[albumina_creatinina][k] >= 300 or df[cad][k] == 1)):
#                 df['recomendacion_medicacion_hipertension_start_ACEI_or_ARB'][k] = 1
#             if (df[tas][k] > tas_medicacion_hipertension_start_from_2_umbral_inferior and 
#                 df[tad][k] > tad_medicacion_hipertension_start_from_2_umbral_inferior and 
#                 (df[albumina_creatinina][k] <300 or df[cad][k] == 0)):
#                 df['recomendacion_medicacion_hipertension_start_from_2_drugs'][k] = 1
#             if (df[tas][k] > tas_medicacion_hipertension_ACEI_or_ARB_and_CCB_or_Diuretic_umbral_inferior and 
#                 df[tas][k] < tas_medicacion_hipertension_ACEI_or_ARB_and_CCB_or_Diuretic_umbral_superior and 
#                 df[tad][k] > tad_medicacion_hipertension_ACEI_or_ARB_and_CCB_or_Diuretic_umbral_inferior and 
#                 df[tad][k] < tad_medicacion_hipertension_ACEI_or_ARB_and_CCB_or_Diuretic_umbral_superior and 
#                 (df[albumina_creatinina][k] >= 300 or df[cad][k] == 1)):
#                 df['recomendacion_medicacion_hipertension_start_ACEI_or_ARB_and_CCB_or_Diuretic'][k] = 1
#         except KeyError as e:
#             print(f"No se encuentra la variable {e} en el dataset.")
#         except Exception as e:
#             print(f"Error procesando la fila {k}: {e}")

#     available_vars.append('recomendacion_medicacion_hipertension_start_one')
#     available_vars.append('recomendacion_medicacion_hipertension_ACEI_or_ARB')
#     available_vars.append('recomendacion_medicacion_hipertension_start_from_2')
#     available_vars.append('recomendacion_medicacion_hipertension_ACEI_or_ARB_and_CCB_or_Diuretic')
#     available_vars.append('recomendacion_medicacion_hipertension_con_enfermedad_renal')
#     df_vars_identificadas = df[available_vars]

#     return df, df_vars_identificadas


#página 8
def hipertension_resistente(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
    tas = config['variables']['mediciones_generales']['tas']['siglas dataset']
    tad = config['variables']['mediciones_generales']['tad']['siglas dataset']
    tas_umbral_inferior = config['parametros']['hipertension_resistente']['tas_umbral_inferior']
    tad_umbral_inferior = config['parametros']['hipertension_resistente']['tad_umbral_inferior']

    df[f'reglas hipertension resistente ({tas}>{tas_umbral_inferior} o {tad}>{tad_umbral_inferior})'] = 0

    missing_vars = set()
    available_vars = []

    required_vars = [tas, tad]
    
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
            if df[tas][k]>tas_umbral_inferior or df[tad][k]>tad_umbral_inferior:
                df[f'reglas hipertension resistente ({tas}>{tas_umbral_inferior} o {tad}>{tad_umbral_inferior})'][k]=1
        except:
            pass
        # try:
        #     Si en texto libre nos dice que con una sola visita que tas/tad>180/110 + CV entonces hipertensión. En otros casos m´sa visitas.
        # except:
        #     pass
    available_vars.append(f'reglas hipertension resistente ({tas}>{tas_umbral_inferior} o {tad}>{tad_umbral_inferior})')
    df_vars_identificadas = df[available_vars]

    return df, df_vars_identificadas


# página 8
# ¿QUÉ REGLA CONSTRUYO CON ESTO? MED CATEGORIA 3: SI TIENE ENFERMEDAD RENAL, PUEDE QUE NO LLEVE LA COMBINACION ANTEIOR DE MEDICACION PARA HTA
def medicacion_hipertension_con_enfermedad_renal(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
    tas = config['variables']['mediciones_generales']['tas']['siglas dataset']
    tad = config['variables']['mediciones_generales']['tad']['siglas dataset']
    diuretic = config['variables']['situacion_clinica']['medicacion']['diuretic']['siglas dataset']
    two_antihypertensive = config['variables']['situacion_clinica']['medicacion']['two_antihypertensive']['siglas dataset']
    tas_umbral_inferior = config['parametros']['hipertension']['tas_umbral_inferior']
    tad_umbral_inferior = config['parametros']['hipertension']['tad_umbral_inferior']

    df['reglas recomendacion medicacion hipertension con enfermedad renal'] = 0

    missing_vars = set()
    available_vars = []

    required_vars = [tas, tad, diuretic, two_antihypertensive]
    
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
            if df[tas][k] > tas_umbral_inferior and df[tad][k] > tad_umbral_inferior and df[diuretic][k] == 1 and df[two_antihypertensive][k] == 1:
                df['reglas recomendacion medicacion hipertension con enfermedad renal'][k] = 1
        except KeyError as e:
            print(f"No se encuentra la variable {e} en el dataset.")

    available_vars.append('reglas recomendacion medicacion hipertension con enfermedad renal')
    df_vars_identificadas = df[available_vars]

    return df, df_vars_identificadas

#página 8
def hipertriglicidemia(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
    tg=config['variables']['analisis_sangre']['tg']['siglas dataset']
    tg_umbral=config['parametros']['hipertriglicidemia']['tg_umbral']

    df[f'reglas hipertriglicidemia ({tg}>{tg_umbral})'] = 0

    missing_vars = set()
    available_vars = []

    required_vars = [tg]
    
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
            if df[tg][k] >= tg_umbral:
                df[f'reglas hipertriglicidemia ({tg}>{tg_umbral})'][k] = 1
        except KeyError as e:
            print(f"No se encuentra la variable {e} en el dataset.")

    available_vars.append(f'reglas hipertriglicidemia ({tg}>{tg_umbral})')
    df_vars_identificadas = df[available_vars]

    return df, df_vars_identificadas

#página 8
def baja_lipoproteina_hdl(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
    hdl=config['variables']['analisis_sangre']['hdl']['siglas dataset']
    genero=config['variables']['informacion_basica']['genero']['siglas dataset']
    hdl_umbral_hombres=config['parametros']['baja_lipoproteina_hdl']['hdl_umbral_hombres']
    hdl_umbral_mujeres=config['parametros']['baja_lipoproteina_hdl']['hdl_umbral_mujeres']

    df[f'reglas baja lipoproteina hdl ({hdl}<{hdl_umbral_hombres} para hombres y {hdl}<{hdl_umbral_mujeres} para mujeres)'] = 0

    missing_vars = set()
    available_vars = []

    required_vars = [hdl]
    
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
            if (df[hdl][k] < hdl_umbral_hombres and df[genero][k]==2) or (df[hdl][k] < hdl_umbral_mujeres and df[genero][k]==1):
                df[f'reglas baja lipoproteina hdl ({hdl}<{hdl_umbral_hombres} para hombres y {hdl}<{hdl_umbral_mujeres} para mujeres)'][k] = 1
        except KeyError as e:
            print(f"No se encuentra la variable {e} en el dataset.")

    available_vars.append(f'reglas baja lipoproteina hdl ({hdl}<{hdl_umbral_hombres} para hombres y {hdl}<{hdl_umbral_mujeres} para mujeres)')
    df_vars_identificadas = df[available_vars]

    return df, df_vars_identificadas

#página 9
def hipercolesterolemia(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
    ldl=config['variables']['analisis_sangre']['ldl']['siglas dataset']
    ldl_umbral=config['parametros']['hipercolesterolemia']['ldl_umbral']

    df[f'reglas hipercolesterolemia ({ldl}>{ldl_umbral})'] = 0

    missing_vars = set()
    available_vars = []

    required_vars = [ldl]
    
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
            if df[ldl][k] > ldl_umbral:
                df[f'reglas hipercolesterolemia ({ldl}>{ldl_umbral})'][k] = 1
        except KeyError as e:
            print(f"No se encuentra la variable {e} en el dataset.")

    available_vars.append(f'reglas hipercolesterolemia ({ldl}>{ldl_umbral})')
    df_vars_identificadas = df[available_vars]

    return df, df_vars_identificadas











# página 8
# def dislipemia(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
#     tg = config['variables']['analisis_sangre']['tg']['siglas dataset']
#     hdl = config['variables']['analisis_sangre']['hdl']['siglas dataset']
#     genero = config['variables']['informacion_basica']['genero']['siglas dataset']  # 0 hombre, 1 mujer
#     tg_umbral_inferior = config['parametros']['dislipemia_con_diabetes']['tg_umbral_inferior']
#     hdl_umbral_superior_hombre = config['parametros']['dislipemia_con_diabetes']['hdl_umbral_superior_hombre']
#     hdl_umbral_superior_mujer = config['parametros']['dislipemia_con_diabetes']['hdl_umbral_superior_mujer']

#     df['dislipemia'] = 0

#     missing_vars = set()
#     available_vars = []

#     required_vars = [tg, hdl, genero]

#     for var in required_vars:
#         if var not in df.columns:
#             missing_vars.add(var)
#         else:
#             available_vars.append(var)

#     if missing_vars:
#         with open(path_variables_ausentes, 'w') as f:
#             for var in missing_vars:
#                 f.write(f"No se encuentra la variable '{var}' en el dataset.\n")

#     for k in range(len(df)):
#         try:
#             if df[tg][k] > tg_umbral_inferior:
#                 df['dislipemia'][k] = 1
#             if df[hdl][k] < hdl_umbral_superior_hombre and df[genero][k] == 2:
#                 df['dislipemia'][k] = 1
#             if [hdl][k] < hdl_umbral_superior_mujer and df[genero][k] == 1:
#                 df['dislipemia'][k] = 1
                    

#             # if (df[tg][k] > tg_umbral_superior or 
#             #     (df[hdl][k] < hdl_umbral_inferior_hombre and df[genero][k] == "2") or #2=Hombre
#             #     (df[hdl][k] < hdl_umbral_inferior_mujer and df[genero][k] == "1")): #1=Mujer
#             #     print("Datos: ",df[hdl][k],df[genero][k],(df[hdl][k] < hdl_umbral_inferior_hombre and df[genero][k] == "2"))
#             #     df['dislipemia'][k] = 1
#         except KeyError as e:
#             pass
#         except Exception as e:
#             print(f"Error procesando la fila {k}: {e}")

#     available_vars.append('dislipemia')
#     df_vars_identificadas = df[available_vars]

#     return df, df_vars_identificadas

# def dislipemia_con_diabetes(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
#     diabetes = config['variables']['situacion_clinica']['diagnosticos']['diabetes']['siglas dataset']
#     dislipemia = config['variables']['situacion_clinica']['diagnosticos']['dislipemia']['siglas dataset']

#     df['reglas dislipemia con diabetes'] = 0

#     missing_vars = set()
#     available_vars = []

#     required_vars = [diabetes, dislipemia]

#     for var in required_vars:
#         if var not in df.columns:
#             missing_vars.add(var)
#         else:
#             available_vars.append(var)

#     if missing_vars:
#         with open(path_variables_ausentes, 'w') as f:
#             for var in missing_vars:
#                 f.write(f"No se encuentra la variable '{var}' en el dataset.\n")
    
#     for k in range(len(df)):
#         try:
#             if df['reglas_'+diabetes][k]==1 and df['reglas_'+dislipemia][k]==1:
#                 df['reglas dislipemia con diabetes'][k] = 1
#         except:
#             pass
    
#     available_vars.append('reglas dislipemia con diabetes')
#     df_vars_identificadas = df[available_vars]

#     return df, df_vars_identificadas

#página 9
# def medicacion_estatinas_prevencion_primaria(df:pd.DataFrame,config,path_variables_ausentes='variables_ausentes.txt'):
#     edad= config['variables']['informacion_basica']['edad']['siglas dataset']
#     diabetes = config['variables']['situacion_clinica']['diagnosticos']['diabetes']['siglas dataset']
#     factor_riesgo_cv= config['variables']['situacion_clinica']['diagnosticos']['factor_riesgo_cv']['siglas dataset']
#     enfermedad_cv= config['variables']['situacion_clinica']['diagnosticos']['enfermedad_cv']['siglas dataset']
#     terapia_con_estatinas= config['variables']['situacion_clinica']['diagnosticos']['terapia_con_estatinas']['siglas dataset']
#     ldl= config['variables']['analisis_sangre']['ldl']['siglas dataset']
#     intolerante_estatinas= config['variables']['situacion_clinica']['diagnosticos']['intolerante_estatinas']['siglas dataset']

#     df["recomendacion_medicacion_usar terapia con estatinas de intensidad moderada además de la terapia de estilo de vida"]=0
#     df["recomendacion_medicacion_iniciar terapia con estatinas además de la terapia de estilo de vida"]=0
#     df["recomendacion_medicacion_usar terapia con estatinas de alta intensidad para reducir el colesterol LDL en un 50 porciento respecto al valor inicial y alcanzar una meta de colesterol LDL < 70 mg/dL"]=0
#     df["recomendacion_medicacion_agregar ezetimiba o un inhibidor de PCSK9 a la terapia máxima tolerada con estatinas"]=0
#     df["recomendacion_medicacion_continuar con el tratamiento de estatinas"]=0
#     df["recomendacion_medicacion_posible iniciación moderada-intensa de tratamiento de estatinas tras valorar el potencial beneficio y riesgo"]=0
#     df["recomendacion_medicacion_acido bempedoico"]

#     missing_vars = set()
#     available_vars = []

#     required_vars = [edad, diabetes, factor_riesgo_cv, enfermedad_cv, terapia_con_estatinas, ldl, intolerante_estatinas]

#     for var in required_vars:
#         if var not in df.columns:
#             missing_vars.add(var)
#         else:
#             available_vars.append(var)

#     if missing_vars:
#         with open(path_variables_ausentes, 'w') as f:
#             for var in missing_vars:
#                 f.write(f"No se encuentra la variable '{var}' en el dataset.\n")
    
#     for k in range(len(df)):
#         try:
#             if df[diabetes][k]==1 and df[edad][k]>=40 and df[edad][k]<=75 and df[enfermedad_cv][k]==0:
#                 df["recomendacion_medicacion_usar terapia con estatinas de intensidad moderada además de la terapia de estilo de vida"][k]=1
#             if df[diabetes][k]==1 and df[edad][k]>=20 and df[edad][k]<=39 and df[factor_riesgo_cv][k]==1:
#                 df["recomendacion_medicacion_iniciar terapia con estatinas además de la terapia de estilo de vida"][k]=1
#             if df[diabetes][k]==1 and df[edad][k]>=40 and df[edad][k]<=75 and df[factor_riesgo_cv][k]==1:
#                 df["recomendacion_medicacion_usar terapia con estatinas de alta intensidad para reducir el colesterol LDL en un 50 porciento respecto al valor inicial y alcanzar una meta de colesterol LDL < 70 mg/dL"][k]=1
#             if df[diabetes][k]==1 and df[edad][k]>=40 and df[edad][k]<=75 and df[factor_riesgo_cv][k]==1 and df[ldl][k]>=70:
#                 df["recomendacion_medicacion_agregar ezetimiba o un inhibidor de PCSK9 a la terapia máxima tolerada con estatinas"][k]=1
#             if df[diabetes][k]==1 and df[edad][k]>75 and df[terapia_con_estatinas][k]==1:
#                 df["recomendacion_medicacion_continuar con el tratamiento de estatinas"]=0
#             if df[diabetes][k]==1 and df[edad][k]>75 and df[terapia_con_estatinas][k]==0:
#                 df["recomendacion_medicacion_posible iniciación moderada-intensa de tratamiento de estatinas tras valorar el potencial beneficio y riesgo"]=0
#             if df[diabetes][k]==1 and df[intolerante_estatinas][k]==1:
#                 df["recomendacion_medicacion_acido bempedoico"]
#         except:
#             pass

#     available_vars.append('recomendacion_medicacion_usar terapia con estatinas de intensidad moderada además de la terapia de estilo de vida')
#     available_vars.append('recomendacion_medicacion_iniciar terapia con estatinas además de la terapia de estilo de vidaa')
#     available_vars.append('recomendacion_medicacion_usar terapia con estatinas de alta intensidad para reducir el colesterol LDL en un 50 porciento respecto al valor inicial y alcanzar una meta de colesterol LDL < 70 mg/dL')
#     available_vars.append('recomendacion_medicacion_agregar ezetimiba o un inhibidor de PCSK9 a la terapia máxima tolerada con estatinas')
#     available_vars.append('recomendacion_medicacion_continuar con el tratamiento de estatinas')
#     available_vars.append('recomendacion_medicacion_posible iniciación moderada-intensa de tratamiento de estatinas tras valorar el potencial beneficio y riesgo')
#     available_vars.append('recomendacion_medicacion_acido bempedoico')
#     df_vars_identificadas = df[available_vars]

#     return df, df_vars_identificadas


# def medicacion_estatinas_prevencion_secundaria(df:pd.DataFrame,config,path_variables_ausentes='variables_ausentes.txt'):
#     diabetes = config['variables']['situacion_clinica']['diagnosticos']['diabetes']['siglas dataset']
#     enfermedad_cv= config['variables']['situacion_clinica']['diagnosticos']['enfermedad_cv']['siglas dataset']
#     intolerante_estatinas= config['variables']['situacion_clinica']['diagnosticos']['intolerante_estatinas']['siglas dataset']

#     df["recomendacion_medicacion_se recomienda la terapia para alcanzar una reducción del colesterol LDL del 50 porciento respecto al valor inicial y una meta de colesterol LDL < 55 mg/dL (< 1.4 mmol/L). Si no se alcanza esta meta con la terapia máxima tolerada con estatinas, se recomienda agregar ezetimiba o un inhibidor de PCSK9 con beneficio comprobado en esta población"]=0
#     df["recomendacion_medicacion_terapia con inhibidor de PCSK9 con tratamiento de anticuerpo monoclonal, terapia con ácido bempedoico, o terapia con inhibidor de PCSK9 con inclisiran siRNA"]=0

#     missing_vars = set()
#     available_vars = []

#     required_vars = [diabetes, enfermedad_cv, intolerante_estatinas]

#     for var in required_vars:
#         if var not in df.columns:
#             missing_vars.add(var)
#         else:
#             available_vars.append(var)

#     if missing_vars:
#         with open(path_variables_ausentes, 'w') as f:
#             for var in missing_vars:
#                 f.write(f"No se encuentra la variable '{var}' en el dataset.\n")
    
#     for k in range(len(df)):
#         try:
#             if df[diabetes][k]==1 and df[enfermedad_cv][k]==1:
#                 df["recomendacion_medicacion_se recomienda la terapia para alcanzar una reducción del colesterol LDL del 50 porciento respecto al valor inicial y una meta de colesterol LDL < 55 mg/dL (< 1.4 mmol/L). Si no se alcanza esta meta con la terapia máxima tolerada con estatinas, se recomienda agregar ezetimiba o un inhibidor de PCSK9 con beneficio comprobado en esta población"][k]=1
#             if df[diabetes][k]==1 and df[enfermedad_cv][k] and df[enfermedad_cv][k]==1:
#                 df["recomendacion_medicacion_terapia con inhibidor de PCSK9 con tratamiento de anticuerpo monoclonal, terapia con ácido bempedoico, o terapia con inhibidor de PCSK9 con inclisiran siRNA"][k]=1
#         except:
#             pass

#     available_vars.append('recomendacion_medicacion_se recomienda la terapia para alcanzar una reducción del colesterol LDL del 50 porciento respecto al valor inicial y una meta de colesterol LDL < 55 mg/dL (< 1.4 mmol/L). Si no se alcanza esta meta con la terapia máxima tolerada con estatinas, se recomienda agregar ezetimiba o un inhibidor de PCSK9 con beneficio comprobado en esta población')
#     available_vars.append('recomendacion_medicacion_terapia con inhibidor de PCSK9 con tratamiento de anticuerpo monoclonal, terapia con ácido bempedoico, o terapia con inhibidor de PCSK9 con inclisiran siRNA')
#     df_vars_identificadas = df[available_vars]

#     return df, df_vars_identificadas




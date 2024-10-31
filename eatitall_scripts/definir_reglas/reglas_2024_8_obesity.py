#################################################################################################################################################################
#### Reglas extraídas del documento "8. Obesity and Weight Management for the Prevention and Treatment of Type 2 Diabetes: Standards of Care in Diabetes –2024" (5. Section8-OBESIDAD DM) #####
################################################################################################################################################################

import pandas as pd

def añadir_reglas(df:pd.DataFrame,config):
    df,_=obseidad_clase_i(df,config, path_variables_ausentes='variables_ausentes.txt')
    df,_=obseidad_clase_ii(df,config, path_variables_ausentes='variables_ausentes.txt')
    df,_=obseidad_clase_iii(df,config, path_variables_ausentes='variables_ausentes.txt')

    # df,_=hipertension(df,config, path_variables_ausentes='variables_ausentes.txt')

    return df


# página 2
def obseidad_clase_i(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
    imc = config['variables']['mediciones_generales']['imc']['siglas dataset']
    imc_umbral_inferior = config['parametros']['obesidad_clase_i']['obesidad_clase_i_umbral_inferior']
    imc_umbral_superior = config['parametros']['obesidad_clase_i']['obesidad_clase_i_umbral_superior']

    df[f'obesidad_clase_i ({imc_umbral_inferior}<{imc}<{imc_umbral_superior})'] = 0

    missing_vars = set()
    available_vars = []

    required_vars = [imc]
    
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
            if imc in df.columns and df[imc][k] >= imc_umbral_inferior and df[imc][k] < imc_umbral_superior:
                df[f'obesidad_clase_i ({imc_umbral_inferior}<{imc}<{imc_umbral_superior})'][k] = 1
        except KeyError:
            pass

    available_vars.append(f'obesidad_clase_i ({imc_umbral_inferior}<{imc}<{imc_umbral_superior})')
    df_vars_identificadas = df[available_vars]
  
    return df, df_vars_identificadas

def obseidad_clase_ii(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
    imc = config['variables']['mediciones_generales']['imc']['siglas dataset']
    imc_umbral_inferior = config['parametros']['obesidad_clase_ii']['obesidad_clase_ii_umbral_inferior']
    imc_umbral_superior = config['parametros']['obesidad_clase_ii']['obesidad_clase_ii_umbral_superior']

    df[f'obesidad_clase_ii ({imc_umbral_inferior}<{imc}<{imc_umbral_superior})'] = 0

    missing_vars = set()
    available_vars = []

    required_vars = [imc]
    
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
            if imc in df.columns and df[imc][k] >= imc_umbral_inferior and df[imc][k] < imc_umbral_superior:
                df[f'obesidad_clase_ii ({imc_umbral_inferior}<{imc}<{imc_umbral_superior})'][k] = 1
        except KeyError:
            pass

    available_vars.append(f'obesidad_clase_ii ({imc_umbral_inferior}<{imc}<{imc_umbral_superior})')
    df_vars_identificadas = df[available_vars]
    
    return df, df_vars_identificadas

def obseidad_clase_iii(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
    imc = config['variables']['mediciones_generales']['imc']['siglas dataset']
    imc_umbral = config['parametros']['obesidad_clase_iii']['obesidad_clase_iii_umbral']

    df[f'obesidad_clase_ii ({imc}>{imc_umbral})'] = 0

    missing_vars = set()
    available_vars = []

    required_vars = [imc]
    
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
            if imc in df.columns and df[imc][k] >= imc_umbral:
                df[f'obesidad_clase_ii ({imc}>{imc_umbral})'][k] = 1
        except KeyError:
            pass

    available_vars.append(f'obesidad_clase_ii ({imc}>{imc_umbral})')
    df_vars_identificadas = df[available_vars]

    return df, df_vars_identificadas

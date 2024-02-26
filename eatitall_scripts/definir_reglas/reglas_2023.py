# Añadimos la regla para detectar la PREDIABETES (RIESGO DE TENER DIÁBETES)
# Regla extraída del documento "Estándares de atención en DIABETES GUÍA 2023" para atención primaria en las páginas 4 y 5

import pandas as pd

def añadir_reglas(df:pd.DataFrame,config):
    df=prediabetes(df,config)
    df=diabetes(df)
    df=diabetes_con_cv(df,config)
    df=diabetes_con_hipertension(df,config)
    df=diabetes_con_lipidos(df,config)
    df=diabetes_con_estatinas(df,config)
    df=diabetes_mayores_de_65(df,config)
    df=diabetes_mayores_de_65_y_salud_saludable(df,config)
    df=diabetes_mayores_de_65_y_salud_compleja(df,config)
    df=diabetes_mayores_de_65_y_salud_muy_compleja(df,config)
    return df

# Regla extraída del documento "Estándares de atención en DIABETES GUÍA 2023" para atención primaria en las páginas 4 y 5
def prediabetes(df:pd.DataFrame,config) -> pd.DataFrame:
    imc=config['variables']['imc']['siglas dataset']
    tas=config['variables']['tas']['siglas dataset']
    tad=config['variables']['tad']['siglas dataset']
    tg=config['variables']['tg']['siglas dataset']
    hba1c=config['variables']['hba1c']['siglas dataset']
    fg=config['variables']['fg']['siglas dataset']
    imc_umbral_inferior_prediabetes=config['parametros']['prediabetes']['imc_umbral_inferior']
    tas_umbral_inferior_prediabetes=config['parametros']['prediabetes']['tas_umbral_inferior']
    tad_umbral_inferior_prediabetes=config['parametros']['prediabetes']['tad_umbral_inferior']
    tg_umbral_inferior_prediabetes=config['parametros']['prediabetes']['tg_umbral_inferior']
    hba1c_umbral_inferior_prediabetes=config['parametros']['prediabetes']['hba1c_umbral_inferior']
    hba1c_umbral_superior_prediabetes=config['parametros']['prediabetes']['hba1c_umbral_superior']
    fg_umbral_inferior_prediabetes=config['parametros']['prediabetes']['fg_umbral_inferior']
    fg_umbral_superior_prediabetes=config['parametros']['prediabetes']['fg_umbral_superior']
    df['prediabetes']=0 #Definimos una columna y la inicializamos con 0s. Los pacientes con prediabetes serán un 1.
    for k in range(0,len(df)):
        # 1- Personas con sobrepeso
        if df[imc][k]>=imc_umbral_inferior_prediabetes: # NOTA: No estamos teniendo en cuenta que para asiáticos-americanos el IMC>=23 porque no tenemos el dato del origen de cada persona
            if df[tas][k]>=tas_umbral_inferior_prediabetes:
                df['prediabetes'][k]=1
            if df[tad][k]>=tad_umbral_inferior_prediabetes:
                df['prediabetes'][k]=1
            if df[tg][k]>tg_umbral_inferior_prediabetes:
                df['prediabetes'][k]=1
        # 2- Personas con prediabetes (A1C)
        if df[hba1c][k]>=hba1c_umbral_inferior_prediabetes and df[hba1c][k]<=hba1c_umbral_superior_prediabetes:
            df['prediabetes'][k]=1
        # 3- FPG
        if df[fg][k]>=fg_umbral_inferior_prediabetes and df[fg][k]<=fg_umbral_superior_prediabetes:
            df['prediabetes'][k]=1
    return df

# Regla extraída del documento "Estándares de atención en DIABETES GUÍA 2023" para atención primaria en la página 5
def diabetes(df:pd.DataFrame) -> pd.DataFrame:
    # Si siguieramos la tabla de la página 4:
    # df['diabetes']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    # for k in range(0,len(df)):
    #     # Personas con prediabetes (A1C)
    #     if df['hba1c'][k]>=6.5:
    #         df['prediabetes'][k]=1
    #     # FPG
    #     if df['fg'][k]>=1266:
    #         df['prediabetes'][k]=1
    # Pero sabemos que todos son diabéticos, así que:
    df['diabetes']=1
    return df

# Regla extraída del documento "Estándares de atención en DIABETES GUÍA 2023" para atención primaria en la página 17
def diabetes_con_cv(df:pd.DataFrame,config) -> pd.DataFrame:
    acv=config['variables']['acv']['siglas dataset']
    acv_identificador=config['parametros']['cardiovascular']['identificador']
    df['diabetes_con_cv']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    for k in range(0,len(df)):
        if df['diabetes'][k]==1:
            if df[acv][k]==acv_identificador:
                df['diabetes_con_cv'][k]=1
    return df

# Regla extraída del documento "Estándares de atención en DIABETES GUÍA 2023" para atención primaria en la página 17
def diabetes_con_hipertension(df:pd.DataFrame,config) -> pd.DataFrame:
    tas=config['variables']['tas']['siglas dataset']
    tad=config['variables']['tad']['siglas dataset']
    tas_umbral_inferior_hipertension=config['parametros']['hipertension']['tas_umbral_inferior']
    tad_umbral_inferior_hipertension=config['parametros']['hipertension']['tad_umbral_inferior']
    df['diabetes_con_hipertension']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    for k in range(0,len(df)):
        if df['diabetes'][k]==1: #Se marcan con 1 los pacientes con diabetes
            if df[tas][k]>=tas_umbral_inferior_hipertension or df[tad][k]>=tad_umbral_inferior_hipertension:
                df['diabetes_con_hipertension'][k]=1
    return df

# Regla extraída del documento "Estándares de atención en DIABETES GUÍA 2023" para atención primaria en las páginas 18-19
def diabetes_con_lipidos(df:pd.DataFrame,config):
    tg=config['variables']['tg']['siglas dataset']
    tg_umbral_inferior_lipidos=config['parametros']['lipidos']['tg_umbral_inferior']
    df['diabetes_con_lipidos']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    for k in range(0,len(df)):
        if df['diabetes'][k]==1: #Se marcan con 1 los pacientes con diabetes
            if df[tg][k]>=tg_umbral_inferior_lipidos:
                df['diabetes_con_lipidos'][k]=1
    return df

# Regla de elaboración propia (ISABIAL)
def diabetes_con_estatinas(df:pd.DataFrame,config) -> pd.DataFrame:
    estatina=config['variables']['estatina']['siglas dataset']
    estatina_identificador=config['parametros']['estatina']['identificador']
    df['diabetes_con_estatinas']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    for k in range(0,len(df)):
        if df['diabetes'][k]==1: #Se marcan con 1 los pacientes con diabetes
            if df[estatina][k]==estatina_identificador:
                df['diabetes_con_estatinas'][k]=1
    return df

# Regla de elaboración propia trivial
def diabetes_mayores_de_65(df:pd.DataFrame,config) -> pd.DataFrame:
    edad=config['variables']['edad']['siglas dataset']
    df['diabetes_mayores_de_65']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    for k in range(0,len(df)):
        if df['diabetes'][k]==1: #Se marcan con 1 los pacientes con diabetes
            if df[edad][k]>=65:
                df['diabetes_mayores_de_65'][k]=1
    return df

# Regla extraída del documento "Estándares de atención en DIABETES GUÍA 2023" para atención primaria en la página 26
def diabetes_mayores_de_65_y_salud_saludable(df:pd.DataFrame,config) -> pd.DataFrame:
    edad=config['variables']['edad']['siglas dataset']
    tas=config['variables']['tas']['siglas dataset']
    tad=config['variables']['tad']['siglas dataset']
    fg=config['variables']['fg']['siglas dataset']
    fg_umbral_inferior_mayores_65_saludables=config['parametros']['mayores_de_65_saludable']['fg_umbral_inferior']
    fg_umbral_superior_mayores_65_saludables=config['parametros']['mayores_de_65_saludable']['fg_umbral_superior']
    tas_umbral_superior_mayores_65_saludables=config['parametros']['mayores_de_65_saludable']['tas_umbral_superior']
    tad_umbral_superior_mayores_65_saludables=config['parametros']['mayores_de_65_saludable']['tad_umbral_superior']
    df['diabetes_mayores_de_65_y_salud_saludable']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    for k in range(0,len(df)):
        if df['diabetes'][k]==1: #Se marcan con 1 los pacientes con diabetes
            if df[edad][k]>=65:
                if df[fg][k]>=fg_umbral_inferior_mayores_65_saludables and df[fg][k]<=fg_umbral_superior_mayores_65_saludables:
                    df['diabetes_mayores_de_65_y_salud_saludable'][k]=1
                if df[tas][k]<=tas_umbral_superior_mayores_65_saludables and df[tad][k]<=tad_umbral_superior_mayores_65_saludables:
                    df['diabetes_mayores_de_65_y_salud_saludable'][k]=1
    return df

# Regla extraída del documento "Estándares de atención en DIABETES GUÍA 2023" para atención primaria en la página 26
def diabetes_mayores_de_65_y_salud_compleja(df:pd.DataFrame,config) -> pd.DataFrame:
    edad=config['variables']['edad']['siglas dataset']
    tas=config['variables']['tas']['siglas dataset']
    tad=config['variables']['tad']['siglas dataset']
    fg=config['variables']['fg']['siglas dataset']
    fg_umbral_inferior_mayores_65_compleja=config['parametros']['mayores_de_65_compleja']['fg_umbral_inferior']
    fg_umbral_superior_mayores_65_compleja=config['parametros']['mayores_de_65_compleja']['fg_umbral_superior']
    tas_umbral_superior_mayores_65_compleja=config['parametros']['mayores_de_65_compleja']['tas_umbral_superior']
    tad_umbral_superior_mayores_65_compleja=config['parametros']['mayores_de_65_compleja']['tad_umbral_superior']
    df['diabetes_mayores_de_65_y_salud_compleja']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    for k in range(0,len(df)):
        if df['diabetes'][k]==1: #Se marcan con 1 los pacientes con diabetes
            if df[edad][k]>=65:
                if df[fg][k]>=fg_umbral_inferior_mayores_65_compleja and df[fg][k]<=fg_umbral_superior_mayores_65_compleja:
                    df['diabetes_mayores_de_65_y_salud_compleja'][k]=1
                if df[tas][k]<=tas_umbral_superior_mayores_65_compleja and df[tad][k]<=tad_umbral_superior_mayores_65_compleja:
                    df['diabetes_mayores_de_65_y_salud_compleja'][k]=1
    return df

# Regla extraída del documento "Estándares de atención en DIABETES GUÍA 2023" para atención primaria en la página 26
def diabetes_mayores_de_65_y_salud_muy_compleja(df:pd.DataFrame,config) -> pd.DataFrame:
    edad=config['variables']['edad']['siglas dataset']
    tas=config['variables']['tas']['siglas dataset']
    tad=config['variables']['tad']['siglas dataset']
    fg=config['variables']['fg']['siglas dataset']
    fg_umbral_inferior_mayores_65_muy_compleja=config['parametros']['mayores_de_65_muy_compleja']['fg_umbral_inferior']
    fg_umbral_superior_mayores_65_muy_compleja=config['parametros']['mayores_de_65_muy_compleja']['fg_umbral_superior']
    tas_umbral_superior_mayores_65_muy_compleja=config['parametros']['mayores_de_65_muy_compleja']['tas_umbral_superior']
    tad_umbral_superior_mayores_65_muy_compleja=config['parametros']['mayores_de_65_muy_compleja']['tad_umbral_superior']
    df['diabetes_mayores_de_65_y_salud_muy_compleja']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    for k in range(0,len(df)):
        if df[edad][k]>=65:
            if df[fg][k]>=fg_umbral_inferior_mayores_65_muy_compleja and df[fg][k]<=fg_umbral_superior_mayores_65_muy_compleja:
                df['diabetes_mayores_de_65_y_salud_muy_compleja'][k]=1
            if df[tas][k]<=tas_umbral_superior_mayores_65_muy_compleja and df[tad][k]<=tad_umbral_superior_mayores_65_muy_compleja:
                df['diabetes_mayores_de_65_y_salud_muy_compleja'][k]=1
    return df
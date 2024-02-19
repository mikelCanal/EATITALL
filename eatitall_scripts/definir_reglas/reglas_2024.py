# Añadimos la regla para detectar la PREDIABETES (RIESGO DE TENER DIÁBETES)
# Regla extraída del documento "Estándares de atención en DIABETES GUÍA 2023" para atención primaria en las páginas 4 y 5

import pandas as pd

def prediabetes(df:pd.DataFrame,imc:str,imc_umbral_inferior_prediabetes:float,tas:str,tas_umbral_inferior_prediabetes:float,
                tad:str,tad_umbral_inferior_prediabetes:float,tg:str,tg_umbral_inferior_prediabetes:float,hba1c:str,
                hba1c_umbral_inferior_prediabetes:float,hba1c_umbral_superior_prediabetes:float,fg:str,
                fg_umbral_inferior_prediabetes:float,fg_umbral_superior_prediabetes:float) -> pd.DataFrame:
    df['prediabetes']=0 #Definimos una columna y la inicializamos con 0s. Los pacientes con prediabetes serán un 1.
    for k in range(0,len(df)):
        # 1- Personas con sobrepeso
        if df[imc][k]>=imc_umbral_inferior_prediabetes: # NOTA: No estamos teniendo en cuenta que para asiáticos-americanos el IMC>=23 porque no tenemos el dato del origen de cada persona
            if df[tas][k]>=tas_umbral_inferior_prediabetes:
                df['prediabetes'][k]=1
            if df[tad][k]>=tad_umbral_inferior_prediabetes:
                df['prediabetes'][k]=1
            # if df[colesterol]<colesterol_umbral_superior_prediabetes:
            #     df['prediabetes'][k]=1
            if df[tg][k]>tg_umbral_inferior_prediabetes:
                df['prediabetes'][k]=1
            # actividadFisica==1
            # -->obseidad==1
            # acanthosisNigricans==1
        # 2- Personas con prediabetes (A1C)
        if df[hba1c][k]>=hba1c_umbral_inferior_prediabetes and df[hba1c][k]<=hba1c_umbral_superior_prediabetes:
            df['prediabetes'][k]=1
        # 3- FPG
        if df[fg][k]>=fg_umbral_inferior_prediabetes and df[fg][k]<=fg_umbral_superior_prediabetes:
            df['prediabetes'][k]=1
        # gdm==1
        # hiv==1
        # pancreatitis==1
        # ifg (ADA)==1

    return df

def diabetes(df:pd.DataFrame,hba1c:str,hba1c_umbral_inferior_diabetes:float,fg:str,
            fg_umbral_inferior_diabetes:float,pg2h_75g_ogtt:str,pg2h_75g_ogtt_umbral_inferior_diabetes:float,
            ga:str,ga_umbral_inferior_diabetes:float) -> pd.DataFrame:
    # Si siguieramos la tabla de la página 4:
    # df['diabetes']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    # for k in range(0,len(df)):
    #     if df[hba1c][k]>=hba1c_umbral_inferior_diabetes:
    #         df['diabetes'][k]=1
    #     if df[fg][k]>=fg_umbral_inferior_diabetes:
    #         df['diabetes'][k]=1
    #     if df[pg2h_75g_ogtt][k]>=pg2h_75g_ogtt_umbral_inferior_diabetes:
    #         df['diabetes'][k]=1
    #     if df[ga][k]>=ga_umbral_inferior_diabetes:
    #         df['diabetes'][k]=1
    # # Pero sabemos que todos son diabéticos, así que:
    df['diabetes']=1
    return df

def diabetes_con_cv(df:pd.DataFrame,acv:str,acv_identificador:int) -> pd.DataFrame:
    df['diabetes_con_cv']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    for k in range(0,len(df)):
        if df['diabetes'][k]==1:
            if df[acv][k]==acv_identificador:
                df['diabetes_con_cv'][k]=1
    return df

# Añadimos la regla para detectar la DIABETES con EFERMEDADES HIPERTENSIÓN
# Regla extraída del documento "Estándares de atención en DIABETES GUÍA 2023" para atención primaria en la página 17


def diabetes_con_hipertension(df:pd.DataFrame,tas:str,tas_umbral_inferior_hipertension:float,tad:str,
                              tad_umbral_inferior_hipertension:float) -> pd.DataFrame:
    df['diabetes_con_hipertension']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    for k in range(0,len(df)):
        if df['diabetes'][k]==1: #Se marcan con 1 los pacientes con diabetes
            if df[tas][k]>=tas_umbral_inferior_hipertension or df[tad][k]>=tad_umbral_inferior_hipertension:
                df['diabetes_con_hipertension'][k]=1
    return df

# Añadimos la regla para detectar la DIABETES con EFERMEDADES LÍPIDOS
# Regla extraída del documento "Estándares de atención en DIABETES GUÍA 2023" para atención primaria en las páginas 18-19

def diabetes_con_lipidos(df:pd.DataFrame,tg:str,tg_umbral_inferior_lipidos:float):
    df['diabetes_con_lipidos']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    for k in range(0,len(df)):
        if df['diabetes'][k]==1: #Se marcan con 1 los pacientes con diabetes
            if df[tg][k]>=tg_umbral_inferior_lipidos:
                df['diabetes_con_lipidos'][k]=1
    return df

# Añadimos la regla para detectar la DIABETES con MEDICACIÓN ESTATINAS
# Regla de elaboración propia (ISABIAL)

def diabetes_con_estatinas(df:pd.DataFrame,estatina:str,estatina_identificador:int) -> pd.DataFrame:
    df['diabetes_con_estatinas']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    for k in range(0,len(df)):
        if df['diabetes'][k]==1: #Se marcan con 1 los pacientes con diabetes
            if df[estatina][k]==estatina_identificador:
                df['diabetes_con_estatinas'][k]=1
    return df

def diabetes_con_vih(df:pd.DataFrame,estatina:str,vih_identificador:int) -> pd.DataFrame:
    df['diabetes_con_vih']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    for k in range(0,len(df)):
        if df['diabetes'][k]==1: #Se marcan con 1 los pacientes con diabetes
            if df[estatina][k]==vih_identificador:
                df['diabetes_con_vih'][k]=1
    return df

# Añadimos la regla para detectar la DIABETES con ADULTO MAYOR
# Regla de elaboración propia trivial

def diabetes_mayores_de_65(df:pd.DataFrame,edad:int) -> pd.DataFrame:
    df['diabetes_mayores_de_65']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    for k in range(0,len(df)):
        if df['diabetes'][k]==1: #Se marcan con 1 los pacientes con diabetes
            if df[edad][k]>=65:
                df['diabetes_mayores_de_65'][k]=1
    return df

# Añadimos la regla para detectar la DIABETES con ADULTO MAYOR Y SALUD SALUDABLE
# Regla extraída del documento "Estándares de atención en DIABETES GUÍA 2023" para atención primaria en la página 26

def diabetes_mayores_de_65_y_salud_saludable(df:pd.DataFrame,edad:int,fg:str,fg_umbral_inferior_mayores_65_saludables:float,
                                             fg_umbral_superior_mayores_65_saludables:float,tas:str,
                                             tas_umbral_superior_mayores_65_saludables:float,tad:str,
                                             tad_umbral_superior_mayores_65_saludables:float) -> pd.DataFrame:
    df['diabetes_mayores_de_65_y_salud_saludable']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    for k in range(0,len(df)):
        if df['diabetes'][k]==1: #Se marcan con 1 los pacientes con diabetes
            if df[edad][k]>=65:
                if df[fg][k]>=fg_umbral_inferior_mayores_65_saludables and df[fg][k]<=fg_umbral_superior_mayores_65_saludables:
                    df['diabetes_mayores_de_65_y_salud_saludable'][k]=1
                if df[tas][k]<=tas_umbral_superior_mayores_65_saludables and df[tad][k]<=tad_umbral_superior_mayores_65_saludables:
                    df['diabetes_mayores_de_65_y_salud_saludable'][k]=1
    return df

# Añadimos la regla para detectar la DIABETES con ADULTO MAYOR Y SALUD COMPLEJA
# Regla extraída del documento "Estándares de atención en DIABETES GUÍA 2023" para atención primaria en la página 26

def diabetes_mayores_de_65_y_salud_compleja(df:pd.DataFrame,edad:int,fg:str,fg_umbral_inferior_mayores_65_compleja:float,
                                            fg_umbral_superior_mayores_65_compleja:float,tas:str,
                                            tas_umbral_superior_mayores_65_compleja:float,tad:str,
                                            tad_umbral_superior_mayores_65_compleja:float) -> pd.DataFrame:
    df['diabetes_mayores_de_65_y_salud_compleja']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    for k in range(0,len(df)):
        if df['diabetes'][k]==1: #Se marcan con 1 los pacientes con diabetes
            if df[edad][k]>=65:
                if df[fg][k]>=fg_umbral_inferior_mayores_65_compleja and df[fg][k]<=fg_umbral_superior_mayores_65_compleja:
                    df['diabetes_mayores_de_65_y_salud_compleja'][k]=1
                if df[tas][k]<=tas_umbral_superior_mayores_65_compleja and df[tad][k]<=tad_umbral_superior_mayores_65_compleja:
                    df['diabetes_mayores_de_65_y_salud_compleja'][k]=1
    return df

# Añadimos la regla para detectar la DIABETES con ADULTO MAYOR Y SALUD MUY COMPLEJA
# Regla extraída del documento "Estándares de atención en DIABETES GUÍA 2023" para atención primaria en la página 26

def diabetes_mayores_de_65_y_salud_muy_compleja(df:pd.DataFrame,edad:int,fg:str,fg_umbral_inferior_mayores_65_muy_compleja:float,
                                                fg_umbral_superior_mayores_65_muy_compleja:float,tas:str,
                                                tas_umbral_superior_mayores_65_muy_compleja:float,tad:str,
                                                tad_umbral_superior_mayores_65_muy_compleja:float) -> pd.DataFrame:
    df['diabetes_mayores_de_65_y_salud_muy_compleja']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    for k in range(0,len(df)):
        if df[edad][k]>=65:
            if df[fg][k]>=fg_umbral_inferior_mayores_65_muy_compleja and df[fg][k]<=fg_umbral_superior_mayores_65_muy_compleja:
                df['diabetes_mayores_de_65_y_salud_muy_compleja'][k]=1
            if df[tas][k]<=tas_umbral_superior_mayores_65_muy_compleja and df[tad][k]<=tad_umbral_superior_mayores_65_muy_compleja:
                df['diabetes_mayores_de_65_y_salud_muy_compleja'][k]=1
    return df
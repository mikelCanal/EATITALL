
import pandas as pd

# Regla extraída del documento "Diagnosis and Classification of Diabetes: Standards of Care in Diabetes — 2024" (1-diagnosis y classif DM) 
# en las página 2
def hiperglucemia(df,config):
    ga=config['variables']['imc']['siglas dataset']#<-- NO ESTÁ DEFINIDO EN CONFIG PORQUE NO SABEMOS COMO ES EN EL DATASET
    ga_umbral_inferior_diabetes=config['parametros']['diabetes']['ga_umbral_inferior']
    df['hiperglucemia']=0
    for k in range(0,len(df)):
        if ga>ga_umbral_inferior_diabetes:
            df['hiperglucemia'][k]=1
    return df

# Regla extraída de una conversación por mail con Kamila reflexionando sobre los datos de del documento "Diagnosis and Classification of Diabetes: 
# Standards of Care in Diabetes — 2024" (1-diagnosis y classif DM) en las página 2
def hiperglucemia_severa(df,config):
    ga=config['variables']['imc']['siglas dataset']
    hba1c=config['variables']['hba1c']['siglas dataset']
    ga_umbral_inferior_diabetes=config['parametros']['diabetes']['gs_umbral_inferior']
    hba1c_umbral_inferior_diabetes=config['parametros']['diabetes']['hba1c_umbral_inferior']
    df['hiperglucemia_severa']=0
    for k in range(0,len(df)):
        if ga>ga_umbral_inferior_diabetes and hba1c>hba1c_umbral_inferior_diabetes:
            df['hiperglucemia_severa'][k]=1
    return df

# Regla extraída del documento "Diagnosis and Classification of Diabetes: Standards of Care in Diabetes — 2024" (1-diagnosis y classif DM) 
# en las página 8
def prediabetes(df:pd.DataFrame,config) -> pd.DataFrame:
    imc=config['variables']['imc']['siglas dataset']
    tas=config['variables']['tas']['siglas dataset']
    tad=config['variables']['tad']['siglas dataset']
    tg=config['variables']['tg']['siglas dataset']
    hba1c=config['variables']['hba1c']['siglas dataset']
    fg=config['variables']['fg']['siglas dataset']
    hdl_colesterol=config['variables']['hdl_colesterol']['siglas dataset'] #<-- No definido en config ni en el dataset
    gdm=config['variables']['gdm']['siglas dataset'] #<-- No definido en config ni en el dataset
    vih=config['variables']['vih']['siglas dataset'] #<-- No definido en config ni en el dataset
    pancreatitis=config['variables']['pancreatitis']['siglas dataset'] #<-- No definido en config ni en el dataset
    obesidad=config['variables']['obesidad']['siglas dataset']#<-- No definido en config ni en el dataset
    acanthosis_nigricans=config['variables']['acanthosis_nigricans']['siglas dataset']#<-- No definido en config ni en el dataset
    imc_umbral_inferior_prediabetes=config['parametros']['prediabetes']['imc_umbral_inferior']
    tas_umbral_inferior_prediabetes=config['parametros']['prediabetes']['tas_umbral_inferior']
    tad_umbral_inferior_prediabetes=config['parametros']['prediabetes']['tad_umbral_inferior']
    tg_umbral_inferior_prediabetes=config['parametros']['prediabetes']['tg_umbral_inferior']
    hba1c_umbral_inferior_prediabetes=config['parametros']['prediabetes']['hba1c_umbral_inferior']
    hba1c_umbral_superior_prediabetes=config['parametros']['prediabetes']['hba1c_umbral_superior']
    fg_umbral_inferior_prediabetes=config['parametros']['prediabetes']['fg_umbral_inferior']
    fg_umbral_superior_prediabetes=config['parametros']['prediabetes']['fg_umbral_superior']
    hdl_colesterol_umbral_superior_prediabetes=config['parametros']['prediabetes']['hdl_colesterol_umbral_superior']
    gdm_identificador=config['parametros']['prediabetes']['gdm_identificador']
    vih_identificador=config['parametros']['prediabetes']['vih_identificador']
    pancreatitis_identificador=config['parametros']['prediabetes']['pancreatitis_identificador']
    obesidad_identificador=config['parametros']['prediabetes']['obesidad_identificador']
    acanthosis_nigricans_identificador=config['parametros']['prediabetes']['acanthosis_nigricans_identificador']
    df['prediabetes']=0 #Definimos una columna y la inicializamos con 0s. Los pacientes con prediabetes serán un 1.
    for k in range(0,len(df)):
        # 1- Personas con sobrepeso
        if df[imc][k]>=imc_umbral_inferior_prediabetes: # NOTA: No estamos teniendo en cuenta que para asiáticos-americanos el IMC>=23 porque no tenemos el dato del origen de cada persona
            if df[tas][k]>=tas_umbral_inferior_prediabetes:
                df['prediabetes'][k]=1
            if df[tad][k]>=tad_umbral_inferior_prediabetes:
                df['prediabetes'][k]=1
            if df[hdl_colesterol]<hdl_colesterol_umbral_superior_prediabetes:
                df['prediabetes'][k]=1
            if df[tg][k]>tg_umbral_inferior_prediabetes:
                df['prediabetes'][k]=1
            if df[obesidad][k]==obesidad_identificador:
                df['prediabetes'][k]=1
            if df[acanthosis_nigricans][k]==acanthosis_nigricans_identificador:
                df['prediabetes'][k]=1
        # 2- Personas con prediabetes (A1C)
        if df[hba1c][k]>=hba1c_umbral_inferior_prediabetes and df[hba1c][k]<=hba1c_umbral_superior_prediabetes:
            df['prediabetes'][k]=1
        # 3- FPG
        if df[fg][k]>=fg_umbral_inferior_prediabetes and df[fg][k]<=fg_umbral_superior_prediabetes:
            df['prediabetes'][k]=1
        if df[gdm]==gdm_identificador:
            df['prediabetes'][k]=1
        if df[vih]==vih_identificador:
            df['prediabetes'][k]=1
        if df[pancreatitis]==pancreatitis_identificador:
            df['prediabetes'][k]=1
    return df

# Regla extraída del documento "Diagnosis and Classification of Diabetes: Standards of Care in Diabetes — 2024" (1-diagnosis y classif DM) 
# en las página 2
def diabetes(df:pd.DataFrame,config):
    hba1c=config['variables']['hba1c']['siglas dataset']
    fg=config['variables']['fg']['siglas dataset']
    pg2h_75g_ogtt=config['variables']['pg2h_75g_ogtt']['siglas dataset']
    ga=config['variables']['ga']['siglas dataset']
    hba1c_umbral_inferior_diabetes=config['parametros']['diabetes']['hba1c_umbral_inferior']
    fg_umbral_inferior_diabetes=config['parametros']['diabetes']['fg_umbral_inferior']
    pg2h_75g_ogtt_umbral_inferior_diabetes=config['parametros']['diabetes']['pg2h_75g_ogtt_umbral_inferior']
    ga_umbral_inferior_diabetes=config['parametros']['diabetes']['ga_umbral_inferior']
    # df['diabetes']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    # for k in range(0,len(df)):
    #     if df[hba1c][k]>=hba1c_umbral_inferior_diabetes:
    #         df['diabetes'][k]=1
    #     if df[fg][k]>=fg_umbral_inferior_diabetes:
    #         df['diabetes'][k]=1
    #           if df[fg][k]>=100 and df[fg][k]<=100
    #     if df[pg2h_75g_ogtt][k]>=pg2h_75g_ogtt_umbral_inferior_diabetes:
    #         df['diabetes'][k]=1
    #     if df[ga][k]>=ga_umbral_inferior_diabetes:
    #         df['diabetes'][k]=1
    # # Pero sabemos que todos son diabéticos, así que:
    df['diabetes']=1
    return df

# Regla extraída del documento "Diagnosis and Classification of Diabetes: Standards of Care in Diabetes — 2024" (1-diagnosis y classif DM) 
# en las página 11
def diabetes_con_vih(df:pd.DataFrame,config) -> pd.DataFrame:
    vih=config['variables']['vih']['siglas dataset']
    vih_identificador=config['parametros']['vih']['identificador']
    df['diabetes_con_vih']=0 #Definimos una columna y la inicializamos con 0s. Los que la tengan serán un 1.
    for k in range(0,len(df)):
        if df['diabetes'][k]==1: #Se marcan con 1 los pacientes con diabetes
            if df[vih][k]==vih_identificador:
                df['diabetes_con_vih'][k]=1
    return df

def corr_hba1c_fg(df:pd.DataFrame,config) -> pd.DataFrame:
    hba1c=config['variables']['hba1c']['siglas dataset']
    fg=config['variables']['fg']['siglas dataset']
    porcentaje_hba1c_5=config['parametros']['corr_hba1c_fg']['porcentaje_hba1c_5']
    hba1c_5_umbral_inferior=config['parametros']['corr_hba1c_fg']['hba1c_5_umbral_inferior']
    hba1c_5_umbral_superior=config['parametros']['corr_hba1c_fg']['hba1c_5_umbral_superior']
    porcentaje_hba1c_6=config['parametros']['corr_hba1c_fg']['porcentaje_hba1c_6']
    hba1c_6_umbral_inferior=config['parametros']['corr_hba1c_fg']['hba1c_6_umbral_inferior']
    hba1c_6_umbral_superior=config['parametros']['corr_hba1c_fg']['hba1c_6_umbral_superior']
    porcentaje_hba1c_7=config['parametros']['corr_hba1c_fg']['porcentaje_hba1c_7']
    hba1c_7_umbral_inferior=config['parametros']['corr_hba1c_fg']['hba1c_7_umbral_inferior']
    hba1c_7_umbral_superior=config['parametros']['corr_hba1c_fg']['hba1c_7_umbral_superior']
    porcentaje_hba1c_8=config['parametros']['corr_hba1c_fg']['porcentaje_hba1c_8']
    hba1c_8_umbral_inferior=config['parametros']['corr_hba1c_fg']['hba1c_8_umbral_inferior']
    hba1c_8_umbral_superior=config['parametros']['corr_hba1c_fg']['hba1c_8_umbral_superior']
    porcentaje_hba1c_9=config['parametros']['corr_hba1c_fg']['porcentaje_hba1c_9']
    hba1c_9_umbral_inferior=config['parametros']['corr_hba1c_fg']['hba1c_9_umbral_inferior']
    hba1c_9_umbral_superior=config['parametros']['corr_hba1c_fg']['hba1c_9_umbral_superior']
    porcentaje_hba1c_10=config['parametros']['corr_hba1c_fg']['porcentaje_hba1c_10']
    hba1c_10_umbral_inferior=config['parametros']['corr_hba1c_fg']['hba1c_10_umbral_inferior']
    hba1c_10_umbral_superior=config['parametros']['corr_hba1c_fg']['hba1c_10_umbral_superior']
    porcentaje_hba1c_11=config['parametros']['corr_hba1c_fg']['porcentaje_hba1c_11']
    hba1c_11_umbral_inferior=config['parametros']['corr_hba1c_fg']['hba1c_11_umbral_inferior']
    hba1c_11_umbral_superior=config['parametros']['corr_hba1c_fg']['hba1c_11_umbral_superior']
    porcentaje_hba1c_12=config['parametros']['corr_hba1c_fg']['porcentaje_hba1c_12']
    hba1c_12_umbral_inferior=config['parametros']['corr_hba1c_fg']['hba1c_12_umbral_inferior']
    hba1c_12_umbral_superior=config['parametros']['corr_hba1c_fg']['hba1c_12_umbral_superior']
    df['corr_hba1c_fg']=0 #Definimos una columna y la inicializamos con 0s. Si hay correlación ponemos un 1.
    for k in range(0,len(df)):
        if df[hba1c][k]>=porcentaje_hba1c_5+0.5 and df[hba1c][k]<porcentaje_hba1c_5+0.5 and df[fg][k]>=hba1c_5_umbral_inferior and df[fg][k]<=hba1c_5_umbral_superior:
            df['corr_hba1c_fg'][k]=1
        if df[hba1c][k]>=porcentaje_hba1c_6+0.5 and df[hba1c][k]<porcentaje_hba1c_6+0.5 and df[fg][k]>=hba1c_6_umbral_inferior and df[fg][k]<=hba1c_6_umbral_superior:
            df['corr_hba1c_fg'][k]=1
        if df[hba1c][k]>=porcentaje_hba1c_7+0.5 and df[hba1c][k]<porcentaje_hba1c_7+0.5 and df[fg][k]>=hba1c_7_umbral_inferior and df[fg][k]<=hba1c_7_umbral_superior:
            df['corr_hba1c_fg'][k]=1
        if df[hba1c][k]>=porcentaje_hba1c_8+0.5 and df[hba1c][k]<porcentaje_hba1c_8+0.5 and df[fg][k]>=hba1c_8_umbral_inferior and df[fg][k]<=hba1c_8_umbral_superior:
            df['corr_hba1c_fg'][k]=1
        if df[hba1c][k]>=porcentaje_hba1c_9+0.5 and df[hba1c][k]<porcentaje_hba1c_9+0.5 and df[fg][k]>=hba1c_9_umbral_inferior and df[fg][k]<=hba1c_9_umbral_superior:
            df['corr_hba1c_fg'][k]=1
        if df[hba1c][k]>=porcentaje_hba1c_10+0.5 and df[hba1c][k]<porcentaje_hba1c_10+0.5 and df[fg][k]>=hba1c_10_umbral_inferior and df[fg][k]<=hba1c_10_umbral_superior:
            df['corr_hba1c_fg'][k]=1
        if df[hba1c][k]>=porcentaje_hba1c_11+0.5 and df[hba1c][k]<porcentaje_hba1c_11+0.5 and df[fg][k]>=hba1c_11_umbral_inferior and df[fg][k]<=hba1c_11_umbral_superior:
            df['corr_hba1c_fg'][k]=1
        if df[hba1c][k]>=porcentaje_hba1c_12+0.5 and df[hba1c][k]<porcentaje_hba1c_12+0.5 and df[fg][k]>=hba1c_12_umbral_inferior and df[fg][k]<=hba1c_12_umbral_superior:
            df['corr_hba1c_fg'][k]=1
    return df

def hipoglucemia(df:pd.DataFrame,config) -> pd.DataFrame:
    hba1c=config['variables']['hba1c']['siglas dataset']
    hba1c_nivel1_umbral_inferior=config['parametros']['hipoglucemia']['nivel1_hba1c_umbral_inferior']
    hba1c_nivel1_umbral_superior=config['parametros']['hipoglucemia']['nivel1_hba1c_umbral_superior']
    hba1c_nivel2_umbral_inferior=config['parametros']['hipoglucemia']['nivel2_hba1c_umbral_inferior']
    df['hipoglucemia']=0
    for k in range(0,len(df)):
        if df[hba1c][k]>=hba1c_nivel1_umbral_inferior and df[hba1c][k]<hba1c_nivel1_umbral_superior:
            df['hipoglucemia'][k]=1
        if df[hba1c][k]<hba1c_nivel2_umbral_inferior:
            df['hipoglucemia'][k]=2
    return df

def bajo_nivel_insulina(df:pd.DataFrame,config) -> pd.DataFrame:
    insulina=config['variables']['insulina']['siglas dataset']
    insulina_umbral_inferior=config['parametros']['bajo_nivel_insulina']['insulina_umbral_inferior']
    df['bajo_nivel_insulina']=0
    for k in range(0,len(df)):
        if df[insulina][k]<insulina_umbral_inferior:
            df['bajo_nivel_insulina'][k]=1
    return df
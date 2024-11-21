import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import plot_tree
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import json
from io import BytesIO
import ollama
import re
from itertools import combinations

from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict

import sys


class ckm:
    def añadir_ckms(df,config):
        df=ckm.ckm1(df,config)
        df=ckm.ckm2(df,config)
        df=ckm.ckm3(df,config)
        df=ckm.ckm4(df,config)
        return df

    def ckm1(df,config):
        imc = config['variables']['mediciones_generales']['imc']['siglas dataset']
        fg = config['variables']['analisis_sangre']['fg']['siglas dataset']
        hba1c = config['variables']['analisis_sangre']['hba1c']['siglas dataset']
        df['ckm1']=0
        for k in range(0,len(df)):
            if df[imc][k]>=25 or df[fg][k]>=100 or (df[hba1c][k]>=5.7 and df[hba1c][k]<=6.4):
                df['ckm1'][k]=1
        return df



    def ckm2(df,config):
        tg=config['variables']['analisis_sangre']['tg']['siglas dataset']
        tg_umbral=config['parametros']['hipertriglicidemia']['tg_umbral']
        hipertriglicidemia=f'reglas hipertriglicidemia ({tg}>{tg_umbral})'
        tas = config['variables']['mediciones_generales']['tas']['siglas dataset']
        tad = config['variables']['mediciones_generales']['tad']['siglas dataset']
        tas_umbral_inferior = config['parametros']['hipertension']['tas_umbral_inferior']
        tad_umbral_inferior = config['parametros']['hipertension']['tad_umbral_inferior']
        hipertension=f'reglas hipertension ({tas}>{tas_umbral_inferior} o {tad}>{tad_umbral_inferior})'
        df['ckm2']=0
        for k in range(0,len(df)):
            if df[hipertriglicidemia][k]==1 or df[hipertension][k]==1: #or diabetes==1
                df['ckm2'][k]=1

        return df

    def ckm3(df,config):
        edad='Edad'
        df['ckm3']=0
        for k in range(0,len(df)):
            if df[edad][k]>=70:
                df['ckm3'][k]=1
        return df

    def ckm4(df,config):
        edad='Edad'
        df['ckm4']=0

        for k in range(0,len(df)):
            if df[edad][k]<70:
                df['ckm4'][k]=1
        return df

class reglas_2024_10_cardiovascular_disease_and_risk_management:

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
    
    def añadir_reglas(df:pd.DataFrame,config):
        df,_=reglas_2024_10_cardiovascular_disease_and_risk_management.tension_alta(df,config, path_variables_ausentes='variables_ausentes.txt')
        df,_=reglas_2024_10_cardiovascular_disease_and_risk_management.hipertension(df,config, path_variables_ausentes='variables_ausentes.txt')
        # df,_=diabetes_con_hipertension(df,config, path_variables_ausentes='variables_ausentes.txt')
        # df,_=tratamiento_hipertension(df,config, path_variables_ausentes='variables_ausentes.txt')
        df,_=reglas_2024_10_cardiovascular_disease_and_risk_management.presion_arterial_ligeramente_elevada_hipertension(df,config, path_variables_ausentes='variables_ausentes.txt')
        # df,_=medicacion_hipertension(df,config, path_variables_ausentes='variables_ausentes.txt')
        df,_=reglas_2024_10_cardiovascular_disease_and_risk_management.hipertension_resistente(df,config, path_variables_ausentes='variables_ausentes.txt')
        df,_=reglas_2024_10_cardiovascular_disease_and_risk_management.medicacion_hipertension_con_enfermedad_renal(df,config, path_variables_ausentes='variables_ausentes.txt')
        # df,_=dislipemia(df,config, path_variables_ausentes='variables_ausentes.txt')
        # df,_=dislipemia_con_diabetes(df,config, path_variables_ausentes='variables_ausentes.txt')
        # df,_=medicacion_estatinas_prevencion_primaria(df,config, path_variables_ausentes='variables_ausentes.txt')
        # df,_=medicacion_estatinas_prevencion_secundaria(df,config, path_variables_ausentes='variables_ausentes.txt')
        df,_=reglas_2024_10_cardiovascular_disease_and_risk_management.hipertriglicidemia(df,config, path_variables_ausentes='variables_ausentes.txt')
        df,_=reglas_2024_10_cardiovascular_disease_and_risk_management.baja_lipoproteina_hdl(df,config, path_variables_ausentes='variables_ausentes.txt')
        df,_=reglas_2024_10_cardiovascular_disease_and_risk_management.hipercolesterolemia(df,config, path_variables_ausentes='variables_ausentes.txt')

        return df

class reglas_2024_2_diagnosis_and_classification_of_diabetis:
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
    
    def añadir_reglas(df:pd.DataFrame,config):
        df,_=reglas_2024_2_diagnosis_and_classification_of_diabetis.hiperglucemia(df,config, path_variables_ausentes='variables_ausentes.txt')
        df,_=reglas_2024_2_diagnosis_and_classification_of_diabetis.diabetes(df,config, path_variables_ausentes='variables_ausentes.txt')
        df,_=reglas_2024_2_diagnosis_and_classification_of_diabetis.prediabetes(df,config, path_variables_ausentes='variables_ausentes.txt')
        df,_=reglas_2024_2_diagnosis_and_classification_of_diabetis.riesgo_prediabetes(df,config, path_variables_ausentes='variables_ausentes.txt')
        return df

class reglas_2024_6_glycemic_goals_and_hypoglycemia:
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
    def añadir_reglas(df:pd.DataFrame,config):
        df,_=reglas_2024_6_glycemic_goals_and_hypoglycemia.corr_hba1c_fg(df,config, path_variables_ausentes='variables_ausentes.txt')
        df,_=reglas_2024_6_glycemic_goals_and_hypoglycemia.bajo_nivel_insulina(df,config, path_variables_ausentes='variables_ausentes.txt')
        df,_=reglas_2024_6_glycemic_goals_and_hypoglycemia.hipoglucemia(df,config, path_variables_ausentes='variables_ausentes.txt')
        return df

class reglas_2024_8_obesity:
    # página 2
    def obseidad_clase_i(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
        imc = config['variables']['mediciones_generales']['imc']['siglas dataset']
        imc_umbral_inferior = config['parametros']['obesidad_clase_i']['obesidad_clase_i_umbral_inferior']
        imc_umbral_superior = config['parametros']['obesidad_clase_i']['obesidad_clase_i_umbral_superior']

        df[f'reglas obesidad_clase_i ({imc_umbral_inferior}<{imc}<{imc_umbral_superior})'] = 0

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
                    df[f'reglas obesidad_clase_i ({imc_umbral_inferior}<{imc}<{imc_umbral_superior})'][k] = 1
            except KeyError:
                pass

        available_vars.append(f'reglas obesidad_clase_i ({imc_umbral_inferior}<{imc}<{imc_umbral_superior})')
        df_vars_identificadas = df[available_vars]
    
        return df, df_vars_identificadas

    def obseidad_clase_ii(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
        imc = config['variables']['mediciones_generales']['imc']['siglas dataset']
        imc_umbral_inferior = config['parametros']['obesidad_clase_ii']['obesidad_clase_ii_umbral_inferior']
        imc_umbral_superior = config['parametros']['obesidad_clase_ii']['obesidad_clase_ii_umbral_superior']

        df[f'reglas obesidad_clase_ii ({imc_umbral_inferior}<{imc}<{imc_umbral_superior})'] = 0

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
                    df[f'reglas obesidad_clase_ii ({imc_umbral_inferior}<{imc}<{imc_umbral_superior})'][k] = 1
            except KeyError:
                pass

        available_vars.append(f'reglas obesidad_clase_ii ({imc_umbral_inferior}<{imc}<{imc_umbral_superior})')
        df_vars_identificadas = df[available_vars]
        
        return df, df_vars_identificadas

    def obseidad_clase_iii(df: pd.DataFrame, config, path_variables_ausentes='variables_ausentes.txt'):
        imc = config['variables']['mediciones_generales']['imc']['siglas dataset']
        imc_umbral = config['parametros']['obesidad_clase_iii']['obesidad_clase_iii_umbral']

        df[f'reglas obesidad_clase_iii ({imc}>{imc_umbral})'] = 0

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
                    df[f'reglas obesidad_clase_iii ({imc}>{imc_umbral})'][k] = 1
            except KeyError:
                pass

        available_vars.append(f'reglas obesidad_clase_iii ({imc}>{imc_umbral})')
        df_vars_identificadas = df[available_vars]

        return df, df_vars_identificadas
    
    def añadir_reglas(df:pd.DataFrame,config):
        df,_=reglas_2024_8_obesity.obseidad_clase_i(df,config, path_variables_ausentes='variables_ausentes.txt')
        df,_=reglas_2024_8_obesity.obseidad_clase_ii(df,config, path_variables_ausentes='variables_ausentes.txt')
        df,_=reglas_2024_8_obesity.obseidad_clase_iii(df,config, path_variables_ausentes='variables_ausentes.txt')

        # df,_=hipertension(df,config, path_variables_ausentes='variables_ausentes.txt')

        return df

class normalizar_datos:
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

class extraccion_entidades:
    def otras_ecv(df,variables=['otras_ecv']): #dummy vars
        df = pd.get_dummies(df, columns=variables)
        return df

    def dx_principal(df,variables=['dx_principal']): #dummy vars
        df = pd.get_dummies(df, columns=variables)
        return df

    def dx_secundario(df,variables=['dx_asociados','dx_asociados_2','dx_asociados_3','dx_asociados_4','dx_asociados_5']):
        # Crear un DataFrame vacío para las dummies
        dummy_df = pd.DataFrame(index=df.index)
        
        # Crear un conjunto para mantener un registro de los valores únicos ya procesados
        seen_values = set()
        
        # Procesar cada columna especificada
        for column in variables:
            if column in df.columns:
                for index, value in df[column].iteritems():
                    # Formatear el nombre de la columna dummy basada en el valor del texto
                    dummy_name = 'Diagnósticos asociados_' + str(value)
                    
                    # Revisar si el valor ya ha sido visto y usado para crear una columna dummy
                    if dummy_name not in seen_values:
                        # Si no está en seen_values, lo añadimos y creamos la columna en dummy_df
                        seen_values.add(dummy_name)
                        dummy_df[dummy_name] = 0
                    
                    # Asignar 1 a la fila correspondiente en la columna dummy
                    dummy_df.at[index, dummy_name] = 1

        # Concatenar el DataFrame original con el de dummies
        result_df = pd.concat([df, dummy_df], axis=1)
        result_df.drop(columns=variables, inplace=True)
        return result_df

    def medicamentos(df,variables=['medicamento','medicamento_2','medicamento_3','medicamento_4','medicamento_5','medicamento_6',
                                'medicamento_7','medicamento_8','medicamento_9','medicamento_10','medicamento_11','medicamento_12']):
        # Crear un DataFrame vacío para las dummies
        dummy_df = pd.DataFrame(index=df.index)
        
        # Crear un conjunto para mantener un registro de los valores únicos ya procesados
        seen_values = set()
        
        # Procesar cada columna especificada
        for column in variables:
            if column in df.columns:
                for index, value in df[column].iteritems():
                    # Formatear el nombre de la columna dummy basada en el valor del texto
                    dummy_name = 'Medicamento_' + str(value)
                    
                    # Revisar si el valor ya ha sido visto y usado para crear una columna dummy
                    if dummy_name not in seen_values:
                        # Si no está en seen_values, lo añadimos y creamos la columna en dummy_df
                        seen_values.add(dummy_name)
                        dummy_df[dummy_name] = 0
                    
                    # Asignar 1 a la fila correspondiente en la columna dummy
                    dummy_df.at[index, dummy_name] = 1

        # Concatenar el DataFrame original con el de dummies
        result_df = pd.concat([df, dummy_df], axis=1)
        result_df.drop(columns=variables, inplace=True)
        return result_df

    def rec_nutricionales(df,variable='Recomendaciones nutricionales recibidas'):
        #Hacer un prompt
        #Procesar la salida del prompt para quedarnos con una lista
        #Aplicarlo a todas las filas
        #***igual vale al pena conseguir las listas con GPT4o y Ctrl+C Ctrl+V
        #Aplicar dummy variables dada una lista ['alimento1','alimento2',...]
        # df = pd.get_dummies(df, columns=variables)

        unique_values = set([item for sublist in df[variable+'-Lista'] for item in sublist])

        # 2. Crear columnas dummy para cada valor único
        for value in unique_values:
            df[variable+'_'+value] = df[variable+'-Lista'].apply(lambda x: 1 if value in x else 0)
        df.drop(columns=[variable+'-Lista'], inplace=True)
        df.drop(columns=[variable], inplace=True)
        return df

    def cons_dietetico(df,variable='Consejo dietético'):
        # df['cons_dietetico_1']=0
        # df['cons_dietetico_2']=0
        # df=df.drop('cons_dietetico', axis=1)
        # df = pd.get_dummies(df, columns=variables)
        # 1. Expandir todas las listas en la columna a una sola lista
        unique_values = set([item for sublist in df[variable+'-Lista'] for item in sublist])

        # 2. Crear columnas dummy para cada valor único
        for value in unique_values:
            df[variable+'_'+value] = df[variable+'-Lista'].apply(lambda x: 1 if value in x else 0)
        df.drop(columns=[variable+'-Lista'], inplace=True)
        df.drop(columns=[variable], inplace=True)
        return df

    def juicio_cl(df):
        df['Juicio clínico.1']=0
        df['Juicio clínico.2']=0
        df=df.drop('Juicio clínico', axis=1)
        return df

    def texto_libre_a_categorias(df,variable='Consejo dietético',categorias=""):
        if True==False:
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
            return df
        if True==True:
            df=pd.get_dummies(df, columns=[variable])
            return df

st.title("Generación automática de perfiles clínicos con recomendaciones nutricionales")

categorias_rec_nutri = st.text_input("Escribe aquí la lista de categorías de las Recomendaciones nutricionales (suplementos). Formato categoría1: descripción de la categoría1; categoría2:descripción de la categoría2; etc. \nEjemplo: hidroferol: vitmaina D, pérdida ósea; diacare: Dieta completa normocalórica e hiperproteica con un perfil cardiosaludable")
categorias_consejo_dietetico = st.text_input("Escribe aquí la lista de categorías del Consejo dietético. Formato categoría1: descripción de la categoría1; categoría2:descripción de la categoría2; etc. \nEjemplo: dieta_hipocalórica: dieta baja en calorías, restricción de calorías; dieta_osmocalórica: dieta normocalórica")

st.write("Cargue los Historiales Clínicos Electrónicos en formato csv con el separador ';'. (Cargar el archivo LABELS)")
uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

if uploaded_file is not None:
    # Para leer el archivo CSV
    df = pd.read_csv(uploaded_file,sep=';')
    # Mostrar el dataframe en la aplicación
else:
    st.write("Por favor, carga un archivo CSV.")

df_crudo=df

st.title('Estadísticas generales')
# st.write("Columnas disponibles en el DataFrame:", df_crudo.columns)

# Filtrar solo valores string de la columna "Diagnósticos principal"
string_values = df_crudo['Diagnósticos principal'].astype(str)

# Contar las frecuencias de cada diagnóstico
frequencies = string_values.value_counts().reset_index()
frequencies.columns = ['Diagnóstico', 'Frecuencia']

# Selección de cuántas variables mostrar
num_variables = st.slider(
    "Selecciona cuántos diagnósticos ver",
    min_value=1,
    max_value=30,
    value=5
)

# Mostrar la tabla con los diagnósticos más frecuentes
st.table(frequencies.head(num_variables))

# Crear botones de descarga para cada diagnóstico
for diagnosis in frequencies.head(num_variables)['Diagnóstico']:
    sub_df = df_crudo[df_crudo[['Diagnósticos principal']].apply(lambda row: diagnosis in row.values, axis=1)]
    
    # Crear un archivo Excel en memoria
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        sub_df.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)

    # Botón de descarga
    st.download_button(
        label=f"Descargar pacientes con {diagnosis}",
        data=output,
        file_name=f"Pacientes_{diagnosis}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"diagnostico_principal_{diagnosis}"
    )

# Filtrar columnas que contienen "Diagnósticos asociados"
associated_columns = [col for col in df_crudo.columns if "Diagnósticos asociados" in col]

if associated_columns:
    # Combinar todas las columnas de diagnósticos asociados en una sola serie
    all_associated = pd.concat([df_crudo[col].dropna() for col in associated_columns])

    # Contar las frecuencias de cada diagnóstico
    frequencies = all_associated.value_counts().reset_index()
    frequencies.columns = ['Diagnóstico', 'Frecuencia']

    # Selección de cuántas variables mostrar
    num_variables = st.slider(
        "Selecciona cuántos diagnósticos asociados ver",
        min_value=1,
        max_value=30,
        value=5
    )

    # Mostrar la tabla con los diagnósticos más frecuentes
    st.table(frequencies.head(num_variables))

    # Crear botones de descarga para cada diagnóstico
    for diagnosis in frequencies.head(num_variables)['Diagnóstico']:
        sub_df = df_crudo[df_crudo[associated_columns].apply(lambda row: diagnosis in row.values, axis=1)]
        
        # Crear un archivo Excel en memoria
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            sub_df.to_excel(writer, index=False, sheet_name='Sheet1')
        output.seek(0)

        # Botón de descarga
        st.download_button(
            label=f"Descargar pacientes con {diagnosis}",
            data=output,
            file_name=f"Pacientes_{diagnosis}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"diagnostico_asociado_{diagnosis}"
        )
else:
    st.error("No se encontraron columnas de 'Diagnósticos asociados'.")


# Función para extraer la etiqueta entre guiones
def extraer_etiqueta(texto):
    if pd.isna(texto):
        return ""  # Manejar valores NaN
    import re
    match = re.search(r'-(.*?)\-', texto)
    return match.group(1).strip() if match else ""

# Crear la nueva columna con las etiquetas
df_crudo['Anotación consejo dietético'] = df_crudo['Consejo dietético'].apply(extraer_etiqueta)

# Filtrar solo valores string de la columna "Anotación consejo dietético"
string_values = df_crudo['Anotación consejo dietético'].astype(str)

# Contar las frecuencias de cada diagnóstico
frequencies = string_values.value_counts().reset_index()
frequencies.columns = ['Consejo dietético anotado', 'Frecuencia']

# Selección de cuántas variables mostrar
num_variables = st.slider(
    "Selecciona cuántos consejos dietéticos anotados ver",
    min_value=1,
    max_value=30,
    value=5
)

# Mostrar la tabla con los diagnósticos más frecuentes
st.table(frequencies.head(num_variables))

# Crear botones de descarga para cada diagnóstico
for consejo in frequencies.head(num_variables)['Consejo dietético anotado']:
    sub_df = df_crudo[df_crudo[['Anotación consejo dietético']].apply(lambda row: consejo in row.values, axis=1)]
    
    # Crear un archivo Excel en memoria
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        sub_df.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)

    # Botón de descarga
    st.download_button(
        label=f"Descargar pacientes con {consejo}",
        data=output,
        file_name=f"Pacientes_{consejo}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"anotado_{consejo}"
    )


# Filtrar solo valores string de la columna "Diagnósticos principal"
string_values = df_crudo['Recomendaciones nutricionales recibidas'].astype(str)

# Contar las frecuencias de cada diagnóstico
frequencies = string_values.value_counts().reset_index()
frequencies.columns = ['Recomendaciones nutricionales recibidas', 'Frecuencia']

# Selección de cuántas variables mostrar
num_variables = st.slider(
    "Selecciona cuántas recomendaciones nutricionales recibidas ver",
    min_value=1,
    max_value=30,
    value=5
)

# Mostrar la tabla con los diagnósticos más frecuentes
st.table(frequencies.head(num_variables))

# Crear botones de descarga para cada diagnóstico
for recomendaciones in frequencies.head(num_variables)['Recomendaciones nutricionales recibidas']:
    sub_df = df_crudo[df_crudo[['Recomendaciones nutricionales recibidas']].apply(lambda row: recomendaciones in row.values, axis=1)]
    
    # Crear un archivo Excel en memoria
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        sub_df.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)

    # Botón de descarga
    st.download_button(
        label=f"Descargar pacientes con {recomendaciones}",
        data=output,
        file_name=f"Pacientes_{recomendaciones}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"{recomendaciones}"
    )


def convert_df_to_bytes(df_str):
    """
    Convierte el string de un CSV a un formato adecuado para la descarga en Streamlit.
    """
    return BytesIO(df_str.encode())

st.write("Vuelve a cargar los Historiales Clínicos Electrónicos en formato csv con el separador ';'. (Cargar el archivo LABELS)")
uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

if uploaded_file is not None:
    # Para leer el archivo CSV
    df = pd.read_csv(uploaded_file,sep=';')
    # Mostrar el dataframe en la aplicación
else:
    st.write("Por favor, carga un archivo CSV.")

config_path='config.json'

with open(config_path, 'r') as archivo:
    config = json.load(archivo)


df=normalizar_datos.convertir_si_no_a_0_1(df)
df=normalizar_datos.convertir_mujer_hombre_a_1_2(df)
df=normalizar_datos.convertir_lugar_nac_a_numero(df)
df=normalizar_datos.convertir_dm_tipo_a_numero(df)
df=normalizar_datos.convertir_ic_tipo_a_numero(df)
df=normalizar_datos.convertir_careme_estadio_a_numero(df)
df=normalizar_datos.convertir_intolerancias_a_numero(df)
df=normalizar_datos.eliminar_variables(df,variables=['Record ID','Código del paciente','Presión arterial','Pauta de ejercicio físico recomendada','Complete?','Complete?.1','Complete?.2'],eliminar_nans=False)
df=normalizar_datos.fecha_a_timestamp(df,variables=['Fecha diagnóstico DM'])
df=normalizar_datos.comas_a_puntos(df,variables=['FIB-4','Densidad'])
df=normalizar_datos.ajustar_decimal_dividir_por_mil(df,variables=['FIB-4'])
df=normalizar_datos.ajustar_punto_decimal(df,variables=['Densidad'])
df=normalizar_datos.gestionar_nans(df)

df['Fecha diagnóstico DM'] = pd.to_datetime(df['Fecha diagnóstico DM'], unit='s')
df['Fecha diagnóstico DM'] = df['Fecha diagnóstico DM'].dt.strftime('%Y-%m-%d')

csv = df.to_csv(index=False)
b64 = convert_df_to_bytes(csv)

# Crear el botón de descarga para esta combinación
st.download_button(
    label=f"Descargar datos normalizados como CSV",
    data=b64,
    file_name=f"data_norm.csv",
    mime='text/csv',
    key=f"data_norm"
        )

df=reglas_2024_2_diagnosis_and_classification_of_diabetis.añadir_reglas(df,config)
df=reglas_2024_6_glycemic_goals_and_hypoglycemia.añadir_reglas(df,config)
df=reglas_2024_10_cardiovascular_disease_and_risk_management.añadir_reglas(df,config)
df=reglas_2024_8_obesity.añadir_reglas(df,config)
df=ckm.añadir_ckms(df,config)

csv = df.to_csv(index=False)
b64 = convert_df_to_bytes(csv)

# Crear el botón de descarga para esta combinación
st.download_button(
    label=f"Descargar datos normalizados y con las reglas como CSV",
    data=b64,
    file_name=f"data_norm_reglas.csv",
    mime='text/csv',
    key=f"data_norm_reglas"
        )

df=extraccion_entidades.otras_ecv(df,variables=['Otras ECV']) #dummy vars
df=extraccion_entidades.dx_principal(df,variables=['Diagnósticos principal']) #dummy vars
df=extraccion_entidades.dx_secundario(df,variables=['Diagnósticos asociados','Diagnósticos asociados.1','Diagnósticos asociados.2',
                                                    'Diagnósticos asociados.3','Diagnósticos asociados.4'])
df=extraccion_entidades.medicamentos(df,variables=['Medicamento','Medicamento.1','Medicamento.2','Medicamento.3','Medicamento.4',
                                                   'Medicamento.5','Medicamento.6','Medicamento.7','Medicamento.8',
                                                   'Medicamento.9','Medicamento.10','Medicamento.11'])
# df=extraccion_entidades.rec_nutricionales(df,variable='Recomendaciones nutricionales recibidas')
# df=extraccion_entidades.cons_dietetico(df,variable='Consejo dietético')
df=extraccion_entidades.texto_libre_a_categorias(df,variable='Recomendaciones nutricionales recibidas',categorias=categorias_rec_nutri)
df=extraccion_entidades.texto_libre_a_categorias(df,variable='Consejo dietético',categorias=categorias_consejo_dietetico)
df=extraccion_entidades.juicio_cl(df)
# # Generamos un csv con los datos generados con las reglas y lo guardamos

# df=normalizar_datos.gestionar_nans(df)

csv = df.to_csv(index=False)
b64 = convert_df_to_bytes(csv)

# Crear el botón de descarga para esta combinación
st.download_button(
    label=f"Descargar datos normalizados, con las reglas y extracción de entidades como CSV",
    data=b64,
    file_name=f"data_norm_reglas_extraccion_entidades.csv",
    mime='text/csv',
    key=f"data_norm_reglas_extraccion_entidades"
        )

# df = df.drop('Recomendaciones nutricionales recibidas', axis=1)
# df = df.drop('Consejo dietético', axis=1)


# Calculando los porcentajes
porcentaje_ckm1 = df['ckm1'].mean() * 100
porcentaje_ckm2 = df['ckm2'].mean() * 100
porcentaje_ckm3 = df['ckm3'].mean() * 100
porcentaje_ckm4 = df['ckm4'].mean() * 100

# Creando subconjuntos donde cada ckm es 1
subdataset_ckm1 = df[df['ckm1'] == 1]
subdataset_ckm2 = df[df['ckm2'] == 1]
subdataset_ckm3 = df[df['ckm3'] == 1]
subdataset_ckm4 = df[df['ckm4'] == 1]

# Función para convertir DataFrame a CSV para descarga
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')


# Configurando Streamlit
st.title('Porcentaje de pacientes según CKM')

st.title('CKM-1')
st.write(f"Porcentaje de pacientes en 'CKM-1': {porcentaje_ckm1:.2f}%")
st.progress(porcentaje_ckm1 / 100)
st.write("Pacientes con IMC>=25 o Glucemia en ayunas>=100 o 5,7<=Hemoglobina glicosilada<=6,4")
st.write("Consejo dietético: Dieta hipocalórica moderada, controlada en glúcidos (baja en glúcidos de elevado índice glucémico, limitada en carbohidratos refinados y azúcares simples/añadidos; prioritaria en carbohidratos complejos), rica en fibra (30-40g/día) y en proteína de alto valor biológico (en torno al 15-20% VCT")
csv_ckm1 = convert_df_to_csv(subdataset_ckm1)
st.download_button(label="Descargar datos para ckm1", data=csv_ckm1, file_name='subdataset_ckm1.csv', mime='text/csv')

st.title('CKM-2')
st.write(f"Porcentaje de pacientes en 'CKM-2': {porcentaje_ckm2:.2f}%")
st.progress(porcentaje_ckm2 / 100)
st.write("Pacientes con hipertensión o hipertriglicidemia")
st.write("Consejo dietético: Dieta hipocalórica (déficit de 500-750 kcal/día; aproximadamente 1200-1500 kcal/día en mujeres y 1500-1800 kcal/día en hombres), controlada en grasas [25-35% VCT; <1% AGT, <7% AGS, <7-10% AGP, 15-20% AGM, <300mg/día colesterol] y en glúcidos [45-55% del VCT; baja en glúcidos de elevado índice glucémico, rica en carbohidratos complejos y en fibra (35-40g/día, en una porción insoluble/soluble 3:1)]. Además, deberá ser una dieta hiposódica (<2.400mg/día), exenta en alcohol y limitada en xantinas (cafeína, teofilina, teobromina, etc)")
csv_ckm2 = convert_df_to_csv(subdataset_ckm2)
st.download_button(label="Descargar datos para ckm2", data=csv_ckm2, file_name='subdataset_ckm2.csv', mime='text/csv')

st.title('CKM-3')
st.write(f"Porcentaje de pacientes en 'CKM-3': {porcentaje_ckm3:.2f}%")
st.progress(porcentaje_ckm3 / 100)
st.write("Pacientes con edad >=70")
st.write("Consejo dietético: Dieta hipocalórica (considerar dieta muy baja en calorías bajo supervisión: 800-1000 kcal/día) , con control estricto de grasas [25-35% del VCT; <1% AGT, <7% AGS, <7-10% AGP, 15-20% AGM, <300mg/día colesterol], controlada en proteínas (≈0,8 g/kg/día) y en glúcidos (45-50% del VCT, priorizando carbohidratos complejos y evitando aquellos de elevado índice glucémico), rica en fibra soluble e insoluble (35-40g/día) e hiposódica (<2.300mg/día). Exenta en alcohol y en xantinas.")
csv_ckm3 = convert_df_to_csv(subdataset_ckm3)
st.download_button(label="Descargar datos para ckm3", data=csv_ckm3, file_name='subdataset_ckm3.csv', mime='text/csv')

st.title('CKM-4')
st.write(f"Porcentaje de pacientes en 'CKM-4': {porcentaje_ckm4:.2f}%")
st.progress(porcentaje_ckm4 / 100)
st.write("Pacientes con edad <70")
st.write("Consejo dietético: Dieta hipocalórica, baja en grasas [25-30% del VCT, con un control más estricto en grasas saturadas (<5% del VCT) y preferencia por grasas monoinsaturadas y poliinsaturadas], controlada en proteínas (0,6-0,8g/kg/día) y en glúcidos [fibra: mantener una ingesta adecuada (25-30 g/día), pero limitar el consumo de alimentos ricos en potasio y fósforo que también contienen fibra (como ciertos granos enteros y frutas)]. Muy baja en sodio (<2.000 mg/día) para minimizar la retención de líquidos. Exenta en alcohol y en xantinas.")
csv_ckm4 = convert_df_to_csv(subdataset_ckm4)
st.download_button(label="Descargar datos para ckm4", data=csv_ckm4, file_name='subdataset_ckm4.csv', mime='text/csv')


# Lista de DataFrames para facilitar el procesamiento
datasets = [subdataset_ckm1, subdataset_ckm2, subdataset_ckm3, subdataset_ckm4]
dataset_names = ["ckm1", "ckm2", "ckm3", "ckm4"]

# Filtrar columnas excluyendo las que comienzan con ciertos prefijos
exclusion_prefixes = ["reglas", "Otras ECV", "Diagnósticos principal", "Diagnósticos asociados", "Medicamento", "Recomendaciones nutricionales recibidas", "Consejo dietético","Juicio clínico"]

# Función para excluir columnas basadas en prefijos
def filter_columns(df):
    return [col for col in df.columns if not any(col.startswith(prefix) for prefix in exclusion_prefixes)]

# Crear un diccionario para almacenar los resultados
results = {}

for i, df in enumerate(datasets):
    # Filtrar columnas
    columns = filter_columns(df)
    # Calcular promedio y desviación estándar
    avg = df[columns].mean()
    std = df[columns].std()
    
    # Añadir resultados al diccionario
    results[f"avg_{dataset_names[i]}"] = avg
    results[f"std_{dataset_names[i]}"] = std

# Convertir resultados a DataFrame
results_df = pd.DataFrame(results)

# Mostrar en Streamlit
st.title('Estadística')
st.write("Tabla de promedio y desviación estándar")
st.dataframe(results_df)

st.title("Subperfiles clínicos")
st.write("Las reglas que aparecen para cada clúster son las que comparten todos los pacientes del clúster. Es posible que individualmente algún paciente cumpla con otras reglas")
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
import streamlit as st

# Función para realizar el clustering jerárquico y devolver los clusters en texto
def hierarchical_clustering(data, max_d):
    # Filtrar columnas que comienzan con "reglas", "Otras ECV", etc.
    exclusion_prefixes = [
        "reglas", "Otras ECV", "Diagnósticos principal", 
        "Diagnósticos asociados", "Medicamento", 
        "Recomendaciones nutricionales recibidas", "Consejo dietético", "Juicio clínico"
    ]
    filtered_columns = [col for col in data.columns if not any(col.startswith(prefix) for prefix in exclusion_prefixes)]
    diagnosis_columns = [col for col in data.columns if col.startswith('reglas')]
    recommendation_columns = [col for col in data.columns if col.startswith('Recomendaciones nutricionales recibidas')]
    diet_advice_columns = [col for col in data.columns if col.startswith('Consejo dietético')]
    diagnosis_data = data[diagnosis_columns]
    stats_data = data[filtered_columns]

    # Clustering jerárquico
    linked = linkage(diagnosis_data, method='ward')

    # Asignar clusters basados en una distancia máxima
    labels = fcluster(linked, max_d, criterion='distance')
    
    # Agrupar los índices de pacientes por cluster
    cluster_dict = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_dict[label].append(data.index[idx])

    # Formatear los clusters para su visualización
    cluster_info = {}
    patient_details = {}
    for key, value in cluster_dict.items():
        # Modo de cada diagnóstico para los pacientes en el clúster, filtrar por modos que son 1
        cluster_diagnoses = diagnosis_data.loc[value].mode().iloc[0]
        diagnosis_list = [diag for diag, mode in cluster_diagnoses.iteritems() if mode == 1 and (diagnosis_data.loc[value][diag] == 1).all()]
        cluster_info[key] = {
            'Número de pacientes': len(value),
            'Diagnósticos comunes': diagnosis_list
        }
        patient_details[key] = data.loc[value]
    return cluster_info, patient_details, stats_data, cluster_dict, recommendation_columns, diet_advice_columns

# Interfaz de Streamlit
st.title('Análisis Jerárquico de Diagnósticos Médicos en Texto por Dataset')

# Barra deslizante para que el usuario seleccione la distancia máxima para el clustering
max_d = st.slider('Seleccione la distancia máxima para definir los clústers (cuanto mayor sea la distancia, menos clústers)', 1, 10, 5)

# Barra deslizante para que el usuario seleccione cuántas variables frecuentes mostrar
top_n = st.slider('Seleccione el número de recomendaciones y consejos dietéticos más frecuentes a mostrar', 1, 10, 5)

# Lista de subdatasets
datasets = {
    'ckm1': subdataset_ckm1,
    'ckm2': subdataset_ckm2,
    'ckm3': subdataset_ckm3,
    'ckm4': subdataset_ckm4
}

# Iterar sobre cada subdataset y realizar el clustering
for dataset_name, dataset in datasets.items():
    st.header(f'Análisis para {dataset_name}')
    
    # Realizar el clustering con la distancia seleccionada por el usuario
    clusters, patient_data, stats_data, cluster_dict, recommendation_columns, diet_advice_columns = hierarchical_clustering(dataset, max_d)

    # Ordenar los clusters por clave para que se muestren en orden
    sorted_clusters = dict(sorted(clusters.items()))

    # Mostrar los resultados del clustering en orden
    for cluster_id, info in sorted_clusters.items():
        st.subheader(f'Clúster {cluster_id}')
        st.write(f'Número de pacientes: {info["Número de pacientes"]}')
        st.write(f'Diagnósticos comunes: {", ".join(info["Diagnósticos comunes"]).replace("reglas ","")}')
        
        # Obtener datos del cluster actual
        cluster_data = stats_data.loc[cluster_dict[cluster_id]]

        # Calcular promedio y desviación estándar
        cluster_avg = cluster_data.mean()
        cluster_std = cluster_data.std()

        # Crear un DataFrame para mostrar los resultados estadísticos
        stats_summary = pd.DataFrame({
            'Promedio': cluster_avg,
            'Desviación Estándar': cluster_std
        })

        # Mostrar tabla estadística
        st.write("Tabla Estadística:")
        st.dataframe(stats_summary)

        # Filtrar datos para recomendaciones nutricionales
        recommendation_data = dataset.loc[cluster_dict[cluster_id], recommendation_columns]
        recommendation_counts = recommendation_data.sum().sort_values(ascending=False).head(top_n)
        recommendation_summary = pd.DataFrame({
            'Recomendación Nutricional': recommendation_counts.index,
            'Frecuencia': recommendation_counts.values
        })
        recommendation_summary['Recomendación Nutricional'] = recommendation_summary['Recomendación Nutricional'].str.replace("Recomendaciones nutricionales recibidas_", "", regex=False)


        # Mostrar tabla de recomendaciones nutricionales más frecuentes
        st.write("Recomendaciones Nutricionales Más Frecuentes:")
        st.dataframe(recommendation_summary)

        # Filtrar datos para consejos dietéticos
        diet_advice_data = dataset.loc[cluster_dict[cluster_id], diet_advice_columns]
        diet_advice_counts = diet_advice_data.sum().sort_values(ascending=False).head(top_n)
        diet_advice_summary = pd.DataFrame({
            'Consejo Dietético': diet_advice_counts.index,
            'Frecuencia': diet_advice_counts.values
        })
        diet_advice_summary['Consejo Dietético'] = diet_advice_summary['Consejo Dietético'].str.replace("Consejo dietético_","")

        # Mostrar tabla de consejos dietéticos más frecuentes
        st.write("Consejos Dietéticos Más Frecuentes:")
        st.dataframe(diet_advice_summary)

        # Botón para descargar datos de cada clúster
        csv = patient_data[cluster_id].to_csv(index=False)
        st.download_button(
            label=f"Descargar datos del Clúster {cluster_id} - {dataset_name}",
            data=csv,
            file_name=f'cluster_{cluster_id}_{dataset_name}.csv',
            mime='text/csv',
        )

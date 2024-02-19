# Generamos nuevas variables en función de las anteriores

import pandas as pd

def imc(df:pd.DataFrame,config) -> pd.DataFrame:
    imc=config['variables']['imc']['siglas dataset']
    peso=config['variables']['peso']['siglas dataset']
    talla=config['variables']['talla']['siglas dataset']
    df[imc]=df[peso]/(df[talla]*df[talla])
    return df

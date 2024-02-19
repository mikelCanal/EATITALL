
def convertir_si_no_a_0_1(df):
    columnas_si_no = []
    for col in df.columns:
        if df[col].dtype == 'object':  # Verificar si la columna es de tipo object (potencialmente string)
            valores_unicos = df[col].dropna().unique()  # Obtener valores únicos sin contar NaN
            if set(valores_unicos) <= {'Sí', 'No'}:  # Verificar si todos los valores son 'Sí' o 'No'
                columnas_si_no.append(col)

    for columna in columnas_si_no:
        df[columna] = df[columna].map({'Si': 1, 'No': 0})
    return df


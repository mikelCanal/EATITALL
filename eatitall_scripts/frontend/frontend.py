import sys

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Título de la aplicación
st.title('Dashboard de Visualización con Streamlit')

# Generar datos aleatorios
data = pd.DataFrame({
    'Fecha': pd.date_range(start='1/1/2022', periods=100),
    'Ventas': np.random.randint(100, 500, size=100)
})

# Mostrar datos en la aplicación
st.write("Datos de ventas generados al azar:")
st.write(data)

# Crear un gráfico de barras
st.subheader('Gráfico de barras de las ventas diarias')
fig, ax = plt.subplots()
ax.bar(data['Fecha'], data['Ventas'], color='blue')
ax.set_xlabel('Fecha')
ax.set_ylabel('Ventas')
ax.set_title('Ventas diarias')
plt.xticks(rotation=45)
st.pyplot(fig)

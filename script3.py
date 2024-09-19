import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar los datos desde el path específico
data = pd.read_excel("C:\\Users\\Daniel Bolaños\\Downloads\\Cuestionario de aerolínea (respuestas).xlsx")

# Selección de las columnas necesarias para los gráficos
df_cleaned = data[['Edad:', 'Ingresos Anuales:', 'Frecuencia de Viaje:']].copy()

# Renombrar columnas para un uso más sencillo
df_cleaned.columns = ['Edad', 'Ingresos', 'Frecuencia_Viaje']

# Eliminar filas con datos faltantes
df_cleaned.dropna(subset=['Edad', 'Ingresos', 'Frecuencia_Viaje'], inplace=True)

# 1. Gráfico de barras de la distribución de Edad
plt.figure(figsize=(8, 6))
df_cleaned['Edad'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Distribución de Edad')
plt.xlabel('Edad')
plt.ylabel('Cantidad de respuestas')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Gráfico de barras de la distribución de Ingresos
plt.figure(figsize=(8, 6))
df_cleaned['Ingresos'].value_counts().sort_index().plot(kind='bar', color='lightgreen')
plt.title('Distribución de Ingresos Anuales')
plt.xlabel('Ingresos')
plt.ylabel('Cantidad de respuestas')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Frecuencia de Viaje por Edad
plt.figure(figsize=(10, 6))
df_pivot_frecuencia_edad = df_cleaned.groupby(['Edad', 'Frecuencia_Viaje']).size().unstack().fillna(0)
df_pivot_frecuencia_edad.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
plt.title('Frecuencia de Viaje por Edad')
plt.xlabel('Edad')
plt.ylabel('Cantidad de respuestas')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Distribución de Edad por Ingresos Anuales
plt.figure(figsize=(10, 6))
df_pivot_edad_ingresos = df_cleaned.groupby(['Edad', 'Ingresos']).size().unstack().fillna(0)
df_pivot_edad_ingresos.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
plt.title('Distribución de Edad por Ingresos Anuales')
plt.xlabel('Edad')
plt.ylabel('Cantidad de respuestas')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

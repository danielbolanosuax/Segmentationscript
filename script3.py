import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar los datos
data = pd.read_excel("C:\\Users\\Daniel Bolaños\\Downloads\\Cuestionario de aerolínea (respuestas).xlsx")

# 2. Asignar nombres más cortos a las columnas para facilidad de manejo
data.rename(columns={
    'La seguridad es una prioridad fundamental al elegir una aerolínea.': 'Seguridad',
    'Prefiero una aerolínea que ofrezca información detallada sobre sus procedimientos de seguridad y pagaría más por ello': 'Proc_Seguridad',
    'El precio de los boletos es el factor más importante al elegir una aerolínea.': 'Precio',
    'Estoy dispuesto/a a pagar un poco más por una mejor experiencia de vuelo.': 'Pago_Mejora_Experiencia',
    'La puntualidad es crucial para mí cuando viajo en avión.': 'Puntualidad',
    'Prefiero una aerolínea con una buena reputación en cuanto a la gestión de retrasos.': 'Reputacion_Retrasos',
    'Es importante para mí que una aerolínea ofrezca vuelos frecuentes en mis rutas de viaje habituales.': 'Vuelos_Frecuentes',
    'Valoro la disponibilidad de múltiples opciones de horarios para mis vuelos.': 'Opciones_Horarios',
    'La comodidad del asiento y el espacio en el avión son muy importantes para mí.': 'Comodidad_Asiento',
    'Estoy interesado/a en servicios adicionales que mejoren la comodidad durante el vuelo, como entretenimiento a bordo o espacio adicional para las piernas.': 'Servicios_Adicionales',
    'La calidad y variedad de la comida ofrecida en el avión influyen en mi elección de aerolínea.': 'Calidad_Comida',
    'Prefiero aerolíneas que ofrecen opciones de comida especiales o dietéticas.': 'Comida_Especial',
    'Una plataforma de reserva fácil de usar es esencial para mí al elegir una aerolínea.': 'Reserva_Facil',
    'Valoro la posibilidad de hacer cambios en mi reserva sin complicaciones.': 'Cambios_Reserva'
}, inplace=True)

# 3. Seleccionar las columnas relevantes para clustering (con nombres más cortos)
columnas_clustering = [
    'Seguridad', 'Proc_Seguridad', 'Precio', 'Pago_Mejora_Experiencia',
    'Puntualidad', 'Reputacion_Retrasos', 'Vuelos_Frecuentes', 'Opciones_Horarios',
    'Comodidad_Asiento', 'Servicios_Adicionales', 'Calidad_Comida',
    'Comida_Especial', 'Reserva_Facil', 'Cambios_Reserva'
]

# 4. Filtrar los datos
df_clustering = data[columnas_clustering]

# 5. Estandarizar los datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clustering)

# 6. Aplicar PCA para reducir la dimensionalidad
pca = PCA(n_components=2)  # Mantener solo las dos primeras componentes principales para visualización
df_pca = pca.fit_transform(df_scaled)

# 7. Ver la varianza explicada por cada componente principal
print(f'Varianza explicada por cada componente: {pca.explained_variance_ratio_}')
print(f'Varianza explicada total: {pca.explained_variance_ratio_.sum()}')

# 8. Aplicar K-Means en los datos reducidos por PCA
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(df_pca)

# 9. Visualización de clusters en las primeras dos componentes principales
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=data['Cluster'], palette='Set1', s=100)
plt.title('Clusters basados en preferencias de aerolínea (PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')

# Mejorar la leyenda
plt.legend(title='Segmentos de Clientes', loc='upper right', title_fontsize='13', fontsize='10')
plt.show()

# Gráfico de barras para comparar la media de cada variable por cluster
cluster_means = data.groupby('Cluster')[columnas_clustering].mean()
cluster_means.T.plot(kind='bar', figsize=(12, 8))
plt.title('Media de respuestas por Cluster')
plt.xlabel('Preguntas')
plt.ylabel('Valor Promedio')
plt.legend(title='Cluster')
plt.show()

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar los datos
data = pd.read_excel("C:\\Users\\Daniel Bolaños\\Downloads\\Cuestionario de aerolínea (respuestas).xlsx")

# Seleccionar las columnas correspondientes a los 7 atributos para el clustering
columns_to_cluster = [
    'La seguridad es una prioridad fundamental al elegir una aerolínea.',  # Safety
    'El precio de los boletos es el factor más importante al elegir una aerolínea.',  # Price
    'La puntualidad es crucial para mí cuando viajo en avión.',  # On-time
    'Es importante para mí que una aerolínea ofrezca vuelos frecuentes en mis rutas de viaje habituales.',  # Frequency
    'La comodidad del asiento y el espacio en el avión son muy importantes para mí.',  # Comfort
    'La calidad y variedad de la comida ofrecida en el avión influyen en mi elección de aerolínea.',  # Food
    'Una plataforma de reserva fácil de usar es esencial para mí al elegir una aerolínea.'  # Ease of reservation
]

# Seleccionar los datos para el clustering
X = data[columns_to_cluster]

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-Means con 3 clusters (puedes cambiar el número si lo deseas)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# --- Mejora de los gráficos para profesionalidad ---

# Definir colores para los clusters
colores_clusters = ['#1f77b4', '#ff7f0e', '#2ca02c']

# --- GRÁFICOS DE DISPERSIÓN PARA CADA ATRIBUTO VS CLUSTERS ---
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(16, 18))
axs = axs.flatten()

# Generar gráficos de dispersión
for i, column in enumerate(columns_to_cluster):
    scatter = axs[i].scatter(data[column], data['Cluster'], 
                             c=data['Cluster'], cmap='viridis', alpha=0.7, s=50)
    axs[i].set_title(f'Distribución de Clusters: {column}', fontsize=10)
    axs[i].set_xlabel(column[:30] + '...')  # Truncar las etiquetas de los ejes si son largas
    axs[i].set_ylabel('Clusters')
    axs[i].grid(True)
    axs[i].autoscale(enable=True, axis='both')  # Ajustar límites automáticamente

# Eliminar gráficos vacíos si no se usan todos los subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

# Añadir una barra de colores para los clusters
fig.colorbar(scatter, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)

plt.tight_layout()
plt.show()

# --- GRÁFICO DE 4 CUADRANTES: SAFETY VS PRICE ---
plt.figure(figsize=(8, 6))

# Definir variables para los ejes
safety_column = 'La seguridad es una prioridad fundamental al elegir una aerolínea.'
price_column = 'El precio de los boletos es el factor más importante al elegir una aerolínea.'

# Graficar los puntos en el scatter plot
scatter = plt.scatter(
    data[price_column],
    data[safety_column],
    c=data['Cluster'], cmap='viridis', alpha=0.7, s=100
)

# Títulos y etiquetas de los ejes
plt.title('Clusters basados en Safety y Price', fontsize=14)
plt.xlabel('Importancia del Precio', fontsize=12)
plt.ylabel('Importancia de la Seguridad', fontsize=12)

# Calcular las medias de Safety y Price
price_mean = data[price_column].mean()
safety_mean = data[safety_column].mean()

# Dibujar líneas de la media para dividir en cuadrantes
plt.axhline(y=safety_mean, color='black', linestyle='--', label='Media Safety')  # Línea horizontal
plt.axvline(x=price_mean, color='black', linestyle='--', label='Media Price')  # Línea vertical

# Ajustar límites del gráfico
plt.xlim(data[price_column].min() - 0.5, data[price_column].max() + 0.5)
plt.ylim(data[safety_column].min() - 0.5, data[safety_column].max() + 0.5)

# Mostrar leyenda y grid
plt.legend()
plt.grid(True)

# Añadir la barra de colores para los clusters
plt.colorbar(scatter)

# Mostrar el gráfico
plt.show()

# Guardar los resultados del clustering en un archivo Excel
output_path = r"C:\Users\Daniel Bolaños\Downloads\Clustering_resultados_profesionales.xlsx"
data.to_excel(output_path, index=False)

print(f"Archivo guardado en: {output_path}")

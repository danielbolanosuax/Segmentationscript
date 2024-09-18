import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Cargar los datos
data = pd.read_excel("C:\\Users\\Daniel Bolaños\\Downloads\\Cuestionario de aerolínea (respuestas).xlsx")

print(data.columns)

# Seleccionar las columnas con los nombres originales del archivo
columns_to_cluster = [
    'La seguridad es una prioridad fundamental al elegir una aerolínea.',  # Safety
    'El precio de los boletos es el factor más importante al elegir una aerolínea.',  # Price
    'La puntualidad es crucial para mí cuando viajo en avión.',  # On-time
    'Es importante para mí que una aerolínea ofrezca vuelos frecuentes en mis rutas de viaje habituales.',  # Frequency
    'La comodidad del asiento y el espacio en el avión son muy importantes para mí.',  # Comfort
    'La calidad y variedad de la comida ofrecida en el avión influyen en mi elección de aerolínea.',  # Food
    'Una plataforma de reserva fácil de usar es esencial para mí al elegir una aerolínea.'  # Ease of reservation
]

# Abreviaciones para los gráficos (nombres cortos)
short_column_names = [
    'Seguridad', 'Precio', 'Puntualidad', 'Frecuencia', 'Comodidad', 'Comida', 'Facilidad de Reserva'
]

# --- Eliminación de Filas con NaN ---
# Antes de realizar cualquier procesamiento, eliminamos las filas que contengan valores NaN en las columnas de interés
data_clean = data.dropna(subset=columns_to_cluster)

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_clean[columns_to_cluster])

# Aplicar PCA para reducir a 2 dimensiones y facilitar la visualización
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Obtener las "cargas" de cada atributo en los componentes principales
loadings = pca.components_

# Determinar los atributos más influyentes en cada componente
top_feature_indices_cp1 = abs(loadings[0]).argsort()[::-1][:2]  # Las dos variables más importantes del CP1
top_feature_indices_cp2 = abs(loadings[1]).argsort()[::-1][:2]  # Las dos variables más importantes del CP2

# Atributos más influyentes en el componente principal 1 y 2 (usando los nombres cortos para las etiquetas)
top_features_cp1 = [short_column_names[i] for i in top_feature_indices_cp1]
top_features_cp2 = [short_column_names[i] for i in top_feature_indices_cp2]

# Aplicar K-Means con 3 clusters (puedes cambiar el número si lo deseas)
kmeans = KMeans(n_clusters=3, random_state=42)
data_clean['Cluster'] = kmeans.fit_predict(X_scaled)

# --- PRIMER GRÁFICO: DISPERSIÓN PCA ---
fig1 = px.scatter(
    x=X_pca[:, 0], y=X_pca[:, 1], color=data_clean['Cluster'].astype(str),
    title='Clusters proyectados en PCA',
    labels={'x': f'CP1: {top_features_cp1[0]} / {top_features_cp1[1]}',
            'y': f'CP2: {top_features_cp2[0]} / {top_features_cp2[1]}'},
    color_discrete_sequence=px.colors.qualitative.Vivid,
    template='plotly_white'
)

# Personalizar el diseño
fig1.update_layout(
    title_font=dict(size=20, family='Arial'),
    xaxis=dict(title_font=dict(size=16)),
    yaxis=dict(title_font=dict(size=16)),
    legend_title_text='Cluster'
)

# Mostrar el gráfico
fig1.show()

# --- SEGUNDO GRÁFICO: PCA PROYECTADO CON MARCAS DE AGUA PARA CUADRANTES ---
fig2 = go.Figure()

# Crear gráfico de dispersión
fig2.add_trace(go.Scatter(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    mode='markers',
    marker=dict(color=data_clean['Cluster'], colorscale='Plasma', size=12, opacity=0.8),
    text=data_clean['Cluster'],
    hoverinfo='text'
))

# Títulos y etiquetas
fig2.update_layout(
    title='Clusters en espacio de componentes principales',
    xaxis_title=f'CP1: {top_features_cp1[0]} / {top_features_cp1[1]}',
    yaxis_title=f'CP2: {top_features_cp2[0]} / {top_features_cp2[1]}',
    template='plotly_white'
)

# Añadir líneas de la media para dividir en cuadrantes
cp1_mean = X_pca[:, 0].mean()
cp2_mean = X_pca[:, 1].mean()

fig2.add_shape(type="line", x0=cp1_mean, x1=cp1_mean, y0=min(X_pca[:, 1]), y1=max(X_pca[:, 1]),
               line=dict(color="Black", width=2, dash="dash"))
fig2.add_shape(type="line", x0=min(X_pca[:, 0]), x1=max(X_pca[:, 0]), y0=cp2_mean, y1=cp2_mean,
               line=dict(color="Black", width=2, dash="dash"))

# --- Añadir marcas de agua en cada cuadrante para describir los clusters ---
fig2.add_annotation(
    text="Alto Precio, Alta Comodidad", showarrow=False,
    x=cp1_mean + (X_pca[:, 0].max() - cp1_mean) / 2,  # Parte superior derecha
    y=cp2_mean + (X_pca[:, 1].max() - cp2_mean) / 2,
    font=dict(size=14, color="LightGrey")
)

fig2.add_annotation(
    text="Bajo Precio, Alta Comodidad", showarrow=False,
    x=cp1_mean - (cp1_mean - X_pca[:, 0].min()) / 2,  # Parte superior izquierda
    y=cp2_mean + (X_pca[:, 1].max() - cp2_mean) / 2,
    font=dict(size=14, color="LightGrey")
)

fig2.add_annotation(
    text="Alto Precio, Baja Comodidad", showarrow=False,
    x=cp1_mean + (X_pca[:, 0].max() - cp1_mean) / 2,  # Parte inferior derecha
    y=cp2_mean - (cp2_mean - X_pca[:, 1].min()) / 2,
    font=dict(size=14, color="LightGrey")
)

fig2.add_annotation(
    text="Bajo Precio, Baja Comodidad", showarrow=False,
    x=cp1_mean - (cp1_mean - X_pca[:, 0].min()) / 2,  # Parte inferior izquierda
    y=cp2_mean - (cp2_mean - X_pca[:, 1].min()) / 2,
    font=dict(size=14, color="LightGrey")
)

# Mostrar el gráfico
fig2.show()

# --- Tratamiento de la columna 'Ingresos Anuales:' ---
# Mapeo de rangos de ingresos anuales a valores numéricos (usamos el valor promedio de cada rango)
ingresos_mapping = {
    'Menos de $20,000': 10000,
    'De $20,000 a $50,000': 35000,
    'De $50,000 a $100,000': 75000,
    'Más de $100,000': 100000
}

# Convertir la columna de ingresos a valores numéricos
data_clean['Ingresos Anuales (Num)'] = data_clean['Ingresos Anuales:'].map(ingresos_mapping)

# --- Mapeo de preferencia de clase de vuelo a valores numéricos ---
clase_mapping = {
    'Economy': 1,
    'Premium Economy': 2,
    'Business': 3,
    'First Class': 4
}

# Convertir la columna de preferencia de clase de vuelo a numérico
data_clean['Preferencia de Clase de Vuelo (Num)'] = data_clean['Preferencia de Clase de Vuelo:'].map(clase_mapping)

# Guardar los resultados del clustering en un archivo Excel
output_path = r"C:\Users\Daniel Bolaños\Downloads\Clustering_resultados_profesionales_pca.xlsx"
data_clean.to_excel(output_path, index=False)

print(f"Archivo guardado en: {output_path}")

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Cargar los datos
data = pd.read_excel("C:\\Users\\Daniel Bolaños\\Downloads\\Cuestionario de aerolínea (respuestas).xlsx")

# Mapeo de valores de "Frecuencia de Viaje" a valores numéricos
frecuencia_viaje_mapping = {
    'Raramente (menos de una vez al año)': 1,
    'Ocasionalmente (una o dos veces al año)': 2,
    'Frecuentemente (más de dos veces al año)': 3
}
data['Frecuencia de Viaje (Num)'] = data['Frecuencia de Viaje:'].map(frecuencia_viaje_mapping)

# Convertir 'Preferencia de Clase de Vuelo:' en numérico con LabelEncoder
label_encoder = LabelEncoder()
data['Preferencia de Clase de Vuelo (Num)'] = label_encoder.fit_transform(data['Preferencia de Clase de Vuelo:'])

# --- Función para ejecutar clustering con PCA y visualización ---
def ejecutar_clustering_pca(columnas, nombres_cortos, n_clusters=3):
    # --- Eliminación de Filas con NaN ---
    data_clean = data.dropna(subset=columnas)

    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_clean[columnas])

    # Aplicar PCA para reducir a 2 dimensiones y facilitar la visualización
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Obtener las "cargas" de cada atributo en los componentes principales
    loadings = pca.components_

    # Determinar los atributos más influyentes en cada componente
    top_feature_indices_cp1 = abs(loadings[0]).argsort()[::-1][:2]  # Las dos variables más importantes del CP1
    top_feature_indices_cp2 = abs(loadings[1]).argsort()[::-1][:2]  # Las dos variables más importantes del CP2

    # Atributos más influyentes en el componente principal 1 y 2
    top_features_cp1 = [nombres_cortos[i] for i in top_feature_indices_cp1]
    top_features_cp2 = [nombres_cortos[i] for i in top_feature_indices_cp2]

    # Aplicar K-Means con n_clusters clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data_clean['Cluster'] = kmeans.fit_predict(X_scaled)

    # --- PRIMER GRÁFICO: DISPERSIÓN PCA ---
    fig1 = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1], color=data_clean['Cluster'].astype(str),
        title=f'Clusters basados en {", ".join(nombres_cortos)} (PCA)',
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
        title=f'Clusters en espacio de componentes principales ({", ".join(nombres_cortos)})',
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

    # Añadir marcas de agua en los cuadrantes
    fig2.add_annotation(
        text=f"Alto {top_features_cp1[0]}, Alto {top_features_cp2[0]}", showarrow=False,
        x=cp1_mean + (X_pca[:, 0].max() - cp1_mean) / 2,
        y=cp2_mean + (X_pca[:, 1].max() - cp2_mean) / 2,
        font=dict(size=14, color="LightGrey")
    )

    fig2.add_annotation(
        text=f"Bajo {top_features_cp1[0]}, Alto {top_features_cp2[0]}", showarrow=False,
        x=cp1_mean - (cp1_mean - X_pca[:, 0].min()) / 2,
        y=cp2_mean + (X_pca[:, 1].max() - cp2_mean) / 2,
        font=dict(size=14, color="LightGrey")
    )

    fig2.add_annotation(
        text=f"Alto {top_features_cp1[0]}, Bajo {top_features_cp2[0]}", showarrow=False,
        x=cp1_mean + (X_pca[:, 0].max() - cp1_mean) / 2,
        y=cp2_mean - (cp2_mean - X_pca[:, 1].min()) / 2,
        font=dict(size=14, color="LightGrey")
    )

    fig2.add_annotation(
        text=f"Bajo {top_features_cp1[0]}, Bajo {top_features_cp2[0]}", showarrow=False,
        x=cp1_mean - (cp1_mean - X_pca[:, 0].min()) / 2,
        y=cp2_mean - (cp2_mean - X_pca[:, 1].min()) / 2,
        font=dict(size=14, color="LightGrey")
    )

    # Mostrar el gráfico final con marcas de agua
    fig2.show()

# --- Definir los clusterings con los nombres correctos ---
clusterings = {
    'Cluster 1': (
        ['El precio de los boletos es el factor más importante al elegir una aerolínea.',
         'La comodidad del asiento y el espacio en el avión son muy importantes para mí.',
         'Una plataforma de reserva fácil de usar es esencial para mí al elegir una aerolínea.'],
        ['Precio', 'Comodidad', 'Fac.Reserva']
    ),
    'Cluster 2': (
        ['La seguridad es una prioridad fundamental al elegir una aerolínea.',
         'La puntualidad es crucial para mí cuando viajo en avión.',
         'Prefiero una aerolínea con una buena reputación en cuanto a la gestión de retrasos.'],
        ['Seguridad', 'Puntualidad', 'Gest.Retrasos']
    ),
    'Cluster 3': (
        ['Viajo principalmente por negocios.', 'Frecuencia de Viaje (Num)', 'Preferencia de Clase de Vuelo (Num)'],
        ['Tipo Viaje', 'Frecuencia', 'Pref.Clase']
    ),
    'Cluster 4': (
        ['La seguridad es una prioridad fundamental al elegir una aerolínea.',
         'El precio de los boletos es el factor más importante al elegir una aerolínea.',
         'Una plataforma de reserva fácil de usar es esencial para mí al elegir una aerolínea.'],
        ['Seguridad', 'Precio', 'Fac.Reserva']
    ),
    'Cluster 5': (
        ['El precio de los boletos es el factor más importante al elegir una aerolínea.',
         'La comodidad del asiento y el espacio en el avión son muy importantes para mí.',
         'La calidad y variedad de la comida ofrecida en el avión influyen en mi elección de aerolínea.'],
        ['Precio', 'Comodidad', 'Comida']
    )
}

# --- Ejecutar los cinco clusterings ---
for cluster_name, (variables, short_names) in clusterings.items():
    print(f"\nEjecutando {cluster_name}...")
    ejecutar_clustering_pca(variables, short_names)

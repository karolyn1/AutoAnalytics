import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import zscore

# =========================
# Funciones auxiliares
# =========================
def detectar_tipos(df):
    return {
        "numericas": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categoricas": df.select_dtypes(include=["object"]).columns.tolist(),
        "temporales": df.select_dtypes(include=["datetime64"]).columns.tolist(),
        "booleanas": df.select_dtypes(include=["bool"]).columns.tolist()
    }

def estadisticas_basicas(df, columnas_numericas):
    return df[columnas_numericas].describe()

def matriz_correlacion(df, columnas_numericas):
    corr = df[columnas_numericas].corr(method="pearson")
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

def outliers_zscore(df, columna):
    df["zscore"] = zscore(df[columna])
    return df[df["zscore"].abs() > 3]

def outliers_iqr(df, columna):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[columna] < Q1 - 1.5*IQR) | (df[columna] > Q3 + 1.5*IQR)]

def outliers_isolation_forest(df, columnas_numericas):
    iso = IsolationForest(contamination=0.1, random_state=42)
    df["anomaly"] = iso.fit_predict(df[columnas_numericas])
    return df[df["anomaly"] == -1]

def clustering_kmeans(df, columnas_numericas, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster_kmeans"] = kmeans.fit_predict(df[columnas_numericas])
    return df

def clustering_dbscan(df, columnas_numericas, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df["cluster_dbscan"] = dbscan.fit_predict(df[columnas_numericas])
    return df

# =========================
# Interfaz Streamlit
# =========================
st.set_page_config(page_title="Análisis Inteligente de Datos", layout="wide")

st.sidebar.title("⚙️ Opciones")
archivo = st.sidebar.file_uploader("Carga tu archivo CSV o Excel", type=["csv","xlsx"])

st.title("📊 Dashboard de Análisis Automatizado")

if archivo:
    if archivo.name.endswith(".csv"):
        df = pd.read_csv(archivo)
    else:
        df = pd.read_excel(archivo)

    st.write("### 👀 Vista previa de los datos")
    st.dataframe(df.head())

    tipos = detectar_tipos(df)
    st.sidebar.write("Tipos de variables detectadas:", tipos)

    # Tabs para organizar resultados
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Estadísticas", "🔗 Correlación", "🚨 Outliers", "🧩 Clustering", "📊 Visualizaciones"])

    with tab1:
        if tipos["numericas"]:
            st.write("#### Estadísticas descriptivas")
            st.write(estadisticas_basicas(df, tipos["numericas"]))

    with tab2:
        if tipos["numericas"]:
            st.write("#### Matriz de correlación")
            matriz_correlacion(df, tipos["numericas"])

    with tab3:
        if tipos["numericas"]:
            columna_outlier = st.selectbox("Selecciona columna para outliers", tipos["numericas"])
            st.write("##### Outliers Z-score")
            st.dataframe(outliers_zscore(df, columna_outlier))
            st.write("##### Outliers IQR")
            st.dataframe(outliers_iqr(df, columna_outlier))
            st.write("##### Outliers Isolation Forest")
            st.dataframe(outliers_isolation_forest(df, tipos["numericas"]))

    with tab4:
        if tipos["numericas"]:
            n_clusters = st.slider("Número de clusters (KMeans)", 2, 10, 3)
            df = clustering_kmeans(df, tipos["numericas"], n_clusters)
            df = clustering_dbscan(df, tipos["numericas"])
            st.write("#### Resultados con clustering")
            st.dataframe(df.head())
            st.download_button("⬇️ Descargar resultados en CSV", df.to_csv(index=False).encode("utf-8"), "resultados.csv", "text/csv")

    with tab5:
        if tipos["numericas"]:
            st.write("#### Histogramas interactivos")
            for col in tipos["numericas"]:
                fig = px.histogram(df, x=col, nbins=20, title=f"Histograma de {col}")
                st.plotly_chart(fig, use_container_width=True)

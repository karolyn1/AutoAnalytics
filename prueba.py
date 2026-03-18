import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

def graficos_basicos(df, columnas_numericas):
    for col in columnas_numericas:
        fig, axes = plt.subplots(1,2, figsize=(10,4))
        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f"Histograma de {col}")
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f"Boxplot de {col}")
        st.pyplot(fig)

# =========================
# Interfaz Streamlit
# =========================
st.title("📊 Análisis Automatizado de Datos")

archivo = st.file_uploader("Carga tu archivo CSV o Excel", type=["csv","xlsx"])

if archivo:
    if archivo.name.endswith(".csv"):
        df = pd.read_csv(archivo)
    else:
        df = pd.read_excel(archivo)

    st.write("### Vista previa de los datos")
    st.dataframe(df.head())

    tipos = detectar_tipos(df)
    st.write("### Tipos de variables detectadas:", tipos)

    # Estadísticas básicas
    if tipos["numericas"]:
        st.write("### Estadísticas descriptivas")
        st.write(estadisticas_basicas(df, tipos["numericas"]))

        # Correlación
        st.write("### Matriz de correlación")
        matriz_correlacion(df, tipos["numericas"])

        # Outliers
        columna_outlier = st.selectbox("Selecciona columna para detección de outliers", tipos["numericas"])
        st.write("#### Outliers Z-score")
        st.dataframe(outliers_zscore(df, columna_outlier))
        st.write("#### Outliers IQR")
        st.dataframe(outliers_iqr(df, columna_outlier))
        st.write("#### Outliers Isolation Forest")
        st.dataframe(outliers_isolation_forest(df, tipos["numericas"]))

        # Clustering
        st.write("### Clustering automático")
        n_clusters = st.slider("Número de clusters (KMeans)", 2, 10, 3)
        df = clustering_kmeans(df, tipos["numericas"], n_clusters)
        df = clustering_dbscan(df, tipos["numericas"])
        st.dataframe(df.head())

        # Visualizaciones
        st.write("### Visualizaciones básicas")
        graficos_basicos(df, tipos["numericas"])

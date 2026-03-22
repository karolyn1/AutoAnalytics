import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io
import base64

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import zscore
from matplotlib.backends.backend_pdf import PdfPages

# =========================
# Configuración de página
# =========================
st.set_page_config(page_title="Análisis Inteligente de Datos", layout="wide")

# Estilos visuales y carga de Font Awesome
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">

<style>
.main-title{
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 12px;
}
.section-title{
    font-size: 24px;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 12px;
}
.sub-title{
    font-size: 18px;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 8px;
}
.sidebar-title{
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Funciones auxiliares
# =========================
def detectar_tipos(df):
    return {
        "numericas": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categoricas": df.select_dtypes(include=["object"]).columns.tolist(),
        "texto": df.select_dtypes(include=["string"]).columns.tolist(),
        "temporales": df.select_dtypes(include=["datetime64"]).columns.tolist(),
        "booleanas": df.select_dtypes(include=["bool"]).columns.tolist()
    }

def estadisticas_basicas(df, columnas_numericas):
    # Usa Pandas para generar estadísticas descriptivas
    return df[columnas_numericas].describe()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def matriz_correlacion_fig(df, columnas_numericas, figsize=(20,8), triangular=True, cmap="coolwarm"):

    # Subconjunto y copia para no modificar df original
    datos = df[columnas_numericas].copy()

    # Imputar NaN solo en columnas numéricas (con la media)
    datos = datos.apply(lambda s: s.fillna(s.mean()), axis=0)

    # Calcular ambas correlaciones
    corr_pearson = datos.corr(method="pearson")
    corr_spearman = datos.corr(method="spearman")

    # Máscara para mostrar solo triángulo inferior, si se desea, esto es para evitar valores duplicados
    mask = None
    if triangular:
        mask = np.triu(np.ones_like(corr_pearson, dtype=bool))

    # Crear figura con dos subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Heatmap Pearson
    sns.heatmap(
        corr_pearson, mask=mask, annot=True, fmt=".2f",
        cmap=cmap, vmin=-1, vmax=1, center=0,
        square=True, cbar_kws={"shrink": 0.8}, ax=axes[0]
    )
    axes[0].set_title("Matriz de correlación (Pearson)")

    # Heatmap Spearman
    sns.heatmap(
        corr_spearman, mask=mask, annot=True, fmt=".2f",
        cmap=cmap, vmin=-1, vmax=1, center=0,
        square=True, cbar_kws={"shrink": 0.8}, ax=axes[1]
    )
    axes[1].set_title("Matriz de correlación (Spearman)")

    plt.tight_layout()
    return fig, corr_pearson, corr_spearman

def outliers_zscore(df, columna):
    df_temp = df.copy()
    df_temp[columna] = df_temp[columna].fillna(df_temp[columna].mean())

    # Usa SciPy para calcular Z-score
    df_temp["zscore"] = zscore(df_temp[columna])
    return df_temp[df_temp["zscore"].abs() > 3]

def outliers_iqr(df, columna):
    df_temp = df.copy()
    df_temp[columna] = df_temp[columna].fillna(df_temp[columna].mean())

    # Usa Pandas y NumPy para cálculo por rango intercuartílico
    Q1 = df_temp[columna].quantile(0.25)
    Q3 = df_temp[columna].quantile(0.75)
    IQR = Q3 - Q1

    return df_temp[
        (df_temp[columna] < Q1 - 1.5 * IQR) |
        (df_temp[columna] > Q3 + 1.5 * IQR)
    ]

def outliers_isolation_forest(df, columnas_numericas):
    df_temp = df.copy()
    datos = df_temp[columnas_numericas].copy()
    datos = datos.fillna(datos.mean())

    # Usa Scikit-learn con IsolationForest
    iso = IsolationForest(contamination=0.1, random_state=42)
    df_temp["anomaly"] = iso.fit_predict(datos)
    return df_temp[df_temp["anomaly"] == -1]

def clustering_kmeans(df, columnas_numericas, n_clusters=3):
    df_temp = df.copy()
    datos = df_temp[columnas_numericas].copy()
    datos = datos.fillna(datos.mean())

    # Usa Scikit-learn con KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_temp["cluster_kmeans"] = kmeans.fit_predict(datos)
    return df_temp

def clustering_dbscan(df, columnas_numericas, eps=0.5, min_samples=5):
    df_temp = df.copy()
    datos = df_temp[columnas_numericas].copy()
    datos = datos.fillna(datos.mean())

    # Usa Scikit-learn con DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df_temp["cluster_dbscan"] = dbscan.fit_predict(datos)
    return df_temp

def fig_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)

    # Usa io y base64 para convertir imágenes a HTML
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def fig_to_pdf_bytes(figs):
    buffer = io.BytesIO()

    # Usa Matplotlib PdfPages para crear PDF
    with PdfPages(buffer) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")

    buffer.seek(0)
    return buffer.getvalue()

def tabla_a_figura(df, titulo="Tabla"):
    df = df.copy()
    fig, ax = plt.subplots(figsize=(12, min(0.5 * len(df) + 2, 12)))
    ax.axis("off")
    ax.set_title(titulo, fontsize=12, pad=12)

    # Usa Matplotlib para convertir tablas en figuras exportables a PDF
    tabla = ax.table(
        cellText=df.astype(str).values,
        colLabels=df.columns,
        loc="center"
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8)
    tabla.scale(1, 1.2)
    return fig

def tabla_serie_a_figura(df, titulo="Tabla"):
    df_reset = df.reset_index()
    return tabla_a_figura(df_reset, titulo)

def histograma_matplotlib(df, columna):
    datos = df[columna].dropna()

    # Usa Matplotlib para histogramas que se exportan a PDF
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(datos, bins=20)
    ax.set_title(f"Histograma de {columna}")
    ax.set_xlabel(columna)
    ax.set_ylabel("Frecuencia")
    return fig

def generar_html_estadisticas(stats_df):
    # Usa HTML generado con Pandas
    return f"""
    <html>
    <head><meta charset="utf-8"><title>Estadísticas</title></head>
    <body>
        <h1>Estadísticas descriptivas</h1>
        {stats_df.to_html(border=1)}
    </body>
    </html>
    """.encode("utf-8")

def generar_pdf_estadisticas(stats_df):
    fig = tabla_serie_a_figura(stats_df.round(2), "Estadísticas descriptivas")
    return fig_to_pdf_bytes([fig])

def generar_html_correlacion(corr_df, corr_fig):
    img = fig_to_base64(corr_fig)

    # Usa HTML + imagen convertida en base64
    return f"""
    <html>
    <head><meta charset="utf-8"><title>Correlación</title></head>
    <body>
        <h1>Matriz de correlación</h1>
        <img src="data:image/png;base64,{img}" style="max-width:100%;"><br><br>
        {corr_df.to_html(border=1)}
    </body>
    </html>
    """.encode("utf-8")

def generar_pdf_correlacion(corr_df, corr_fig):
    fig_tabla = tabla_a_figura(corr_df.round(2), "Tabla de correlación")
    return fig_to_pdf_bytes([corr_fig, fig_tabla])

def generar_html_outliers(z_df, iqr_df, iso_df, columna):
    # Usa HTML + tablas de Pandas
    return f"""
    <html>
    <head><meta charset="utf-8"><title>Outliers</title></head>
    <body>
        <h1>Detección de outliers</h1>
        <h2>Columna seleccionada: {columna}</h2>
        <h3>Outliers Z-score</h3>
        {z_df.to_html(border=1, index=False)}
        <h3>Outliers IQR</h3>
        {iqr_df.to_html(border=1, index=False)}
        <h3>Outliers Isolation Forest</h3>
        {iso_df.to_html(border=1, index=False)}
    </body>
    </html>
    """.encode("utf-8")

def generar_pdf_outliers(z_df, iqr_df, iso_df):
    figs = []
    if not z_df.empty:
        figs.append(tabla_a_figura(z_df.head(20), "Outliers Z-score"))
    if not iqr_df.empty:
        figs.append(tabla_a_figura(iqr_df.head(20), "Outliers IQR"))
    if not iso_df.empty:
        figs.append(tabla_a_figura(iso_df.head(20), "Outliers Isolation Forest"))
    if not figs:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No se encontraron outliers", ha="center", va="center", fontsize=14)
        figs.append(fig)
    return fig_to_pdf_bytes(figs)

def generar_html_clustering(df_cluster):
    # Usa HTML + tabla de Pandas
    return f"""
    <html>
    <head><meta charset="utf-8"><title>Clustering</title></head>
    <body>
        <h1>Resultados de clustering</h1>
        {df_cluster.to_html(border=1, index=False)}
    </body>
    </html>
    """.encode("utf-8")

def generar_pdf_clustering(df_cluster):
    fig = tabla_a_figura(df_cluster.head(30), "Resultados de clustering")
    return fig_to_pdf_bytes([fig])

def generar_html_visualizaciones(df, columnas_numericas):
    partes = ["""
    <html>
    <head><meta charset="utf-8"><title>Visualizaciones</title></head>
    <body>
        <h1>Histogramas</h1>
    """]

    for col in columnas_numericas:
        fig = histograma_matplotlib(df, col)
        img = fig_to_base64(fig)
        partes.append(f"<h2>Histograma de {col}</h2><img src='data:image/png;base64,{img}' style='max-width:100%;'><br><br>")
        plt.close(fig)

    partes.append("</body></html>")
    return "".join(partes).encode("utf-8")

def generar_pdf_visualizaciones(df, columnas_numericas):
    figs = []
    for col in columnas_numericas:
        figs.append(histograma_matplotlib(df, col))
    return fig_to_pdf_bytes(figs)

# =========================
# Sidebar
# =========================
st.sidebar.markdown(
    '<div class="sidebar-title"><i class="fa-solid fa-sliders"></i> Opciones</div>',
    unsafe_allow_html=True
)

archivo = st.sidebar.file_uploader("Carga tu archivo CSV o Excel", type=["csv", "xlsx"])

# =========================
# Título principal
# =========================
st.markdown(
    '<div class="main-title"><i class="fa-solid fa-chart-line"></i> Dashboard de Análisis Automatizado</div>',
    unsafe_allow_html=True
)

if archivo:
    if archivo.name.endswith(".csv"):
        df = pd.read_csv(archivo)
    else:
        df = pd.read_excel(archivo)

    st.markdown(
        '<div class="section-title"><i class="fa-solid fa-table"></i> Vista previa de los datos</div>',
        unsafe_allow_html=True
    )

    # Streamlit muestra la vista previa de los datos
    st.dataframe(df.head())

    tipos = detectar_tipos(df)

    st.sidebar.markdown(
        '<div class="sub-title"><i class="fa-solid fa-layer-group"></i> Tipos de variables detectadas</div>',
        unsafe_allow_html=True
    )
    st.sidebar.write(tipos)

    # Streamlit crea la navegación por tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Estadísticas",
        "Correlación",
        "Outliers",
        "Clustering",
        "Visualizaciones"
    ])

    with tab1:
        if tipos["numericas"]:
            # Pandas genera las estadísticas descriptivas
            stats_df = estadisticas_basicas(df, tipos["numericas"])

            # Streamlit organiza el encabezado y los botones de descarga
            header_col1, header_col2, header_col3 = st.columns([7, 1, 1])

            with header_col1:
                st.markdown(
                    '<div class="section-title"><i class="fa-solid fa-square-poll-vertical"></i> Estadísticas descriptivas</div>',
                    unsafe_allow_html=True
                )

            # Streamlit crea botón para descargar PDF
            # El PDF se genera con Matplotlib
            with header_col2:
                st.download_button(
                    "PDF",
                    data=generar_pdf_estadisticas(stats_df),
                    file_name="estadisticas.pdf",
                    mime="application/pdf",
                    key="tab_est_pdf",
                    use_container_width=True
                )

            # Streamlit crea botón para descargar HTML
            # El HTML se genera con Pandas
            with header_col3:
                st.download_button(
                    "HTML",
                    data=generar_html_estadisticas(stats_df),
                    file_name="estadisticas.html",
                    mime="text/html",
                    key="tab_est_html",
                    use_container_width=True
                )

            # Streamlit muestra la tabla
            # La tabla viene de Pandas
            st.write(stats_df)

    with tab2:
        if len(tipos["numericas"]) < 2:
            st.warning("Se requieren al menos 2 columnas numéricas para calcular correlaciones.")
        else:
    
            if tipos["numericas"]:
        # Calcula ambas correlaciones y la figura combinada
                corr_fig, corr_pearson, corr_spearman = matriz_correlacion_fig(
                df, tipos["numericas"], triangular=True
        )

        # --- Header general ---
                header_col1, _, _ = st.columns([7, 1, 1])
                with header_col1:
                    st.markdown(
                '<div class="section-title"><i class="fa-solid fa-link"></i> Matriz de correlación</div>',
                unsafe_allow_html=True
            )

        # --- Bloque Pearson ---
                st.subheader("Pearson")
                st.dataframe(corr_pearson.style.background_gradient(cmap="coolwarm"))
                st.download_button(
            "Descargar Pearson (CSV)",
            data=corr_pearson.to_csv().encode("utf-8"),
            file_name="correlacion_pearson.csv",
            mime="text/csv",
            key="pearson_csv"
        )

        # --- Bloque Spearman ---
                st.subheader("Spearman")
                st.dataframe(corr_spearman.style.background_gradient(cmap="coolwarm"))
                st.download_button(
            "Descargar Spearman (CSV)",
            data=corr_spearman.to_csv().encode("utf-8"),
            file_name="correlacion_spearman.csv",
            mime="text/csv",
            key="spearman_csv"
        )

        # Figura con ambos heatmaps
                st.pyplot(corr_fig)
            else:
                st.info("No hay columnas numéricas para calcular correlaciones.")
    with tab3:
        if tipos["numericas"]:
            # Streamlit crea el selector de columna
            columna_outlier = st.selectbox("Selecciona columna para outliers", tipos["numericas"])

            # SciPy se usa en Z-score
            # Pandas y NumPy se usan en IQR
            # Scikit-learn se usa en Isolation Forest
            z_df = outliers_zscore(df, columna_outlier)
            iqr_df = outliers_iqr(df, columna_outlier)
            iso_df = outliers_isolation_forest(df, tipos["numericas"])

            header_col1, header_col2, header_col3 = st.columns([7, 1, 1])

            with header_col1:
                st.markdown(
                    '<div class="section-title"><i class="fa-solid fa-triangle-exclamation"></i> Detección de outliers</div>',
                    unsafe_allow_html=True
                )

            # Streamlit crea botón para descargar PDF
            # El PDF usa Matplotlib
            with header_col2:
                st.download_button(
                    "PDF",
                    data=generar_pdf_outliers(z_df, iqr_df, iso_df),
                    file_name="outliers.pdf",
                    mime="application/pdf",
                    key="tab_out_pdf",
                    use_container_width=True
                )

            # Streamlit crea botón para descargar HTML
            # El HTML usa tablas generadas con Pandas
            with header_col3:
                st.download_button(
                    "HTML",
                    data=generar_html_outliers(z_df, iqr_df, iso_df, columna_outlier),
                    file_name="outliers.html",
                    mime="text/html",
                    key="tab_out_html",
                    use_container_width=True
                )

            # Streamlit muestra tabla de outliers por Z-score
            st.markdown(
                '<div class="sub-title"><i class="fa-solid fa-wave-square"></i> Outliers Z-score</div>',
                unsafe_allow_html=True
            )
            st.dataframe(z_df)

            # Streamlit muestra tabla de outliers por IQR
            st.markdown(
                '<div class="sub-title"><i class="fa-solid fa-ruler-combined"></i> Outliers IQR</div>',
                unsafe_allow_html=True
            )
            st.dataframe(iqr_df)

            # Streamlit muestra tabla de outliers por Isolation Forest
            st.markdown(
                '<div class="sub-title"><i class="fa-solid fa-shield-halved"></i> Outliers Isolation Forest</div>',
                unsafe_allow_html=True
            )
            st.dataframe(iso_df)


    with tab4:
        if tipos["numericas"]:
            # Streamlit crea el slider
            n_clusters = st.slider("Número de clusters (KMeans)", 2, 10, 3)

            # Scikit-learn se usa para clustering con KMeans y DBSCAN
            df_kmeans = clustering_kmeans(df, tipos["numericas"], n_clusters)
            df_cluster = clustering_dbscan(df_kmeans, tipos["numericas"])

            header_col1, header_col2, header_col3 = st.columns([7, 1, 1])

            with header_col1:
                st.markdown(
                    '<div class="section-title"><i class="fa-solid fa-object-group"></i> Clustering</div>',
                    unsafe_allow_html=True
                )

            # Streamlit crea botón para descargar PDF
            # El PDF usa Matplotlib
            with header_col2:
                st.download_button(
                    "PDF",
                    data=generar_pdf_clustering(df_cluster),
                    file_name="clustering.pdf",
                    mime="application/pdf",
                    key="tab_cluster_pdf",
                    use_container_width=True
                )

            # Streamlit crea botón para descargar HTML
            # El HTML usa tabla de Pandas
            with header_col3:
                st.download_button(
                    "HTML",
                    data=generar_html_clustering(df_cluster),
                    file_name="clustering.html",
                    mime="text/html",
                    key="tab_cluster_html",
                    use_container_width=True
                )

            # Streamlit muestra la tabla del clustering
            st.markdown(
                '<div class="sub-title"><i class="fa-solid fa-diagram-project"></i> Resultados con clustering</div>',
                unsafe_allow_html=True
            )
            st.dataframe(df_cluster.head())


    with tab5:
        if tipos["numericas"]:
            header_col1, header_col2, header_col3 = st.columns([7, 1, 1])

            with header_col1:
                st.markdown(
                    '<div class="section-title"><i class="fa-solid fa-chart-column"></i> Histogramas interactivos</div>',
                    unsafe_allow_html=True
                )

            # Streamlit crea botón para descargar PDF
            # El PDF usa Matplotlib para generar histogramas
            with header_col2:
                st.download_button(
                    "PDF",
                    data=generar_pdf_visualizaciones(df, tipos["numericas"]),
                    file_name="visualizaciones.pdf",
                    mime="application/pdf",
                    key="tab_vis_pdf",
                    use_container_width=True
                )

            # Streamlit crea botón para descargar HTML
            # El HTML usa histogramas convertidos a imagen
            with header_col3:
                st.download_button(
                    "HTML",
                    data=generar_html_visualizaciones(df, tipos["numericas"]),
                    file_name="visualizaciones.html",
                    mime="text/html",
                    key="tab_vis_html",
                    use_container_width=True
                )

            # Aquí se ubican los histogramas interactivos
            # Usa Plotly Express para mostrar histogramas en pantalla
            for col in tipos["numericas"]:
                fig_plotly = px.histogram(df, x=col, nbins=20, title=f"Histograma de {col}")
                st.plotly_chart(fig_plotly, use_container_width=True)

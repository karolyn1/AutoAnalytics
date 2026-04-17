import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io
import base64
import json
import re
import csv

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
.insight-box{
    background-color: #1a2d1a;
    border-left: 4px solid #4caf50;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    color: #c8e6c9;
    font-size: 0.9rem;
}
.alert-box{
    background-color: #2d1a1a;
    border-left: 4px solid #f44336;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    color: #ffcdd2;
    font-size: 0.9rem;
}
.info-box{
    background-color: #1a2233;
    border-left: 4px solid #2196f3;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    color: #bbdefb;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Funciones auxiliares
# =========================

def detectar_tipos(df):
    # Clasifica las columnas del DataFrame según su tipo de dato
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

def matriz_correlacion_fig(df, columnas_numericas, figsize=(20, 8), triangular=True, cmap="coolwarm"):

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
    # Detecta outliers usando Z-score con SciPy. Marca valores con |z| > 3
    df_temp = df.copy()
    df_temp[columna] = df_temp[columna].fillna(df_temp[columna].mean())
    df_temp["zscore"] = zscore(df_temp[columna])
    return df_temp[df_temp["zscore"].abs() > 3]

def outliers_iqr(df, columna):
    # Detecta outliers usando el Rango Intercuartílico (IQR) con Pandas y NumPy
    df_temp = df.copy()
    df_temp[columna] = df_temp[columna].fillna(df_temp[columna].mean())
    Q1 = df_temp[columna].quantile(0.25)
    Q3 = df_temp[columna].quantile(0.75)
    IQR = Q3 - Q1
    return df_temp[
        (df_temp[columna] < Q1 - 1.5 * IQR) |
        (df_temp[columna] > Q3 + 1.5 * IQR)
    ]

def outliers_isolation_forest(df, columnas_numericas):
    # Detecta outliers usando Isolation Forest de Scikit-learn
    df_temp = df.copy()
    datos = df_temp[columnas_numericas].copy()
    datos = datos.fillna(datos.mean())
    iso = IsolationForest(contamination=0.1, random_state=42)
    df_temp["anomaly"] = iso.fit_predict(datos)
    return df_temp[df_temp["anomaly"] == -1]

def clustering_kmeans(df, columnas_numericas, n_clusters=3):
    # Aplica clustering KMeans con Scikit-learn y agrega la etiqueta de cluster al DataFrame
    df_temp = df.copy()
    datos = df_temp[columnas_numericas].copy()
    datos = datos.fillna(datos.mean())
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_temp["cluster_kmeans"] = kmeans.fit_predict(datos)
    return df_temp

def clustering_dbscan(df, columnas_numericas, eps=0.5, min_samples=5):
    # Aplica clustering DBSCAN con Scikit-learn. Etiqueta -1 indica ruido/outliers
    df_temp = df.copy()
    datos = df_temp[columnas_numericas].copy()
    datos = datos.fillna(datos.mean())
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df_temp["cluster_dbscan"] = dbscan.fit_predict(datos)
    return df_temp

def fig_to_base64(fig):
    # Convierte una figura de Matplotlib a base64 para incrustarla en HTML
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def fig_to_pdf_bytes(figs):
    # Convierte una lista de figuras de Matplotlib a bytes de PDF usando PdfPages
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()

def tabla_a_figura(df, titulo="Tabla"):
    # Convierte un DataFrame en figura de Matplotlib, exportable a PDF
    df = df.copy()
    fig, ax = plt.subplots(figsize=(12, min(0.5 * len(df) + 2, 12)))
    ax.axis("off")
    ax.set_title(titulo, fontsize=12, pad=12)
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
    # Convierte una Serie o DataFrame con índice en figura exportable
    df_reset = df.reset_index()
    return tabla_a_figura(df_reset, titulo)

def histograma_matplotlib(df, columna):
    # Genera un histograma con Matplotlib para exportación a PDF
    datos = df[columna].dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(datos, bins=20)
    ax.set_title(f"Histograma de {columna}")
    ax.set_xlabel(columna)
    ax.set_ylabel("Frecuencia")
    return fig

def generar_html_estadisticas(stats_df):
    # Genera un archivo HTML con la tabla de estadísticas descriptivas usando Pandas
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
    # Genera un PDF con la tabla de estadísticas usando Matplotlib
    fig = tabla_serie_a_figura(stats_df.round(2), "Estadísticas descriptivas")
    return fig_to_pdf_bytes([fig])

def generar_html_correlacion(corr_df, corr_fig):
    # Genera un HTML con el heatmap de correlación incrustado como imagen base64
    img = fig_to_base64(corr_fig)
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
    # Genera un PDF con la figura y la tabla de correlación
    fig_tabla = tabla_a_figura(corr_df.round(2), "Tabla de correlación")
    return fig_to_pdf_bytes([corr_fig, fig_tabla])

def generar_html_outliers(z_df, iqr_df, iso_df, columna):
    # Genera un HTML con las tres tablas de outliers (Z-score, IQR, Isolation Forest)
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
    # Genera un PDF con las tablas de outliers detectados.
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
    # Genera un HTML con la tabla de resultados del clustering.
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
    # Genera un PDF con los primeros 30 registros del clustering.
    fig = tabla_a_figura(df_cluster.head(30), "Resultados de clustering")
    return fig_to_pdf_bytes([fig])

def generar_html_visualizaciones(df, columnas_numericas):
    # Genera un HTML con todos los histogramas incrustados como imágenes base64.
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
    # Genera un PDF con un histograma por cada variable numérica.
    figs = []
    for col in columnas_numericas:
        figs.append(histograma_matplotlib(df, col))
    return fig_to_pdf_bytes(figs)

#  Relaciones bivariadas
def mostrar_relaciones(df, tipos):
    
    # Muestra relaciones entre pares de variables.
    # - Numérica × Numérica: scatter plot con línea de tendencia y coeficiente r de Pearson.
    # - Categórica × Numérica: box plot que compara la distribución de la numérica por cada categoría.
    
    numericas = tipos["numericas"]
    categoricas = tipos["categoricas"]

    # ── Relación Numérica × Numérica
    if len(numericas) >= 2:
        st.markdown("#### 🔢 Numérica × Numérica")

        # El usuario elige las dos variables a comparar
        col1, col2 = st.columns(2)
        with col1:
            var_x = st.selectbox("Variable X:", numericas, key="rel_x")
        with col2:
            # Evitar que X e Y sean la misma variable
            opciones_y = [c for c in numericas if c != var_x]
            var_y = st.selectbox("Variable Y:", opciones_y, key="rel_y")

        # Calcular coeficiente de Pearson entre las dos variables
        datos_par = df[[var_x, var_y]].dropna()
        r = datos_par[var_x].corr(datos_par[var_y])

        # Scatter plot con línea de tendencia usando Plotly
        fig = px.scatter(
            datos_par, x=var_x, y=var_y,
            trendline="ols",
            title=f"{var_x} vs {var_y}  |  Pearson r = {r:.4f}"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Interpretación automática del coeficiente
        fuerza = "muy fuerte" if abs(r) >= 0.9 else "fuerte" if abs(r) >= 0.7 else "moderada" if abs(r) >= 0.4 else "débil"
        direccion = "positiva" if r > 0 else "negativa"
        st.markdown(
            f'<div class="info-box">Correlación {fuerza} {direccion} entre <b>{var_x}</b> y <b>{var_y}</b> (r = {r:.4f}). '
            f'<b>{var_x}</b> explica el <b>{r**2 * 100:.1f}%</b> de la variación en <b>{var_y}</b>.</div>',
            unsafe_allow_html=True
        )

    # ── Relación Categórica × Numérica
    if categoricas and numericas:
        st.markdown("#### 🏷️ Categórica × Numérica")

        col1, col2 = st.columns(2)
        with col1:
            var_cat = st.selectbox("Variable categórica:", categoricas, key="rel_cat")
        with col2:
            var_num = st.selectbox("Variable numérica:", numericas, key="rel_num")

        # Box plot: muestra cómo varía la numérica en cada categoría
        fig2 = px.box(
            df, x=var_cat, y=var_num,
            title=f"Distribución de {var_num} por {var_cat}"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Calcular la media por categoría para el insight
        medias = df.groupby(var_cat)[var_num].mean().sort_values(ascending=False)
        mejor = medias.index[0]
        peor = medias.index[-1]
        st.markdown(
            f'<div class="info-box">La categoría <b>"{mejor}"</b> tiene el mayor promedio de <b>{var_num}</b> '
            f'({medias.iloc[0]:.2f}), mientras que <b>"{peor}"</b> tiene el menor ({medias.iloc[-1]:.2f}).</div>',
            unsafe_allow_html=True
        )

# generación de Insights automáticos
def generar_insights(df, tipos, corr_pearson):
    # Analiza los resultados del EDA y genera frases interpretables automáticamente.
    
    insights = []
    alertas  = []
    info     = []

    numericas   = tipos["numericas"]
    categoricas = tipos["categoricas"]
    n, m = df.shape

    # ── Información general del dataset
    info.append(f"El dataset tiene <b>{n:,} registros</b> y <b>{m} variables</b> "
                f"({len(numericas)} numéricas, {len(categoricas)} categóricas).")

    # ── Valores faltantes 
    total_nulos = df.isnull().sum().sum()
    pct_nulos = total_nulos / (n * m) * 100
    if pct_nulos > 20:
        alertas.append(f"El dataset tiene un <b>{pct_nulos:.1f}%</b> de valores faltantes. Se recomienda revisar la calidad de los datos.")
    elif pct_nulos > 0:
        info.append(f"Se detectó un <b>{pct_nulos:.1f}%</b> de valores faltantes en el dataset.")

    # ── Correlaciones fuertes (desde la matriz de Pearson) 
    if corr_pearson is not None:
        vistos = set()
        for col_a in numericas:
            for col_b in numericas:
                if col_a == col_b:
                    continue
                par = tuple(sorted([col_a, col_b]))
                if par in vistos:
                    continue
                vistos.add(par)
                r = corr_pearson.loc[col_a, col_b]
                if abs(r) >= 0.7:
                    fuerza = "muy fuerte" if abs(r) >= 0.9 else "fuerte"
                    direccion = "positiva" if r > 0 else "negativa"
                    insights.append(f"Correlación {fuerza} {direccion} entre <b>{col_a}</b> y <b>{col_b}</b> (r = {r:.2f}).")

    # ── Asimetría en variables numéricas 
    for col in numericas:
        sk = df[col].dropna().skew()
        if abs(sk) > 1.5:
            lado = "derecha (valores altos extremos)" if sk > 0 else "izquierda (valores bajos extremos)"
            info.append(f"La variable <b>{col}</b> tiene distribución asimétrica hacia la {lado} (skew = {sk:.2f}).")

    # ── Categorías dominantes 
    for col in categoricas:
        vc = df[col].value_counts()
        if len(vc) > 0:
            top_pct = vc.iloc[0] / n * 100
            if top_pct >= 80:
                insights.append(f"En <b>{col}</b>, el valor <b>'{vc.index[0]}'</b> domina con el <b>{top_pct:.1f}%</b> de los registros.")

    return insights, alertas, info

# Exportación JSON
def generar_json_eda(df, tipos, stats_df, corr_pearson, insights, alertas, info):
    """
    Genera un resumen del análisis exploratorio en formato JSON.
    Incluye: resumen del dataset, estadísticas descriptivas,
    correlaciones fuertes, insights, alertas e información general.
    """

    # Eliminar etiquetas HTML de los textos (<b>, etc.)
    def limpiar(texto):
        return re.sub(r"<[^>]+>", "", texto)

    # Correlaciones fuertes (|r| >= 0.7)
    correlaciones = []
    if corr_pearson is not None:
        vistos = set()
        for col_a in tipos["numericas"]:
            for col_b in tipos["numericas"]:
                if col_a == col_b:
                    continue
                par = tuple(sorted([col_a, col_b]))
                if par in vistos:
                    continue
                vistos.add(par)
                r = corr_pearson.loc[col_a, col_b]
                if abs(r) >= 0.7:
                    correlaciones.append({
                        "variable_1": col_a,
                        "variable_2": col_b,
                        "pearson_r": round(float(r), 4),
                        "fuerza": "muy fuerte" if abs(r) >= 0.9 else "fuerte",
                        "direccion": "positiva" if r > 0 else "negativa"
                    })

    reporte = {
        "resumen_dataset": {
            "registros": len(df),
            "variables": len(df.columns),
            "numericas": len(tipos["numericas"]),
            "categoricas": len(tipos["categoricas"]),
            "valores_faltantes": int(df.isnull().sum().sum())
        },
        "estadisticas_descriptivas": stats_df.round(4).to_dict() if stats_df is not None else {},
        "correlaciones_fuertes": correlaciones,
        "insights": [limpiar(x) for x in insights],
        "alertas":  [limpiar(x) for x in alertas],
        "informacion": [limpiar(x) for x in info],
        "generado_en": str(pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"))
    }

    return json.dumps(reporte, ensure_ascii=False, indent=2).encode("utf-8")

st.sidebar.markdown(
    '<div class="sidebar-title"><i class="fa-solid fa-sliders"></i> Opciones</div>',
    unsafe_allow_html=True
)

archivo = st.sidebar.file_uploader("Carga tu archivo CSV o Excel", type=["csv", "xlsx"])

st.markdown(
    '<div class="main-title"><i class="fa-solid fa-chart-line"></i> Dashboard de Análisis Automatizado</div>',
    unsafe_allow_html=True
)

if archivo:
  

    if archivo:
        if archivo.name.endswith(".csv"):
        # Leer primera línea para detectar separador
            first_line = archivo.readline().decode("utf-8")
            archivo.seek(0)  # volver al inicio

            if ";" in first_line:
                df = pd.read_csv(archivo, sep=";")
            elif "," in first_line:
                df = pd.read_csv(archivo, sep=",")
            else:
            # Si no hay ni coma ni punto y coma, usar read_csv normal
                df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo)

    df = df.dropna() #eliminamos nulos
    st.markdown(
        '<div class="section-title"><i class="fa-solid fa-table"></i> Vista previa de los datos</div>',
        unsafe_allow_html=True
    )
    # Streamlit muestra la vista previa de los datos
    st.dataframe(df.head())

    tipos = detectar_tipos(df) # iniciamos con el análisis detectando el tipo de data

    st.sidebar.markdown(
        '<div class="sub-title"><i class="fa-solid fa-layer-group"></i> Tipos de variables detectadas</div>',
        unsafe_allow_html=True
    )
    st.sidebar.write(tipos) # colocamos en el sidebar los tipos detectados del dataset

    # Calcular correlación una sola vez para compartirla entre tabs
    corr_pearson = None
    stats_df = None
    if len(tipos["numericas"]) >= 2:
        datos_num = df[tipos["numericas"]].copy().apply(lambda s: s.fillna(s.mean()), axis=0)
        corr_pearson = datos_num.corr(method="pearson")
    if tipos["numericas"]:
        stats_df = estadisticas_basicas(df, tipos["numericas"])

    # Generar insights una sola vez para compartirlos entre tabs
    insights, alertas, info = generar_insights(df, tipos, corr_pearson)


    # Panel de insights
    with st.expander(" Insights y Alertas Automáticas", expanded=True):
        if not alertas and not insights and not info:
            st.info("No se generaron insights con los datos actuales.")
        for a in alertas:
            st.markdown(f'<div class="alert-box"> {a}</div>', unsafe_allow_html=True)
        for i in insights:
            st.markdown(f'<div class="insight-box"> {i}</div>', unsafe_allow_html=True)
        for i in info:
            st.markdown(f'<div class="info-box"> {i}</div>', unsafe_allow_html=True)
    # Streamlit crea la navegación por tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Estadísticas",
        "Correlación",
        "Outliers",
        "Clustering",
        "Visualizaciones",
        "Relaciones",       
        "Exportar JSON"     
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
                    "PDF", data=generar_pdf_estadisticas(stats_df),
                    file_name="estadisticas.pdf", mime="application/pdf",
                    key="tab_est_pdf", use_container_width=True
                )
            # Streamlit crea botón para descargar HTML
            # El HTML se genera con Pandas
            with header_col3:
                st.download_button(
                    "HTML", data=generar_html_estadisticas(stats_df),
                    file_name="estadisticas.html", mime="text/html",
                    key="tab_est_html", use_container_width=True
                )
            # Streamlit muestra la tabla
            # La tabla viene de Pandas
            st.write(stats_df)

    with tab2:
        if len(tipos["numericas"]) < 2:
            st.warning("Se requieren al menos 2 columnas numéricas para calcular correlaciones.")
        else:
            corr_fig, corr_pearson_fig, corr_spearman = matriz_correlacion_fig(df, tipos["numericas"], triangular=True)

            header_col1, _, _ = st.columns([7, 1, 1])
            with header_col1:
                st.markdown(
                    '<div class="section-title"><i class="fa-solid fa-link"></i> Matriz de correlación</div>',
                    unsafe_allow_html=True
                )

            st.subheader("Pearson")
            st.dataframe(corr_pearson_fig.style.background_gradient(cmap="coolwarm"))
            st.download_button(
                "Descargar Pearson (CSV)", data=corr_pearson_fig.to_csv().encode("utf-8"),
                file_name="correlacion_pearson.csv", mime="text/csv", key="pearson_csv"
            )

            st.subheader("Spearman")
            st.dataframe(corr_spearman.style.background_gradient(cmap="coolwarm"))
            st.download_button(
                "Descargar Spearman (CSV)", data=corr_spearman.to_csv().encode("utf-8"),
                file_name="correlacion_spearman.csv", mime="text/csv", key="spearman_csv"
            )

            st.pyplot(corr_fig)

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
                    "PDF", data=generar_pdf_outliers(z_df, iqr_df, iso_df),
                    file_name="outliers.pdf", mime="application/pdf",
                    key="tab_out_pdf", use_container_width=True
                )
            # Streamlit crea botón para descargar HTML
            # El HTML usa tablas generadas con Pandas
            with header_col3:
                st.download_button(
                    "HTML", data=generar_html_outliers(z_df, iqr_df, iso_df, columna_outlier),
                    file_name="outliers.html", mime="text/html",
                    key="tab_out_html", use_container_width=True
                )

            # Streamlit muestra tabla de outliers por Z-score
            st.markdown('<div class="sub-title"><i class="fa-solid fa-wave-square"></i> Outliers Z-score</div>', unsafe_allow_html=True)
            st.dataframe(z_df)
            # Streamlit muestra tabla de outliers por IQR
            st.markdown('<div class="sub-title"><i class="fa-solid fa-ruler-combined"></i> Outliers IQR</div>', unsafe_allow_html=True)
            st.dataframe(iqr_df)
             # Streamlit muestra tabla de outliers por Isolation Forest
            st.markdown('<div class="sub-title"><i class="fa-solid fa-shield-halved"></i> Outliers Isolation Forest</div>', unsafe_allow_html=True)
            st.dataframe(iso_df)
            
    with tab4:
        if tipos["numericas"]:
            # Streamlit crea el slider
            n_clusters = st.slider("Número de clusters (KMeans)", 2, 10, 3)

            # Scikit-learn se usa para clustering con KMeans y DBSCAN
            df_kmeans  = clustering_kmeans(df, tipos["numericas"], n_clusters)
            df_cluster = clustering_dbscan(df_kmeans, tipos["numericas"])

            header_col1, header_col2, header_col3 = st.columns([7, 1, 1])
            with header_col1:
                st.markdown(
                    '<div class="section-title"><i class="fa-solid fa-object-group"></i> Clustering</div>',
                    unsafe_allow_html=True
                )
            # Streamlit crea botón para descargar PDF
            with header_col2:
                st.download_button(
                    "PDF", data=generar_pdf_clustering(df_cluster),
                    file_name="clustering.pdf", mime="application/pdf",
                    key="tab_cluster_pdf", use_container_width=True
                )
            # Streamlit crea botón para descargar HTML
            with header_col3:
                st.download_button(
                    "HTML", data=generar_html_clustering(df_cluster),
                    file_name="clustering.html", mime="text/html",
                    key="tab_cluster_html", use_container_width=True
                )

            # Streamlit muestra la tabla del clustering
            st.markdown(
                '<div class="sub-title"><i class="fa-solid fa-diagram-project"></i> Resultados con clustering</div>',
                unsafe_allow_html=True
            )
            st.dataframe(df_cluster.head())

            # Gráficos de clustering
            st.markdown(
                '<div class="sub-title"><i class="fa-solid fa-chart-scatter"></i> Gráficos de clustering</div>',
                unsafe_allow_html=True
            )

            # Cantidad de elementos por cluster KMeans
            conteo_kmeans = df_cluster["cluster_kmeans"].value_counts().sort_index().reset_index()
            conteo_kmeans.columns = ["cluster_kmeans", "cantidad"]

            fig_bar_kmeans = px.bar(
                conteo_kmeans,
                x="cluster_kmeans",
                y="cantidad",
                title="Cantidad de registros por cluster (KMeans)",
                text="cantidad"
            )
            st.plotly_chart(fig_bar_kmeans, use_container_width=True)

            # Cantidad de elementos por cluster DBSCAN
            conteo_dbscan = df_cluster["cluster_dbscan"].value_counts().sort_index().reset_index()
            conteo_dbscan.columns = ["cluster_dbscan", "cantidad"]

            fig_bar_dbscan = px.bar(
                conteo_dbscan,
                x="cluster_dbscan",
                y="cantidad",
                title="Cantidad de registros por cluster (DBSCAN)",
                text="cantidad"
            )
            st.plotly_chart(fig_bar_dbscan, use_container_width=True)

            # Scatter plot usando las dos primeras columnas numéricas
            if len(tipos["numericas"]) >= 2:
                x_col = tipos["numericas"][0]
                y_col = tipos["numericas"][1]

                fig_kmeans = px.scatter(
                    df_cluster,
                    x=x_col,
                    y=y_col,
                    color=df_cluster["cluster_kmeans"].astype(str),
                    title=f"KMeans: {x_col} vs {y_col}",
                    labels={"color": "cluster_kmeans"}
                )
                st.plotly_chart(fig_kmeans, use_container_width=True)

                fig_dbscan = px.scatter(
                    df_cluster,
                    x=x_col,
                    y=y_col,
                    color=df_cluster["cluster_dbscan"].astype(str),
                    title=f"DBSCAN: {x_col} vs {y_col}",
                    labels={"color": "cluster_dbscan"}
                )
                st.plotly_chart(fig_dbscan, use_container_width=True)

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
                    "PDF", data=generar_pdf_visualizaciones(df, tipos["numericas"]),
                    file_name="visualizaciones.pdf", mime="application/pdf",
                    key="tab_vis_pdf", use_container_width=True
                )
            # Streamlit crea botón para descargar HTML
            # El HTML usa histogramas convertidos a imagen
            with header_col3:
                st.download_button(
                    "HTML", data=generar_html_visualizaciones(df, tipos["numericas"]),
                    file_name="visualizaciones.html", mime="text/html",
                    key="tab_vis_html", use_container_width=True
                )

            # Aquí se ubican los histogramas interactivos
            # Usa Plotly Express para mostrar histogramas en pantalla
            for col in tipos["numericas"]:
                fig_plotly = px.histogram(df, x=col, nbins=20, title=f"Histograma de {col}")
                st.plotly_chart(fig_plotly, use_container_width=True)

    with tab6:
        st.markdown(
            '<div class="section-title"><i class="fa-solid fa-arrow-right-arrow-left"></i> Relaciones entre variables</div>',
            unsafe_allow_html=True
        )
        mostrar_relaciones(df, tipos)

    with tab7:
        st.markdown(
            '<div class="section-title"><i class="fa-solid fa-file-code"></i> Exportar resumen en JSON</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="info-box">El archivo JSON contiene el resumen del dataset, estadísticas descriptivas, '
            'correlaciones fuertes, insights, alertas e información general del análisis.</div>',
            unsafe_allow_html=True
        )

        json_bytes = generar_json_eda(df, tipos, stats_df, corr_pearson, insights, alertas, info)

        st.download_button(
            label="⬇️ Descargar JSON",
            data=json_bytes,
            file_name="reporte_eda.json",
            mime="application/json",
            use_container_width=True
        )
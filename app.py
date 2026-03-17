"""
AutoAnalytics - Análisis Inteligente de Datos
Analiza automáticamente cualquier tipo de datos: numéricos, categóricos, texto
"""
import streamlit as st #construye la interfaz
import pandas as pd #manipula datos tobulares
import numpy as np #operaciones matemáticas y generación de datos
import plotly.express as px #para gráficos
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler #libs de ML, normaliza datos
from sklearn.cluster import KMeans #esta realiza clustering
from sklearn.ensemble import IsolationForest #esta detecta anomalías en los datos
from sklearn.decomposition import PCA #esta realiza reducción de dimensiones
from sklearn.metrics import silhouette_score #usada para evaluar clusters
from scipy import stats #usada para z-score en outliers
import io, base64, json, datetime, warnings 
warnings.filterwarnings("ignore")


# Aquí se realiza la configuración de la página con streamlit
st.set_page_config(page_title="AutoAnalytics", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")


st.markdown("""<style>
.stApp{background:#0f1117}
section[data-testid="stSidebar"]{background:#161b27;border-right:1px solid #1e2d40}
.mcard{background:linear-gradient(135deg,#1a2332,#1e2d40);border:1px solid #2a3f55;border-radius:12px;padding:18px;text-align:center;margin:4px 0}
.mcard h2{color:#4db8ff;font-size:2rem;margin:0}.mcard p{color:#8ba3bb;font-size:.82rem;margin:4px 0 0}
.ibox{background:#1a2d1a;border-left:4px solid #4caf50;border-radius:8px;padding:12px 16px;margin:6px 0;color:#c8e6c9;font-size:.9rem}
.abox{background:#2d1a1a;border-left:4px solid #f44336;border-radius:8px;padding:12px 16px;margin:6px 0;color:#ffcdd2;font-size:.9rem}
.ifbox{background:#1a2233;border-left:4px solid #2196f3;border-radius:8px;padding:12px 16px;margin:6px 0;color:#bbdefb;font-size:.9rem}
.wbox{background:#2d2a1a;border-left:4px solid #ff9800;border-radius:8px;padding:12px 16px;margin:6px 0;color:#ffe0b2;font-size:.9rem}
.stitle{font-size:1.3rem;font-weight:700;color:#4db8ff;border-bottom:2px solid #1e3a55;padding-bottom:6px;margin:20px 0 14px}
.stTabs [data-baseweb="tab-list"]{background:#161b27;border-radius:8px;padding:4px}
.stTabs [data-baseweb="tab"]{color:#8ba3bb;border-radius:6px}
.stTabs [data-baseweb="tab"][aria-selected="true"]{background:#1e3a55;color:#4db8ff}
h1,h2,h3{color:#e0e0e0} p,li{color:#b0bec5}
div[data-testid="stFileUploadDropzone"]{background:#161b27 !important;border:2px dashed #2a3f55 !important;border-radius:12px !important}
</style>""", unsafe_allow_html=True)

COLORS = px.colors.qualitative.Plotly #paleta de colores estándar para gráficos

# ── Helpers ────────────────────────────────────────────────────────────────────

# Esto es para realizar un demo, y que la app muestre datos aunque no se cargue nada

def make_demo():
    np.random.seed(42); n=500
    d = pd.DataFrame({
        "edad": np.random.normal(38,12,n).astype(int).clip(18,75),
        "ingreso": np.random.lognormal(8.5,.6,n).round(2),
        "gasto": np.random.lognormal(7.8,.5,n).round(2),
        "antiguedad": np.random.exponential(36,n).astype(int).clip(1,240),
        "n_productos": np.random.poisson(2.5,n).clip(0,10).astype(int),
        "score_credito": np.random.normal(650,80,n).astype(int).clip(300,850),
        "region": np.random.choice(["Norte","Sur","Centro","Este","Oeste"],n,p=[.25,.2,.3,.15,.1]),
        "segmento": np.random.choice(["Premium","Estándar","Básico"],n,p=[.2,.5,.3]),
        "activo": np.random.choice(["Sí","No"],n,p=[.85,.15])
    })
    d["publicidad"] = d["ingreso"]*np.random.uniform(.05,.15,n)
    d["ventas"] = d["publicidad"]*8.7 + np.random.normal(0,500,n)
    idx = np.random.choice(n,15,replace=False)
    d.loc[idx,"ingreso"] *= np.random.uniform(5,10,15)
    return d


#Ingesta y Preprocesamiento de Datos 
def smart_read_excel(file_obj, sheet_name=0):
    """Lee Excel detectando automáticamente dónde están los encabezados reales."""
    raw = pd.read_excel(file_obj, header=None, sheet_name=sheet_name)
    best_row, best_score = 0, 0
    for i in range(min(20, len(raw))):
        score = raw.iloc[i].notna().sum()
        if score > best_score:
            best_score = score
            best_row = i
    file_obj.seek(0)
    df = pd.read_excel(file_obj, header=best_row, sheet_name=sheet_name)
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

     # Eliminar filas duplicadas
    n_antes = len(df)
    df = df.drop_duplicates()
    n_dupes = n_antes - len(df)

    return df, best_row, n_dupes
# Esta función recorre las columnas del dataset y determina el tipo de dato de cada columna ya sea numérica, categórica o booleana, etc...
def detect_types(df):
    """Clasifica columnas: numeric, categorical, temporal, boolean, text_id, text_long."""
    t = {}
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            t[c] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(col):
            t[c] = "temporal"
        elif pd.api.types.is_bool_dtype(col):
            t[c] = "boolean"
        else:
            try:
                pd.to_datetime(col.dropna().iloc[:5], errors="raise")
                t[c] = "temporal"
                continue
            except:
                pass
            nu = col.nunique()
            nt = len(col.dropna())
            avg_len = col.dropna().astype(str).str.len().mean() if nt > 0 else 0
            if nu <= 1:
                t[c] = "constant"
            elif nu <= min(30, max(2, nt * 0.15)) and avg_len < 50:
                t[c] = "categorical"
            elif avg_len > 80:
                t[c] = "text_long"
            elif nu == nt:
                t[c] = "text_id"
            else:
                t[c] = "text"
    return t

# Esta función analiza las variables categóricas y calcula cuantas veces aparece el mismo valor, el porcentaje que consume 
# y la moda o cuál es la que más se repite
def analyze_categorical(df, cat_cols):
    """Análisis profundo de columnas categóricas."""
    results = {}
    for col in cat_cols:
        vc = df[col].value_counts()
        results[col] = {
            "counts": vc,
            "pcts": (vc / len(df) * 100).round(2),
            "n_unique": df[col].nunique(),
            "missing": df[col].isnull().sum(),
            "top": vc.index[0] if len(vc) > 0 else None,
            "top_pct": round(vc.iloc[0] / len(df) * 100, 1) if len(vc) > 0 else 0
        }
    return results

# Esta función detecta los valores atípicos que se desvían de la mediana del dataset
def do_outliers(df, nc, cont=.05):
    cl = df[nc].dropna()
    if len(cl) < 10: return {}
    z = np.abs(stats.zscore(cl, nan_policy="omit"))
    zm = (z > 3).any(axis=1)
    im = pd.Series(False, index=cl.index)
    pc = {}
    for c in nc:
        Q1, Q3 = cl[c].quantile(.25), cl[c].quantile(.75); I = Q3 - Q1
        m = (cl[c] < Q1 - 1.5*I) | (cl[c] > Q3 + 1.5*I); im |= m; pc[c] = int(m.sum())
    iso = IsolationForest(contamination=cont, random_state=42, n_estimators=100)
    ip = iso.fit_predict(cl) == -1
    return {"z_score": {"count": int(zm.sum()), "pct": round(zm.mean()*100, 2)},
            "iqr": {"count": int(im.sum()), "pct": round(im.mean()*100, 2)},
            "isolation_forest": {"count": int(ip.sum()), "pct": round(ip.mean()*100, 2), "indices": cl.index[ip].tolist()},
            "per_column": pc}

# Esta función aplica los clusteres entre la similitud de sus variables numéricas y realiza la predicción
def do_kmeans(df, nc, mk=6):
    cl = df[nc].dropna()
    if len(cl) < 15 or len(nc) < 2: return None
    sc = StandardScaler(); X = sc.fit_transform(cl)
    kr = range(2, min(mk+1, len(cl)//5+2, 8))
    ins, sils = [], []
    for k in kr:
        km = KMeans(n_clusters=k, random_state=42, n_init=10); lb = km.fit_predict(X)
        ins.append(km.inertia_)
        sils.append(silhouette_score(X, lb) if len(set(lb)) > 1 else 0)
    bi = np.argmax(sils); bk = list(kr)[bi]
    km2 = KMeans(n_clusters=bk, random_state=42, n_init=10); lbs = km2.fit_predict(X)
    pca = PCA(n_components=2); pc = pca.fit_transform(X)
    pdf = pd.DataFrame(pc, columns=["PC1","PC2"], index=cl.index)
    pdf["Cluster"] = (lbs+1).astype(str)
    cl2 = cl.copy(); cl2["_c"] = lbs
    return {"best_k": bk, "labels": lbs, "profiles": cl2.groupby("_c")[nc].mean(),
            "pca_df": pdf, "silhouette": round(sils[bi], 4),
            "k_range": list(kr), "inertias": ins, "silhouettes": sils}

# Esta función calcula las matrices de correlación entre todas las variables numéricas y muestra los pares que mayor correlación tengan
def do_corr(df, nc, thr=.6):
    if len(nc) < 2: return None, None, []
    P = df[nc].corr("pearson"); S = df[nc].corr("spearman") #matrices de correlación
    strong = []; seen = set()
    for i, c1 in enumerate(nc):
        for j, c2 in enumerate(nc):
            if i >= j: continue
            pair = (min(c1,c2), max(c1,c2))
            if pair in seen: continue
            seen.add(pair); r = P.loc[c1,c2]
            if abs(r) >= thr:
                strong.append({"col1":c1, "col2":c2, "pearson_r":round(r,4), "spearman_r":round(S.loc[c1,c2],4),
                    "strength":"Muy fuerte" if abs(r)>=.9 else "Fuerte" if abs(r)>=.7 else "Moderada",
                    "direction":"positiva" if r>0 else "negativa"})
    strong.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)
    return P, S, strong

# Esta función recurre todos los valores númericos almacenados en el dataframe y permite hacer la estadística descriptiva automáticamente
def do_stats(df, nc):
    rows = []
    for c in nc:
        s = df[c].dropna()
        if not len(s): continue
        rows.append({"Variable":c, "N":len(s), "Completo%":f"{len(s)/len(df)*100:.1f}",
            "Media":round(s.mean(),4), "Mediana":round(s.median(),4), "Desv.Std":round(s.std(),4),
            "Mín":round(s.min(),4), "Q1":round(s.quantile(.25),4), "Q3":round(s.quantile(.75),4),
            "Máx":round(s.max(),4), "IQR":round(s.quantile(.75)-s.quantile(.25),4),
            "Asimetría":round(s.skew(),4), "Curtosis":round(s.kurtosis(),4),
            "CV%":round(s.std()/s.mean()*100,2) if s.mean()!=0 else np.nan})
    return pd.DataFrame(rows)

# Analiza todos los resultados de las funciones previas y genera un mensaje entendible para el usuario
def auto_insights(df, vt, ol, cl, co, cat_analysis):
    """Genera insights inteligentes basados en TODOS los tipos de datos."""
    ins, ale, inf = [], [], []
    nc = [c for c,t in vt.items() if t=="numeric"]
    cat_cols = [c for c,t in vt.items() if t=="categorical"]
    n, m = df.shape
    mp = df.isnull().sum().sum()/(n*m)*100

    inf.append(f"📊 Dataset: <b>{n:,} registros × {m} variables</b> | {len(nc)} numéricas, {len(cat_cols)} categóricas.")
    if mp > 0:
        (ale if mp > 20 else inf).append(f"⚠️ <b>{mp:.1f}%</b> de valores faltantes en el dataset.")

    # ── Insights categóricos inteligentes ────────────────────────────────────
    for col, data in cat_analysis.items():
        counts = data["counts"]
        n_vals = len(counts)

        # Detectar columnas de resultado/estado (Pass/Fail, Aprobado/Rechazado, etc.)
        col_lower = col.lower()
        vals_lower = [str(v).lower() for v in counts.index]

        # Patrones de resultado
        pass_keywords = ["pass", "passed", "aprobado", "exitoso", "ok", "success", "activo", "sí", "si", "completado"]
        fail_keywords = ["fail", "failed", "fallido", "error", "rechazado", "no", "inactivo", "cancelado", "pendiente"]

        has_pass = any(any(kw in str(v).lower() for kw in pass_keywords) for v in counts.index)
        has_fail = any(any(kw in str(v).lower() for kw in fail_keywords) for v in counts.index)

        if has_pass and has_fail:
            pass_count = sum(v for k, v in counts.items() if any(kw in str(k).lower() for kw in pass_keywords))
            fail_count = sum(v for k, v in counts.items() if any(kw in str(k).lower() for kw in fail_keywords))
            total = pass_count + fail_count
            pass_pct = round(pass_count/total*100, 1) if total > 0 else 0
            fail_pct = round(fail_count/total*100, 1) if total > 0 else 0
            emoji = "✅" if pass_pct >= 70 else "⚠️" if pass_pct >= 50 else "🚨"
            msg = f"{emoji} <b>'{col}'</b>: <b>{pass_count} pasaron ({pass_pct}%)</b> y <b>{fail_count} fallaron ({fail_pct}%)</b>."
            (ins if pass_pct >= 70 else ale).append(msg)

        # Prioridad/nivel
        elif any(kw in col_lower for kw in ["prioridad", "priority", "nivel", "level", "urgencia"]):
            high_keywords = ["alta", "high", "crítica", "critical", "urgente"]
            high_count = sum(v for k, v in counts.items() if any(kw in str(k).lower() for kw in high_keywords))
            if high_count > 0:
                ins.append(f"🔴 <b>'{col}'</b>: <b>{high_count} registros de alta prioridad</b> ({round(high_count/n*100,1)}% del total).")

        # Columnas con concentración alta en un valor
        elif n_vals >= 2:
            top_pct = data["top_pct"]
            if top_pct >= 80:
                ins.append(f"📌 <b>'{col}'</b>: El valor <b>'{data['top']}'</b> domina con <b>{top_pct}%</b> de los registros.")
            elif n_vals <= 5:
                dist_str = " | ".join([f"{k}: {v}" for k, v in counts.head(5).items()])
                inf.append(f"🏷️ <b>'{col}'</b>: {dist_str}")

    # ── Insights de comentarios/issues ───────────────────────────────────────
    for col, t in vt.items():
        if t in ("text", "text_long") or "comentario" in col.lower() or "comment" in col.lower() or "issue" in col.lower():
            filled = df[col].dropna()
            if len(filled) > 0:
                issue_kw = ["issue", "error", "problema", "fallo", "reportado", "bug"]
                issue_count = sum(1 for v in filled if any(kw in str(v).lower() for kw in issue_kw))
                if issue_count > 0:
                    ale.append(f"🐛 Se detectaron <b>{issue_count} registros con issues/problemas reportados</b> en la columna <b>'{col}'</b>.")

    # ── Insights numéricos ────────────────────────────────────────────────────
    if ol:
        iso = ol.get("isolation_forest", {})
        if iso.get("pct", 0) > 0:
            (ale if iso.get("pct", 0) > 5 else ins).append(
                f"🚨 Isolation Forest detectó <b>{iso['count']} registros atípicos ({iso['pct']}%)</b>.")
        pc = ol.get("per_column", {})
        if pc and max(pc.values(), default=0) > 0:
            wc = max(pc, key=pc.get)
            ins.append(f"📌 Variable con más outliers IQR: <b>'{wc}'</b> ({pc[wc]} registros).")

    for c in co[:5]:
        e = "📈" if c["direction"]=="positiva" else "📉"
        ins.append(f"{e} Correlación {c['strength']} <b>{c['direction']}</b>: <b>'{c['col1']}'</b> × <b>'{c['col2']}'</b> (r={c['pearson_r']}).")

    if cl:
        k, s = cl["best_k"], cl["silhouette"]
        q = "excelente" if s>.7 else "buena" if s>.5 else "aceptable" if s>.25 else "débil"
        ins.append(f"🔵 Se detectaron <b>{k} grupos naturales</b> en los datos numéricos (silhouette={s}, calidad {q}).")

    for c in nc[:3]:
        sk = df[c].dropna().skew()
        if abs(sk) > 1.5:
            d = "positiva (cola derecha)" if sk > 0 else "negativa (cola izquierda)"
            ins.append(f"📊 <b>'{c}'</b>: asimetría {d} (skew={sk:.2f}).")

    return ins, ale, inf

def make_report(df, vt, ds, ol, cl, co, ins, ale, inf, ch, cat_analysis):
    nc = [c for c,t in vt.items() if t=="numeric"]
    cat_cols = [c for c,t in vt.items() if t=="categorical"]
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    iso = ol.get("isolation_forest",{}) if ol else {}
    zr = ol.get("z_score",{}) if ol else {}
    iq = ol.get("iqr",{}) if ol else {}

    # Cat section
    cat_html = ""
    for col, data in cat_analysis.items():
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td><td>{data['pcts'][k]}%</td></tr>"
                       for k, v in data["counts"].items())
        cat_html += f"<div class='sec'><h2>🏷️ {col}</h2><table><tr><th>Valor</th><th>Cantidad</th><th>%</th></tr>{rows}</table></div>"

    cr_html = "".join(f"<tr><td>{c['col1']}</td><td>{c['col2']}</td><td style='color:#4caf50'>{c['pearson_r']}</td><td>{c['strength']} {c['direction']}</td></tr>" for c in co)
    ch_html = "".join(f"<div class='sec'><h2>{t}</h2>{h}</div>" for t,h in ch.items())
    cl_html = ""
    if cl:
        p = cl["profiles"].round(3).copy(); p.index = [f"Grupo {i+1}" for i in p.index]
        cl_html = f"<div class='sec'><h2>🔵 Clustering</h2><p>Grupos: <b>{cl['best_k']}</b> | Silhouette: <b>{cl['silhouette']}</b></p><div style='overflow-x:auto'>{p.to_html()}</div></div>"

    return f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>AutoAnalytics Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>body{{font-family:'Segoe UI',sans-serif;background:#0f1117;color:#e0e0e0;margin:0}}
.hdr{{background:linear-gradient(135deg,#1565c0,#0d47a1);padding:36px;text-align:center}}
.hdr h1{{color:white;font-size:2.4rem;margin:0}}.hdr p{{color:#bbdefb;margin:6px 0 0}}
.wrap{{max-width:1200px;margin:0 auto;padding:28px 20px}}
.sec{{background:#161b27;border-radius:12px;padding:22px;margin:18px 0;border:1px solid #1e2d40}}
.sec h2{{color:#4db8ff;border-bottom:2px solid #1e3a55;padding-bottom:8px}}
.i{{background:#1a2d1a;border-left:4px solid #4caf50;padding:10px 14px;margin:6px 0;border-radius:6px;color:#c8e6c9}}
.a{{background:#2d1a1a;border-left:4px solid #f44336;padding:10px 14px;margin:6px 0;border-radius:6px;color:#ffcdd2}}
.n{{background:#1a2233;border-left:4px solid #2196f3;padding:10px 14px;margin:6px 0;border-radius:6px;color:#bbdefb}}
.mr{{display:flex;gap:14px;flex-wrap:wrap}}.mt{{background:#1e2d40;border-radius:8px;padding:14px;flex:1;min-width:120px;text-align:center}}
.mt h3{{color:#4db8ff;font-size:1.7rem;margin:0}}.mt p{{color:#8ba3bb;margin:4px 0 0;font-size:.8rem}}
table{{width:100%;border-collapse:collapse;font-size:.83rem}}
th{{background:#1e3a55;color:#4db8ff;padding:8px}}td{{padding:7px;border-bottom:1px solid #1e2d40;color:#b0bec5}}
.ft{{text-align:center;padding:28px;color:#546e7a;font-size:.83rem}}</style></head><body>
<div class="hdr"><h1>🔬 AutoAnalytics</h1><p>Reporte de Análisis — {now}</p></div>
<div class="wrap">
<div class="sec"><h2>📋 Resumen</h2><div class="mr">
<div class="mt"><h3>{len(df):,}</h3><p>Registros</p></div>
<div class="mt"><h3>{len(df.columns)}</h3><p>Variables</p></div>
<div class="mt"><h3>{len(nc)}</h3><p>Numéricas</p></div>
<div class="mt"><h3>{len(cat_cols)}</h3><p>Categóricas</p></div>
<div class="mt"><h3>{df.isnull().sum().sum()}</h3><p>Faltantes</p></div>
</div></div>
<div class="sec"><h2>🚨 Alertas e Insights</h2>
{"".join(f'<div class="a">{x}</div>' for x in ale) or '<p>✅ Sin alertas.</p>'}
{"".join(f'<div class="i">{x}</div>' for x in ins)}
{"".join(f'<div class="n">{x}</div>' for x in inf)}
</div>
{cat_html}
<div class="sec"><h2>📊 Estadísticas Numéricas</h2><div style="overflow-x:auto">{ds.to_html(index=False) if not ds.empty else "<p>Sin datos numéricos.</p>"}</div></div>
<div class="sec"><h2>🔗 Correlaciones</h2>{"<table><tr><th>Var 1</th><th>Var 2</th><th>r Pearson</th><th>Relación</th></tr>"+cr_html+"</table>" if co else "<p>Sin correlaciones fuertes.</p>"}</div>
<div class="sec"><h2>⚠️ Outliers</h2><div class="mr">
<div class="mt"><h3>{zr.get('count',0)}</h3><p>Z-Score</p></div>
<div class="mt"><h3>{iq.get('count',0)}</h3><p>IQR</p></div>
<div class="mt"><h3>{iso.get('count',0)}</h3><p>IForest</p></div>
</div></div>
{cl_html}{ch_html}
<div class="ft"><p>Generado por <b>AutoAnalytics</b> — {now}</p></div>
</div></body></html>"""

# Generación del archivo reporte en formato JSON.
def make_json_report(df, vt, ds, ol, corrs, ins, ale, inf):
    """Genera un resumen del análisis exploratorio en formato JSON interpretable."""
    nc = [c for c, t in vt.items() if t == "numeric"]
    cat_cols = [c for c, t in vt.items() if t == "categorical"]

    # Limpiar etiquetas HTML de los insights (quitar <b>, etc.)
    import re
    def clean(text): return re.sub(r"<[^>]+>", "", text)

    reporte = {
        "resumen_dataset": {
            "registros": len(df),
            "variables": len(df.columns),
            "numericas": len(nc),
            "categoricas": len(cat_cols),
            "valores_faltantes": int(df.isnull().sum().sum())
        },
        "estadisticas_descriptivas": ds.to_dict(orient="records") if not ds.empty else [],
        "correlaciones_fuertes": corrs if corrs else [],
        "outliers": {
            "z_score": ol.get("z_score", {}) if ol else {},
            "iqr": ol.get("iqr", {}) if ol else {},
            "isolation_forest": {
                k: v for k, v in ol.get("isolation_forest", {}).items() if k != "indices"
            } if ol else {}
        },
        "insights": [clean(x) for x in ins],
        "alertas": [clean(x) for x in ale],
        "informacion": [clean(x) for x in inf],
        "generado": datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    }
    return json.dumps(reporte, ensure_ascii=False, indent=2)

# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    st.markdown("""<div style="background:linear-gradient(135deg,#1565c0,#0d47a1,#1a237e);
    padding:28px;border-radius:16px;text-align:center;margin-bottom:22px">
    <h1 style="color:white;font-size:2.2rem;margin:0;font-weight:800">🔬 AutoAnalytics</h1>
    <p style="color:#bbdefb;font-size:.98rem;margin:8px 0 0">
    Análisis Automático · Cualquier tipo de datos · Multi-hoja Excel · IA</p>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Configuración")

        # Drag & drop tip
        st.markdown("""<div class="wbox" style="font-size:.78rem;margin-bottom:8px">
        💡 <b>Mac/Safari:</b> Si el arrastre no funciona, usa el botón <b>"Browse files"</b> o abre la app en <b>Chrome</b>.
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "📂 Cargar datos (CSV o Excel)",
            type=["csv","xlsx","xls"],
            accept_multiple_files=True,
            help="Soporta CSV, XLSX, XLS — incluso con encabezados en filas intermedias"
        )
        use_demo = st.button("🎲 Datos de demostración", use_container_width=True)
        st.markdown("---")
        st.markdown("### 🔧 Parámetros")
        max_k = st.slider("Máx. clusters", 2, 10, 6) # Num. Max de clusters (agrupamiento)
        corr_thr = st.slider("Umbral correlación", 0.3, 0.95, 0.6, 0.05)
        cont = st.slider("Contaminación outliers", 0.01, 0.20, 0.05, 0.01)
        st.markdown("---")
        

    # ── Carga de datos ────────────────────────────────────────────────────────
    df = None; fname = "dataset"; sheet_name_used = None; header_row_used = 0
    all_sheets = []

    if use_demo:
        df = make_demo() #Aquí llamamos al demo, en caso de que no se suba ningún dato; fname = "demo_clientes"
        st.markdown('<div class="ifbox">📊 Datos de demostración: 500 clientes con variables financieras.</div>', unsafe_allow_html=True)

    elif uploaded:
        # Recopilar todos los sheets de todos los archivos
        options = []
        all_loaded = {}
        for f in uploaded:
            try:
                if f.name.endswith(".csv"):
                    loaded = pd.read_csv(f)
                    key = f.name
                    all_loaded[key] = {"df": loaded, "header_row": 0, "sheet": None}
                    options.append(key)
                else:
                    xl = pd.ExcelFile(f)
                    sheets = xl.sheet_names
                    for sh in sheets:
                        f.seek(0)
                        loaded, hrow, dupes = smart_read_excel(f, sheet_name=sh)
                        if dupes > 0:
                            st.warning(f"⚠️ Se eliminaron {dupes} filas duplicadas.")
                        key = f"{f.name} — {sh}" if len(sheets) > 1 else f.name
                        all_loaded[key] = {"df": loaded, "header_row": hrow, "sheet": sh}
                        options.append(key)
            except Exception as e:
                st.error(f"Error cargando {f.name}: {e}")

        if options:
            if len(options) > 1:
                selected = st.selectbox("📂 Seleccionar dataset / hoja:", options)
            else:
                selected = options[0]

            info = all_loaded[selected]
            df = info["df"]
            fname = selected.replace(" — ", "_").replace(".xlsx","").replace(".csv","")
            header_row_used = info["header_row"]
            all_sheets = options

            if header_row_used > 0:
                st.markdown(f'<div class="ifbox">ℹ️ Encabezados detectados en la fila <b>{header_row_used+1}</b> del archivo.</div>', unsafe_allow_html=True)
            if len(options) > 1 and len([o for o in options if selected.split(" — ")[0] in o]) > 1:
                st.markdown(f'<div class="ifbox">📑 El archivo tiene <b>múltiples hojas</b>. Selecciona la que quieras analizar arriba.</div>', unsafe_allow_html=True)

    # ── Pantalla de bienvenida ────────────────────────────────────────────────
    if df is None:
        st.markdown("""<div style="background:#161b27;border:2px dashed #1e3a55;border-radius:16px;
        padding:52px;text-align:center;margin:32px 0">
        <div style="font-size:3rem">📂</div>
        <h2 style="color:#4db8ff">Carga tu dataset para comenzar</h2>
        <p style="color:#8ba3bb">Sube tu archivo desde el panel izquierdo con <b>"Browse files"</b>,<br>
        o usa los <b>datos de demostración</b> para explorar la app.</p>
        <div style="display:flex;justify-content:center;gap:14px;margin-top:16px;flex-wrap:wrap">
        <div style="background:#1e2d40;padding:9px 16px;border-radius:8px;color:#b0bec5">📊 CSV</div>
        <div style="background:#1e2d40;padding:9px 16px;border-radius:8px;color:#b0bec5">📗 XLSX multi-hoja</div>
        <div style="background:#1e2d40;padding:9px 16px;border-radius:8px;color:#b0bec5">📘 XLS</div>
        </div></div>""", unsafe_allow_html=True)
        return

    # ── Análisis ──────────────────────────────────────────────────────────────
    vt = detect_types(df)
    # Conversión forzada de tipos incorrectos
    for col in df.columns:
        if vt.get(col) == "numeric" and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif vt.get(col) == "temporal" and df[col].dtype == object:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    nc = [c for c,t in vt.items() if t=="numeric"]
    cat_cols = [c for c,t in vt.items() if t=="categorical"]
    temp_cols = [c for c,t in vt.items() if t=="temporal"]

    #En este módulo se realiza la ejecución de los datos usando el streamlit que nos permite aplicar ML en aplicaciones.
    with st.spinner("⚡ Analizando datos..."):
        dstats = do_stats(df, nc) # Estadísticas descriptivas
        P, S, corrs = do_corr(df, nc, corr_thr) # Matrices de correlación
        ol = do_outliers(df, nc, cont) if len(nc) >= 1 else {} # Detección de outliers
        cl = do_kmeans(df, nc, max_k) if len(nc) >= 2 else None # Clustering
        cat_analysis = analyze_categorical(df, cat_cols) # Análisis categórico
        ins, ale, inf = auto_insights(df, vt, ol, cl, corrs, cat_analysis) # Generación de insights

    # ── Métricas superiores ───────────────────────────────────────────────────
    cols = st.columns(6)
    mets = [(len(df),"Registros","📝"),(len(df.columns),"Variables","🗂️"),
            (len(nc),"Numéricas","🔢"),(len(cat_cols),"Categóricas","🏷️"),
            (df.isnull().sum().sum(),"Faltantes","❓"),
            (cl["best_k"] if cl else len(temp_cols) or "—","Clusters/Temp","📊")]
    for col,(val,lbl,ico) in zip(cols,mets):
        with col:
            st.markdown(f'<div class="mcard"><div style="font-size:1.3rem">{ico}</div>'
                f'<h2>{val if isinstance(val,str) else f"{val:,}"}</h2><p>{lbl}</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Insights ──────────────────────────────────────────────────────────────
    with st.expander("🔍 **Insights y Alertas Automáticas**", expanded=True):
        if not ale and not ins and not inf:
            st.info("Carga más datos para obtener insights.")
        for a in ale: st.markdown(f'<div class="abox">{a}</div>', unsafe_allow_html=True)
        for i in ins: st.markdown(f'<div class="ibox">{i}</div>', unsafe_allow_html=True)
        for i in inf: st.markdown(f'<div class="ifbox">{i}</div>', unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_labels = ["🏷️ Categóricas","📊 Numéricas","🔗 Correlaciones",
                  "⚠️ Outliers","🔵 Clustering","🔄 Relaciones","💾 Exportar"]
    tabs = st.tabs(tab_labels)
    ch_exp = {}

    # ══ TAB 1: CATEGORICAL (el más importante para este tipo de datos) ════════ DE CATEGORIAS, MUESTRA CONTEO DE CATEGORIAS, GRAFICOS, TABLA CRUZADA, MAPA DE CALOR. USA PLOTLY
    with tabs[0]:
        st.markdown('<div class="stitle">🏷️ Análisis de Variables Categóricas</div>', unsafe_allow_html=True)

        if not cat_cols:
            st.info("No se detectaron variables categóricas en este dataset.")
        else:
            # KPI cards para cada columna categórica
            for col in cat_cols:
                data = cat_analysis[col]
                counts = data["counts"]

                st.markdown(f"#### 📌 {col}")
                vc_df = pd.DataFrame({"Valor": counts.index, "Cantidad": counts.values,
                                       "%": data["pcts"].values})

                col_left, col_right = st.columns([1, 2])
                with col_left:
                    # Mini KPIs
                    for val, cnt in counts.head(6).items():
                        pct = round(cnt/len(df)*100, 1)
                        bar_w = int(pct)
                        # Color según palabras clave
                        v_lower = str(val).lower()
                        if any(k in v_lower for k in ["pass","aprobado","ok","activo","sí","si","exitoso"]):
                            color = "#4caf50"
                        elif any(k in v_lower for k in ["fail","error","rechazado","no","inactivo","cancelado"]):
                            color = "#f44336"
                        elif any(k in v_lower for k in ["alta","high","crítica","urgente"]):
                            color = "#ff5722"
                        elif any(k in v_lower for k in ["media","medium","moderada"]):
                            color = "#ff9800"
                        elif any(k in v_lower for k in ["baja","low"]):
                            color = "#2196f3"
                        else:
                            color = "#4db8ff"
                        st.markdown(f"""<div style="background:#1e2d40;border-radius:8px;padding:10px 14px;margin:5px 0">
                        <div style="display:flex;justify-content:space-between;align-items:center">
                            <span style="color:#e0e0e0;font-size:.9rem"><b>{val}</b></span>
                            <span style="color:{color};font-weight:700">{cnt} <small style="color:#8ba3bb">({pct}%)</small></span>
                        </div>
                        <div style="background:#0f1117;border-radius:3px;height:5px;margin-top:6px">
                            <div style="background:{color};width:{bar_w}%;height:100%;border-radius:3px"></div>
                        </div></div>""", unsafe_allow_html=True)

                with col_right:
                    # Gráfico de torta o barras según cantidad de valores
                    if len(counts) <= 6:
                        fig = px.pie(vc_df, values="Cantidad", names="Valor",
                                     title=f"Distribución de {col}",
                                     template="plotly_dark",
                                     color_discrete_sequence=COLORS,
                                     hole=0.35)
                        fig.update_traces(textposition="inside", textinfo="percent+label")
                    else:
                        fig = px.bar(vc_df.head(15), x="Valor", y="Cantidad",
                                     text="%", title=f"Top valores en {col}",
                                     template="plotly_dark",
                                     color="Cantidad", color_continuous_scale="Blues")
                        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig.update_layout(paper_bgcolor="#161b27", plot_bgcolor="#1a2332",
                                      height=300, margin=dict(t=40,b=20,l=20,r=20),
                                      showlegend=len(counts)<=6,
                                      font=dict(color="#e0e0e0"))
                    st.plotly_chart(fig, use_container_width=True)
                    ch_exp[f"🏷️ {col}"] = fig.to_html(include_plotlyjs=False, full_html=False)

                st.markdown("---")

            # Tabla cruzada entre dos categóricas
            if len(cat_cols) >= 2:
                st.markdown('<div class="stitle">📊 Tabla Cruzada entre Variables</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1: col_a = st.selectbox("Variable fila:", cat_cols, key="cross_a")
                with c2:
                    other = [c for c in cat_cols if c != col_a]
                    col_b = st.selectbox("Variable columna:", other, key="cross_b")

                ct = pd.crosstab(df[col_a], df[col_b], margins=True, margins_name="TOTAL")
                st.dataframe(ct.style.background_gradient(cmap="Blues"), use_container_width=True)

                # Heatmap de tabla cruzada
                ct_no_margins = pd.crosstab(df[col_a], df[col_b])
                fig_ct = px.imshow(ct_no_margins, text_auto=True, aspect="auto",
                                   title=f"Heatmap: {col_a} × {col_b}",
                                   template="plotly_dark",
                                   color_continuous_scale="Blues")
                fig_ct.update_layout(paper_bgcolor="#161b27", height=380, font=dict(color="#e0e0e0"))
                st.plotly_chart(fig_ct, use_container_width=True)
                ch_exp[f"📊 Cruce {col_a}×{col_b}"] = fig_ct.to_html(include_plotlyjs=False, full_html=False)

            # Detalles de filas con issues/comentarios
            text_cols = [c for c,t in vt.items() if t in ("text","text_long")]
            comment_cols = [c for c in text_cols if any(k in c.lower() for k in ["comentario","comment","issue","nota","observacion"])]
            if comment_cols and cat_cols:
                st.markdown('<div class="stitle">🐛 Registros con Comentarios / Issues</div>', unsafe_allow_html=True)
                sel_com = st.selectbox("Columna de comentarios:", comment_cols)
                con_comentario = df[df[sel_com].notna()].copy()
                if len(con_comentario) > 0:
                    st.markdown(f'<div class="abox">Se encontraron <b>{len(con_comentario)} registros</b> con comentarios en <b>{sel_com}</b>.</div>', unsafe_allow_html=True)
                    show_cols = [c for c in df.columns if vt.get(c) in ("categorical","text_id")] + [sel_com]
                    st.dataframe(con_comentario[show_cols].reset_index(drop=True), use_container_width=True)
                else:
                    st.success("No hay registros con comentarios.")

    # ══ TAB 2: ANALISIS DE VARIABLES NUMERICAS ════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown('<div class="stitle">📊 Análisis de Variables Numéricas</div>', unsafe_allow_html=True)
        if not nc:
            st.info("No se detectaron variables numéricas en este dataset.")
        else:
            st.dataframe(dstats.style.format(precision=4).background_gradient(subset=["Media","Desv.Std"], cmap="Blues"),
                         use_container_width=True, height=400)

            # Histogramas
            np2 = min(3, len(nc)); nr2 = (len(nc)+np2-1)//np2
            fig = make_subplots(rows=nr2, cols=np2, subplot_titles=nc)
            for i, c in enumerate(nc):
                fig.add_trace(go.Histogram(x=df[c].dropna(), name=c, marker_color=COLORS[i%len(COLORS)],
                                           opacity=.75, showlegend=False), row=i//np2+1, col=i%np2+1)
            fig.update_layout(template="plotly_dark", paper_bgcolor="#161b27", plot_bgcolor="#1a2332",
                              height=max(260, nr2*200), font=dict(color="#e0e0e0"))
            st.plotly_chart(fig, use_container_width=True)
            ch_exp["📊 Histogramas"] = fig.to_html(include_plotlyjs=False, full_html=False)

            # Boxplots
            fig2 = go.Figure()
            for i, c in enumerate(nc):
                fig2.add_trace(go.Box(y=df[c].dropna(), name=c, marker_color=COLORS[i%len(COLORS)], boxmean="sd"))
            fig2.update_layout(template="plotly_dark", paper_bgcolor="#161b27", plot_bgcolor="#1a2332",
                               height=380, font=dict(color="#e0e0e0"), title="Boxplots")
            st.plotly_chart(fig2, use_container_width=True)
            ch_exp["📦 Boxplots"] = fig2.to_html(include_plotlyjs=False, full_html=False)

        # Temporales
        if temp_cols:
            st.markdown('<div class="stitle">📅 Variables Temporales</div>', unsafe_allow_html=True)
            for col in temp_cols:
                st.markdown(f'<div class="ifbox">📅 <b>{col}</b>: rango {df[col].min()} → {df[col].max()} | {df[col].nunique()} fechas únicas</div>', unsafe_allow_html=True)

        # Vista de tipos
        st.markdown('<div class="stitle">🗂️ Tipos de Variables Detectados</div>', unsafe_allow_html=True)
        icons = {"numeric":"🔢","categorical":"🏷️","temporal":"📅","boolean":"✅","text":"📝","text_long":"📄","text_id":"🔑","constant":"⚪"}
        tdf = pd.DataFrame([{"Variable":c,"Tipo":f"{icons.get(t,'❓')} {t}","Únicos":df[c].nunique(),
            "Faltantes":df[c].isnull().sum(),"Completo%":f"{(1-df[c].isnull().mean())*100:.1f}%"} for c,t in vt.items()])
        st.dataframe(tdf, use_container_width=True)
        with st.expander("👁️ Vista previa del dataset"):
            st.dataframe(df.head(50), use_container_width=True)

    # ══ TAB 3: CORRELACIONES ═══════════════════════════════════════════════════
    with tabs[2]:
        if P is not None and len(nc) >= 2:
            cl2, cr2 = st.columns([2,1])
            with cl2:
                m = st.radio("Método", ["Pearson","Spearman"], horizontal=True)
                cm = P if m=="Pearson" else S
                fig = go.Figure(go.Heatmap(z=cm.values, x=cm.columns.tolist(), y=cm.columns.tolist(),
                    colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                    text=cm.round(2).values, texttemplate="%{text}", textfont={"size":9},
                    hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.4f}<extra></extra>"))
                fig.update_layout(title=f"Correlación ({m})", template="plotly_dark",
                    paper_bgcolor="#161b27", plot_bgcolor="#161b27", height=480, font=dict(color="#e0e0e0"))
                st.plotly_chart(fig, use_container_width=True)
                ch_exp["🔗 Mapa de Calor"] = fig.to_html(include_plotlyjs=False, full_html=False)
            with cr2:
                st.markdown("**Correlaciones fuertes**")
                if corrs:
                    for c in corrs[:10]:
                        r = c["pearson_r"]; cc = "#4caf50" if abs(r)>=.7 else "#ff9800"
                        st.markdown(f"""<div style="background:#1e2d40;border-radius:8px;padding:9px;margin:4px 0">
                        <div style="font-size:.8rem;color:#e0e0e0"><b>{c['col1']}</b> × <b>{c['col2']}</b></div>
                        <div style="background:#0f1117;border-radius:3px;height:5px;margin:4px 0">
                        <div style="background:{cc};width:{int(abs(r)*100)}%;height:100%"></div></div>
                        <div style="font-size:.75rem;color:{cc}">r={r} · {c['strength']} {c['direction']}</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info(f"Sin correlaciones ≥ {corr_thr}")
        else:
            st.info("Se requieren ≥ 2 variables numéricas para correlaciones.")

    # ══ TAB 4: OUTLIERS ═══════════════════════════════════════════════════════
    with tabs[3]:
        if ol:
            zr=ol.get("z_score",{}); iq=ol.get("iqr",{}); iso=ol.get("isolation_forest",{})
            c1,c2,c3 = st.columns(3)
            for col3,(lbl,res,color,ico) in zip([c1,c2,c3],[
                ("Z-Score (|z|>3)",zr,"#2196f3","📏"),("IQR (1.5×)",iq,"#ff9800","📦"),("Isolation Forest",iso,"#f44336","🌲")]):
                with col3:
                    st.markdown(f'<div class="mcard" style="border-top:3px solid {color}"><div style="font-size:1.3rem">{ico}</div>'
                        f'<h2 style="color:{color}">{res.get("count",0)}</h2>'
                        f'<p>{lbl}<br><small>{res.get("pct",0)}% del total</small></p></div>', unsafe_allow_html=True)
            if ol.get("per_column"):
                pc = ol["per_column"]
                pcd = pd.DataFrame(list(pc.items()), columns=["Variable","Outliers IQR"])
                pcd["%"] = (pcd["Outliers IQR"]/len(df)*100).round(2)
                pcd = pcd.sort_values("Outliers IQR", ascending=False)
                fig4 = px.bar(pcd, x="Variable", y="Outliers IQR", text="%",
                    template="plotly_dark", color="Outliers IQR", color_continuous_scale="Reds")
                fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig4.update_layout(paper_bgcolor="#161b27", plot_bgcolor="#1a2332",
                    height=360, showlegend=False, font=dict(color="#e0e0e0"), xaxis=dict(tickangle=45))
                st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Se necesitan variables numéricas para detectar outliers.")

    # ══ TAB 5: CLUSTERING ══════════════════════════════════════════════════════
    with tabs[4]:
        if cl:
            c1,c2,c3 = st.columns(3)
            with c1: st.markdown(f'<div class="mcard" style="border-top:3px solid #4db8ff"><div style="font-size:1.3rem">🔵</div><h2 style="color:#4db8ff">{cl["best_k"]}</h2><p>Clusters óptimos</p></div>', unsafe_allow_html=True)
            with c2:
                sc=cl["silhouette"]; sc_c="#4caf50" if sc>.5 else "#ff9800" if sc>.25 else "#f44336"
                sc_l="Excelente" if sc>.7 else "Buena" if sc>.5 else "Aceptable" if sc>.25 else "Débil"
                st.markdown(f'<div class="mcard" style="border-top:3px solid {sc_c}"><div style="font-size:1.3rem">📏</div><h2 style="color:{sc_c}">{sc}</h2><p>Silhouette ({sc_l})</p></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="mcard" style="border-top:3px solid #ab47bc"><div style="font-size:1.3rem">📊</div><h2 style="color:#ab47bc">{len(cl["k_range"])}</h2><p>k evaluados</p></div>', unsafe_allow_html=True)

            cel, cpca = st.columns(2)
            with cel:
                fe = make_subplots(rows=1, cols=2, subplot_titles=["Codo","Silhouette"])
                fe.add_trace(go.Scatter(x=cl["k_range"],y=cl["inertias"],mode="lines+markers",
                    line=dict(color="#4db8ff",width=2),marker=dict(size=7),showlegend=False),row=1,col=1)
                fe.add_trace(go.Scatter(x=cl["k_range"],y=cl["silhouettes"],mode="lines+markers",
                    line=dict(color="#4caf50",width=2),marker=dict(size=7),showlegend=False),row=1,col=2)
                fe.add_vline(x=cl["best_k"],line_dash="dash",line_color="#f44336",row=1,col=1)
                fe.add_vline(x=cl["best_k"],line_dash="dash",line_color="#f44336",row=1,col=2)
                fe.update_layout(template="plotly_dark",paper_bgcolor="#161b27",plot_bgcolor="#1a2332",height=280,font=dict(color="#e0e0e0"))
                st.plotly_chart(fe, use_container_width=True)
            with cpca:
                fp = px.scatter(cl["pca_df"],x="PC1",y="PC2",color="Cluster",opacity=.75,
                    template="plotly_dark",color_discrete_sequence=COLORS,title=f"Clusters PCA (k={cl['best_k']})")
                fp.update_layout(paper_bgcolor="#161b27",plot_bgcolor="#1a2332",height=280,font=dict(color="#e0e0e0"))
                st.plotly_chart(fp, use_container_width=True)
                ch_exp["🔵 PCA Clusters"] = fp.to_html(include_plotlyjs=False, full_html=False)

            st.markdown('<div class="stitle">📋 Perfiles de Clusters</div>', unsafe_allow_html=True)
            pr = cl["profiles"].round(4).copy(); pr.index = [f"Cluster {i+1}" for i in pr.index]
            st.dataframe(pr.style.background_gradient(cmap="Blues",axis=0), use_container_width=True)
        else:
            st.info("Se requieren ≥ 2 variables numéricas y ≥ 15 registros para clustering.")

    # ══ TAB 6: RELACIONES ══════════════════════════════════════════════════
    with tabs[5]:
        if len(nc) >= 2:
            st.markdown('<div class="stitle">🔄 Dispersión entre Numéricas</div>', unsafe_allow_html=True)
            c1,c2,c3 = st.columns(3)
            with c1: vx = st.selectbox("Var X:", nc, 0, key="rx")
            with c2: vy = st.selectbox("Var Y:", nc, min(1,len(nc)-1), key="ry")
            with c3: vc3 = st.selectbox("Color:", ["(ninguno)"]+cat_cols, key="rc")
            pd2 = df[[vx,vy]+([vc3] if vc3!="(ninguno)" else [])].dropna()
            rv, pv = stats.pearsonr(pd2[vx].values, pd2[vy].values)
            fs = px.scatter(pd2,x=vx,y=vy,color=vc3 if vc3!="(ninguno)" else None,trendline="ols",
                opacity=.65,title=f"{vx} vs {vy}  |  r={rv:.4f}  |  p={pv:.4f}",
                template="plotly_dark",color_discrete_sequence=COLORS)
            fs.update_layout(paper_bgcolor="#161b27",plot_bgcolor="#1a2332",height=440,font=dict(color="#e0e0e0"))
            st.plotly_chart(fs, use_container_width=True)
            r2 = rv**2
            sig = "✅ Significativa (p<0.05)" if pv<.05 else "❌ No significativa"
            st.markdown(f'<div class="ifbox"><b>Pearson:</b> r={rv:.4f} | r²={r2:.4f} | {sig}<br>'
                f'<b>{vx}</b> explica el <b>{r2*100:.1f}%</b> de la varianza en <b>{vy}</b>.</div>', unsafe_allow_html=True)

        if cat_cols and nc:
            st.markdown('<div class="stitle">📊 Categórica × Numérica</div>', unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            with c1: cs = st.selectbox("Categórica:", cat_cols, key="cn1")
            with c2: ns = st.selectbox("Numérica:", nc, key="cn2")
            tops = df[cs].value_counts().head(10).index.tolist()
            flt = df[df[cs].isin(tops)][[cs,ns]].dropna()
            fv = px.violin(flt,x=cs,y=ns,box=True,points="all",color=cs,
                template="plotly_dark",color_discrete_sequence=COLORS)
            fv.update_layout(paper_bgcolor="#161b27",plot_bgcolor="#1a2332",height=420,
                showlegend=False,xaxis=dict(tickangle=30),font=dict(color="#e0e0e0"))
            st.plotly_chart(fv, use_container_width=True)

        if len(cat_cols) >= 1 and not nc:
            st.info("Solo hay variables categóricas. Los análisis de relación están en la pestaña 🏷️ Categóricas.")

    # ══ TAB 7: EXPORTAR ═════════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown('<div class="stitle">💾 Exportar Resultados</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="ifbox"><b>📄 Reporte HTML</b><br>Incluye todos los gráficos, tablas, insights y análisis categórico.</div>', unsafe_allow_html=True)
            if st.button("🔨 Generar Reporte HTML", use_container_width=True):
                with st.spinner("Generando..."):
                    html = make_report(df,vt,dstats,ol,cl,corrs,ins,ale,inf,ch_exp,cat_analysis)
                    b64 = base64.b64encode(html.encode()).decode()
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                    st.markdown(f'<a href="data:text/html;base64,{b64}" download="AutoAnalytics_{fname}_{ts}.html" '
                        f'style="display:block;background:linear-gradient(135deg,#1565c0,#0d47a1);color:white;'
                        f'text-align:center;padding:14px;border-radius:8px;text-decoration:none;font-weight:600;margin-top:10px">'
                        f'⬇️ Descargar Reporte HTML</a>', unsafe_allow_html=True)
                    st.success("✅ Reporte generado.")
        with c2:
            st.markdown('<div class="ifbox"><b>📊 Datos CSV</b><br>Exporta el dataset procesado como CSV.</div>', unsafe_allow_html=True)
            cb = base64.b64encode(df.to_csv(index=False).encode()).decode()
            st.markdown(f'<a href="data:text/csv;base64,{cb}" download="datos_{fname}.csv" '
                f'style="display:block;background:linear-gradient(135deg,#2e7d32,#1b5e20);color:white;'
                f'text-align:center;padding:14px;border-radius:8px;text-decoration:none;font-weight:600;margin-top:10px">'
                f'⬇️ Descargar CSV</a>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="ifbox"><b>🗂️ Reporte JSON</b><br>Exporta estadísticas, correlaciones e insights en formato JSON.</div>', unsafe_allow_html=True)
            if st.button("🔨 Generar Reporte JSON", use_container_width=True):
                json_str = make_json_report(df, vt, dstats, ol, corrs, ins, ale, inf)
                jb64 = base64.b64encode(json_str.encode()).decode()
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                st.markdown(f'<a href="data:application/json;base64,{jb64}" download="AutoAnalytics_{fname}_{ts}.json" '
                    f'style="display:block;background:linear-gradient(135deg,#6a1b9a,#4a148c);color:white;'
                    f'text-align:center;padding:14px;border-radius:8px;text-decoration:none;font-weight:600;margin-top:10px">'
                    f'⬇️ Descargar JSON</a>', unsafe_allow_html=True)
                st.success("✅ JSON generado.")
    

        # Resumen Final
        st.markdown('<div class="stitle">📋 Resumen Final</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🔍 Hallazgos:**")
            for i, x in enumerate(ins[:8], 1): st.markdown(f"{i}. {x}", unsafe_allow_html=True)
        with c2:
            if ale:
                st.markdown("**🚨 Alertas:**")
                for a in ale: st.markdown(f"• {a}", unsafe_allow_html=True)
            st.markdown("**📊 Métricas:**")
            st.markdown(f"• **{len(df):,}** registros × **{len(df.columns)}** variables")
            st.markdown(f"• **{len(cat_cols)}** categóricas · **{len(nc)}** numéricas")
            if cl: st.markdown(f"• Clusters: **{cl['best_k']}** (silhouette: {cl['silhouette']})")
            if corrs: st.markdown(f"• Correlaciones fuertes: **{len(corrs)}**")

if __name__ == "__main__":
    main()

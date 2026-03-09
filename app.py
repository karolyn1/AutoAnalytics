"""
AutoAnalytics - Análisis Inteligente de Datos
Analiza automáticamente cualquier tipo de datos: numéricos, categóricos, texto
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats
import io, base64, json, datetime, warnings
warnings.filterwarnings("ignore")


# Aquí se realiza la configuración de la página con streamlit
st.set_page_config(page_title="AutoAnalytics", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")


st.markdown("""
<style>
.stApp {
    background: #0f1117;
}

section[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #1e2d40;
}

.mcard {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(77, 184, 255, 0.18);
    border-radius: 20px;
    padding: 18px 14px;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    min-height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
}

.mcard:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 28px rgba(0, 0, 0, 0.28);
}

.mcard .micon {
    width: 52px;
    height: 52px;
    margin-bottom: 14px;
    border-radius: 14px;
    background: rgba(77, 184, 255, 0.12);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #4db8ff;
    font-size: 1.25rem;
}

.mcard h2 {
    margin: 0;
    color: #f8fafc;
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1.2;
}

.mcard p {
    margin: 10px 0 0 0;
    color: #94a3b8;
    font-size: 0.95rem;
}

.ibox {
    background: #1a2d1a;
    border-left: 4px solid #4caf50;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    color: #c8e6c9;
    font-size: 0.9rem;
}

.abox {
    background: #2d1a1a;
    border-left: 4px solid #f44336;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    color: #ffcdd2;
    font-size: 0.9rem;
}

.ifbox {
    background: #1a2233;
    border-left: 4px solid #2196f3;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    color: #bbdefb;
    font-size: 0.9rem;
}

.wbox {
    background: #2d2a1a;
    border-left: 4px solid #ff9800;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    color: #ffe0b2;
    font-size: 0.9rem;
}

.stitle {
    font-size: 1.3rem;
    font-weight: 700;
    color: #4db8ff;
    border-bottom: 2px solid #1e3a55;
    padding-bottom: 6px;
    margin: 20px 0 14px;
}

.stTabs [data-baseweb="tab-list"] {
    background: #161b27;
    border-radius: 8px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    color: #8ba3bb;
    border-radius: 6px;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #1e3a55;
    color: #4db8ff;
}

h1, h2, h3 {
    color: #e0e0e0;
}

p, li {
    color: #b0bec5;
}

div[data-testid="stFileUploadDropzone"] {
    background: #161b27 !important;
    border: 2px dashed #2a3f55 !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)
COLORS = px.colors.qualitative.Plotly

# Esto es para realizar un demo, y que la app muestre datos aunque no se cargue nada

def make_demo():
    np.random.seed(42); n=500
    d = pd.DataFrame({
        "edad": np.random.normal(38,12,n).astype(int).clip(18,75),
        "ingreso": np.random.lognormal(8.5,.6,n).round(2),
        "gasto": np.random.lognormal(7.8,.5,n).round(2),
        "antiguedad_meses": np.random.exponential(36,n).astype(int).clip(1,240),
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

def smart_read_excel(file_obj, sheet_name=0):
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
    return df, best_row

def detect_types(df):
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

def analyze_categorical(df, cat_cols):
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

def do_corr(df, nc, thr=.6):
    if len(nc) < 2: return None, None, []
    P = df[nc].corr("pearson"); S = df[nc].corr("spearman")
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

def auto_insights(df, vt, ol, cl, co, cat_analysis):
    ins, ale, inf = [], [], []
    nc = [c for c,t in vt.items() if t=="numeric"]
    cat_cols = [c for c,t in vt.items() if t=="categorical"]
    n, m = df.shape
    mp = df.isnull().sum().sum()/(n*m)*100

    inf.append(f"📊 Dataset: <b>{n:,} registros × {m} variables</b> | {len(nc)} numéricas, {len(cat_cols)} categóricas.")
    if mp > 0:
        (ale if mp > 20 else inf).append(f"⚠️ <b>{mp:.1f}%</b> de valores faltantes en el dataset.")

    for col, data in cat_analysis.items():
        counts = data["counts"]
        n_vals = len(counts)
        col_lower = col.lower()

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

        elif any(kw in col_lower for kw in ["prioridad", "priority", "nivel", "level", "urgencia"]):
            high_keywords = ["alta", "high", "crítica", "critical", "urgente"]
            high_count = sum(v for k, v in counts.items() if any(kw in str(k).lower() for kw in high_keywords))
            if high_count > 0:
                ins.append(f"🔴 <b>'{col}'</b>: <b>{high_count} registros de alta prioridad</b> ({round(high_count/n*100,1)}% del total).")

        elif n_vals >= 2:
            top_pct = data["top_pct"]
            if top_pct >= 80:
                ins.append(f"📌 <b>'{col}'</b>: El valor <b>'{data['top']}'</b> domina con <b>{top_pct}%</b> de los registros.")
            elif n_vals <= 5:
                dist_str = " | ".join([f"{k}: {v}" for k, v in counts.head(5).items()])
                inf.append(f"🏷️ <b>'{col}'</b>: {dist_str}")

    for col, t in vt.items():
        if t in ("text", "text_long") or "comentario" in col.lower() or "comment" in col.lower() or "issue" in col.lower():
            filled = df[col].dropna()
            if len(filled) > 0:
                issue_kw = ["issue", "error", "problema", "fallo", "reportado", "bug"]
                issue_count = sum(1 for v in filled if any(kw in str(v).lower() for kw in issue_kw))
                if issue_count > 0:
                    ale.append(f"🐛 Se detectaron <b>{issue_count} registros con issues/problemas reportados</b> en la columna <b>'{col}'</b>.")

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

def _plotly_png_bytes(fig, width=2400, height=1350, scale=2):
    # Exporta la figura Plotly a PNG con proporción horizontal (16:9)
    # Requiere kaleido instalado en el MISMO Python que corre Streamlit.
    try:
        return fig.to_image(
            format="png",
            width=width,
            height=height,
            scale=scale,
            engine="kaleido"
        )
    except Exception as e:
        raise RuntimeError(
            "No se pudo exportar la imagen con Kaleido. "
            "Asegúrate de instalarlo en la TERMINAL (no dentro de python):\n"
            "py -m pip install -U kaleido\n"
            "y luego reinicia streamlit."
        ) from e


def make_pdf_report_plotly(pages):
    """
    Genera un PDF en HORIZONTAL (landscape) y coloca cada gráfico sin estirarlo.
    """
    import io
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()

    # Tamaño carta en horizontal: 11 x 8.5
    FIG_W, FIG_H = 11, 8.5

    with PdfPages(buf) as pdf:
        for p in pages:
            name = p.get("name", "Gráfico")
            fig_plotly = p.get("fig")

            if fig_plotly is None:
                continue

            img_bytes = _plotly_png_bytes(fig_plotly, width=2400, height=1350, scale=2)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            fig = plt.figure(figsize=(FIG_W, FIG_H))
            ax = fig.add_subplot(111)
            ax.axis("off")

            # Margen y título
            fig.suptitle(name, fontsize=14, fontweight="bold", y=0.98)

            # Área útil (evita que el título choque)
            ax.set_position([0.03, 0.06, 0.94, 0.86])

            # Mostrar sin deformar (mantiene proporción)
            ax.imshow(img, aspect="equal")

            pdf.savefig(fig)
            plt.close(fig)

    buf.seek(0)
    return buf.read()
def make_report(df, vt, ds, ol, cl, co, ins, ale, inf, ch, cat_analysis):
    nc = [c for c,t in vt.items() if t=="numeric"]
    cat_cols = [c for c,t in vt.items() if t=="categorical"]
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    iso = ol.get("isolation_forest",{}) if ol else {}
    zr = ol.get("z_score",{}) if ol else {}
    iq = ol.get("iqr",{}) if ol else {}

    cat_html = ""
    for col, data in cat_analysis.items():
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td><td>{data['pcts'][k]}%</td></tr>"
                       for k, v in data["counts"].items())
        cat_html += f"<div class='sec'><h2>🏷️ {col}</h2><table><tr><th>Valor</th><th>Cantidad</th><th>%</th></tr>{rows}</table></div>"

    cr_html = "".join(
        f"<tr><td>{c['col1']}</td><td>{c['col2']}</td><td style='color:#4caf50'>{c['pearson_r']}</td><td>{c['strength']} {c['direction']}</td></tr>"
        for c in co
    )
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
<div class="sec"><h2>Alertas e Insights</h2>
{"".join(f'<div class="a">{x}</div>' for x in ale) or '<p>✅ Sin alertas.</p>'}
{"".join(f'<div class="i">{x}</div>' for x in ins)}
{"".join(f'<div class="n">{x}</div>' for x in inf)}
</div>
{cat_html}
<div class="sec"><h2>Estadísticas Numéricas</h2><div style="overflow-x:auto">{ds.to_html(index=False) if not ds.empty else "<p>Sin datos numéricos.</p>"}</div></div>
<div class="sec"><h2>Correlaciones</h2>{"<table><tr><th>Var 1</th><th>Var 2</th><th>r Pearson</th><th>Relación</th></tr>"+cr_html+"</table>" if co else "<p>Sin correlaciones fuertes.</p>"}</div>
<div class="sec"><h2>Outliers</h2><div class="mr">
<div class="mt"><h3>{zr.get('count',0)}</h3><p>Z-Score</p></div>
<div class="mt"><h3>{iq.get('count',0)}</h3><p>IQR</p></div>
<div class="mt"><h3>{iso.get('count',0)}</h3><p>IForest</p></div>
</div></div>
{cl_html}{ch_html}
<div class="ft"><p>Generado por <b>AutoAnalytics</b> — {now}</p></div>
</div></body></html>"""

def main():
    # =========================
# ESTILOS
# =========================
    st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">

<style>
.main {
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

.side-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.side-subtitle {
    font-size: 0.95rem;
    font-weight: 600;
    color: #cbd5e1;
    margin-top: 14px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.side-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
}

.stFileUploader, .stSelectbox, .stSlider, .stButton {
    margin-bottom: 10px;
}

.mcard {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(77,184,255,0.18);
    border-radius: 20px;
    padding: 18px 14px;
    text-align: center;
    box-shadow: 0 10px 24px rgba(0,0,0,0.22);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    min-height: 140px;
}

.mcard:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 28px rgba(0,0,0,0.28);
}

.mcard .micon {
    width: 52px;
    height: 52px;
    margin: 0 auto 12px auto;
    border-radius: 14px;
    background: rgba(77,184,255,0.12);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #4db8ff;
    font-size: 1.25rem;
}

.mcard h2 {
    margin: 0;
    color: #f8fafc;
    font-size: 1.55rem;
    font-weight: 800;
}

.mcard p {
    margin: 8px 0 0 0;
    color: #94a3b8;
    font-size: 0.92rem;
}

.ifbox {
    background: linear-gradient(135deg, rgba(77,184,255,0.12), rgba(59,130,246,0.08));
    border: 1px solid rgba(77,184,255,0.25);
    color: #e2e8f0;
    padding: 14px 16px;
    border-radius: 14px;
    margin: 12px 0 18px 0;
    box-shadow: 0 8px 20px rgba(0,0,0,0.16);
}

.empty-box {
    background: linear-gradient(135deg, #131c2e 0%, #0f172a 100%);
    border: 1.5px dashed rgba(77,184,255,0.35);
    border-radius: 24px;
    padding: 60px 30px;
    text-align: center;
    margin: 30px 0;
    box-shadow: 0 14px 30px rgba(0,0,0,0.22);
}

.empty-box .empty-icon {
    width: 90px;
    height: 90px;
    margin: 0 auto 18px auto;
    border-radius: 22px;
    background: rgba(77,184,255,0.10);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #4db8ff;
    font-size: 2.4rem;
}

.empty-box h2 {
    color: #f8fafc;
    margin-bottom: 10px;
    font-size: 1.8rem;
    font-weight: 800;
}

.empty-box p {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

# =========================
# VARIABLES INICIALES
# =========================
df = None
fname = "dataset"
header_row_used = 0

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("""
        <div class="side-title">
            <i class="fa-solid fa-sliders"></i> Configuración
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-box">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "📂 Cargar datos (CSV o Excel)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True
    )
    use_demo = st.button("Datos de demostración", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="side-subtitle">
            <i class="fa-solid fa-screwdriver-wrench"></i> Parámetros
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-box">', unsafe_allow_html=True)
    max_k = st.slider("Máx. clusters", 2, 10, 6) # Num. Max de clusters (agrupamiento)
    corr_thr = st.slider("Umbral correlación", 0.3, 0.95, 0.6, 0.05)
    cont = st.slider("Contaminación outliers", 0.01, 0.20, 0.05, 0.01)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# CARGA DE DATOS
# =========================
if use_demo:
    df = make_demo() #Aquí llamamos al demo, en caso de que no se suba ningún dato
    fname = "demo_clientes"
    st.markdown(
        '<div class="ifbox"><i class="fa-solid fa-flask"></i> Datos de demostración: 500 clientes con variables financieras.</div>',
        unsafe_allow_html=True
    )

elif uploaded:
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
                for sh in xl.sheet_names:
                    f.seek(0)
                    loaded, hrow = smart_read_excel(f, sheet_name=sh)
                    key = f"{f.name} — {sh}" if len(xl.sheet_names) > 1 else f.name
                    all_loaded[key] = {"df": loaded, "header_row": hrow, "sheet": sh}
                    options.append(key)
        except Exception as e:
            st.error(f"Error cargando {f.name}: {e}")

    if options:
        selected = st.selectbox("📂 Seleccionar dataset / hoja:", options) if len(options) > 1 else options[0]
        info = all_loaded[selected]
        df = info["df"]
        fname = selected.replace(" — ", "_").replace(".xlsx", "").replace(".csv", "")
        header_row_used = info["header_row"]

# =========================
# SI NO HAY DATASET
# =========================
if df is None:
    st.markdown("""
    <div class="empty-box">
        <div class="empty-icon">
  
                          <i class="fa-solid fa-cloud-arrow-up"></i>
        </div>
        <h2>Carga tu dataset para comenzar</h2>
        <p>Sube un archivo CSV o Excel para generar el análisis automático.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# =========================
# MENSAJE DE ENCABEZADOS
# =========================
if header_row_used > 0:
    st.markdown(
        f'<div class="ifbox"><i class="fa-solid fa-circle-info"></i> Encabezados detectados en la fila <b>{header_row_used + 1}</b> del archivo.</div>',
        unsafe_allow_html=True
    )

# =========================
# DETECCIÓN DE TIPOS
# =========================
vt = detect_types(df)
nc = [c for c, t in vt.items() if t == "numeric"]
cat_cols = [c for c, t in vt.items() if t == "categorical"]
temp_cols = [c for c, t in vt.items() if t == "temporal"]

# =========================
# ANÁLISIS
# =========================
with st.spinner("⚡ Analizando datos..."):
    dstats = do_stats(df, nc)
    P, S, corrs = do_corr(df, nc, corr_thr)
    ol = do_outliers(df, nc, cont) if len(nc) >= 1 else {}
    cl = do_kmeans(df, nc, max_k) if len(nc) >= 2 else None
    cat_analysis = analyze_categorical(df, cat_cols)
    ins, ale, inf = auto_insights(df, vt, ol, cl, corrs, cat_analysis)

# =========================
# TARJETAS MÉTRICAS
# =========================
cols = st.columns(6)
mets = [
    (len(df), "Registros", "fa-solid fa-database"),
    (len(df.columns), "Variables", "fa-solid fa-table-columns"),
    (len(nc), "Numéricas", "fa-solid fa-chart-line"),
    (len(cat_cols), "Categóricas", "fa-solid fa-tags"),
    (df.isnull().sum().sum(), "Faltantes", "fa-solid fa-circle-question"),
    (cl["best_k"] if cl else (len(temp_cols) if len(temp_cols) > 0 else "—"), "Clusters/Temp", "fa-solid fa-diagram-project")
]

for col, (val, lbl, ico) in zip(cols, mets):
    with col:
        valor = val if isinstance(val, str) else f"{val:,}"
        st.markdown(f"""
            <div class="mcard">
                <div class="micon">
                    <i class="{ico}"></i>
                </div>
                <div class="mnum">{valor}</div>
                <p>{lbl}</p> 
            </div>
""", unsafe_allow_html=True)

# =========================
# INSIGHTS
# =========================
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🔍 Insights y Alertas Automáticas", expanded=True):
    for a in ale:
        st.markdown(f'<div class="abox">{a}</div>', unsafe_allow_html=True)

    for i in ins:
        st.markdown(f'<div class="ibox">{i}</div>', unsafe_allow_html=True)

    for i in inf:
        st.markdown(f'<div class="ifbox">{i}</div>', unsafe_allow_html=True)

tab_labels = ["Categorias","Numéricas","Correlaciones","Outliers","Clustering","Relaciones","Exportar"]
tabs = st.tabs(tab_labels)

ch_exp = {}
pdf_pages = []

    # TAB 0 DE CATEGORIAS, MUESTRA CONTEO DE CATEGORIAS, GRAFICOS, TABLA CRUZADA, MAPA DE CALOR. USA PLOTLY
with tabs[0]:
        st.markdown('<div class="stitle">Análisis de Variables Categorias</div>', unsafe_allow_html=True)
        if not cat_cols:
            st.info("No se detectaron variables Categorias en este dataset.")
        else:
            for col in cat_cols:
                data = cat_analysis[col]
                counts = data["counts"]
                st.markdown(f"{col}")

                vc_df = pd.DataFrame({"Valor": counts.index, "Cantidad": counts.values, "%": data["pcts"].values})
                col_left, col_right = st.columns([1, 2])

                with col_left:
                    for val, cnt in counts.head(6).items():
                        pct = round(cnt/len(df)*100, 1)
                        bar_w = int(pct)
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
                    if len(counts) <= 6:
                        fig_cat = px.pie(vc_df, values="Cantidad", names="Valor", title=f"Distribución de {col}",
                                         template="plotly_dark", color_discrete_sequence=COLORS, hole=0.35)
                        fig_cat.update_traces(textposition="inside", textinfo="percent+label")
                    else:
                        fig_cat = px.bar(vc_df.head(15), x="Valor", y="Cantidad", text="%", title=f"Top valores en {col}",
                                         template="plotly_dark", color="Cantidad", color_continuous_scale="Blues")
                        fig_cat.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig_cat.update_layout(paper_bgcolor="#161b27", plot_bgcolor="#1a2332", height=300,
                                          margin=dict(t=40,b=20,l=20,r=20), showlegend=len(counts)<=6,
                                          font=dict(color="#e0e0e0"))
                    st.plotly_chart(fig_cat, use_container_width=True)
                    pdf_pages.append({"name": f"Categóricas: {col}", "fig": fig_cat})
                    ch_exp[col] = fig_cat.to_html(include_plotlyjs=True, full_html=False)

                st.markdown("---")

            if len(cat_cols) >= 2:
                st.markdown('<div class="stitle">Tabla Cruzada entre Variables</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1: col_a = st.selectbox("Variable fila:", cat_cols, key="cross_a")
                with c2:
                    other = [c for c in cat_cols if c != col_a]
                    col_b = st.selectbox("Variable columna:", other, key="cross_b")

                ct = pd.crosstab(df[col_a], df[col_b], margins=True, margins_name="TOTAL")
                st.dataframe(ct.style.background_gradient(cmap="Blues"), use_container_width=True)

                ct_no_margins = pd.crosstab(df[col_a], df[col_b])
                fig_ct = px.imshow(ct_no_margins, text_auto=True, aspect="auto",
                                   title=f"Heatmap: {col_a} × {col_b}",
                                   template="plotly_dark", color_continuous_scale="Blues")
                fig_ct.update_layout(paper_bgcolor="#161b27", height=380, font=dict(color="#e0e0e0"))
                st.plotly_chart(fig_ct, use_container_width=True)
                pdf_pages.append({"name": f"Cruce {col_a} x {col_b}", "fig": fig_ct})
                ch_exp[f"Cruce {col_a} {col_b}"] = fig_ct.to_html(include_plotlyjs=True, full_html=False)

    # TAB 1
with tabs[1]:
        st.markdown('<div class="stitle">Análisis de Variables Numéricas</div>', unsafe_allow_html=True)
        if not nc:
            st.info("No se detectaron variables numéricas en este dataset.")
        else:
            st.dataframe(dstats.style.format(precision=4).background_gradient(subset=["Media","Desv.Std"], cmap="Blues"),
                         use_container_width=True, height=400)

            np2 = min(3, len(nc)); nr2 = (len(nc)+np2-1)//np2
            fig_hist = make_subplots(rows=nr2, cols=np2, subplot_titles=nc)
            for i, c in enumerate(nc):
                fig_hist.add_trace(go.Histogram(x=df[c].dropna(), name=c, marker_color=COLORS[i%len(COLORS)],
                                                opacity=.75, showlegend=False),
                                   row=i//np2+1, col=i%np2+1)
            fig_hist.update_layout(template="plotly_dark", paper_bgcolor="#161b27", plot_bgcolor="#1a2332",
                                   height=max(260, nr2*200), font=dict(color="#e0e0e0"))
            st.plotly_chart(fig_hist, use_container_width=True)
            pdf_pages.append({"name": "Numéricas: Histogramas", "fig": fig_hist})
            ch_exp["Histogramas"] = fig_hist.to_html(include_plotlyjs=True, full_html=False)

            fig_box = go.Figure()
            for i, c in enumerate(nc):
                fig_box.add_trace(go.Box(y=df[c].dropna(), name=c, marker_color=COLORS[i%len(COLORS)], boxmean="sd"))
            fig_box.update_layout(template="plotly_dark", paper_bgcolor="#161b27", plot_bgcolor="#1a2332",
                                  height=380, font=dict(color="#e0e0e0"), title="Boxplots")
            st.plotly_chart(fig_box, use_container_width=True)
            pdf_pages.append({"name": "Numéricas: Boxplots", "fig": fig_box})
            ch_exp["Boxplots"] = fig_box.to_html(include_plotlyjs=True, full_html=False)

    # TAB 2
with tabs[2]:
        if P is not None and len(nc) >= 2:
            cl2, cr2 = st.columns([2,1])
            with cl2:
                m = st.radio("Método", ["Pearson","Spearman"], horizontal=True)
                cm = P if m=="Pearson" else S
                fig_corr = go.Figure(go.Heatmap(
                    z=cm.values, x=cm.columns.tolist(), y=cm.columns.tolist(),
                    colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                    text=cm.round(2).values, texttemplate="%{text}", textfont={"size":9},
                    hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.4f}<extra></extra>"
                ))
                fig_corr.update_layout(
                    title=f"Correlación ({m})",
                    template="plotly_dark",
                    paper_bgcolor="#161b27",
                    plot_bgcolor="#161b27",
                    height=480,
                    font=dict(color="#e0e0e0")
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                pdf_pages.append({"name": f"Correlaciones: Heatmap ({m})", "fig": fig_corr})
                ch_exp["Mapa de Calor"] = fig_corr.to_html(include_plotlyjs=True, full_html=False)

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

    # TAB 3
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
                fig_out = px.bar(pcd, x="Variable", y="Outliers IQR", text="%",
                                 template="plotly_dark", color="Outliers IQR", color_continuous_scale="Reds")
                fig_out.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_out.update_layout(paper_bgcolor="#161b27", plot_bgcolor="#1a2332",
                                      height=360, showlegend=False, font=dict(color="#e0e0e0"),
                                      xaxis=dict(tickangle=45))
                st.plotly_chart(fig_out, use_container_width=True)
                pdf_pages.append({"name": "Outliers", "fig": fig_out})
        else:
            st.info("Se necesitan variables numéricas para detectar outliers.")

    # TAB 4
with tabs[4]:
        if cl:
            c1,c2,c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="mcard" style="border-top:3px solid #4db8ff"><div style="font-size:1.3rem">🔵</div><h2 style="color:#4db8ff">{cl["best_k"]}</h2><p>Clusters óptimos</p></div>', unsafe_allow_html=True)
            with c2:
                sc = cl["silhouette"]
                sc_c = "#4caf50" if sc>.5 else "#ff9800" if sc>.25 else "#f44336"
                sc_l = "Excelente" if sc>.7 else "Buena" if sc>.5 else "Aceptable" if sc>.25 else "Débil"
                st.markdown(f'<div class="mcard" style="border-top:3px solid {sc_c}"><div style="font-size:1.3rem">📏</div><h2 style="color:{sc_c}">{sc}</h2><p>Silhouette ({sc_l})</p></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="mcard" style="border-top:3px solid #ab47bc"><div style="font-size:1.3rem">📊</div><h2 style="color:#ab47bc">{len(cl["k_range"])}</h2><p>k evaluados</p></div>', unsafe_allow_html=True)

            cel, cpca = st.columns(2)
            with cel:
                fe = make_subplots(rows=1, cols=2, subplot_titles=["Codo","Silhouette"])
                fe.add_trace(go.Scatter(x=cl["k_range"], y=cl["inertias"], mode="lines+markers",
                                        line=dict(color="#4db8ff", width=2), marker=dict(size=7), showlegend=False), row=1, col=1)
                fe.add_trace(go.Scatter(x=cl["k_range"], y=cl["silhouettes"], mode="lines+markers",
                                        line=dict(color="#4caf50", width=2), marker=dict(size=7), showlegend=False), row=1, col=2)
                fe.add_vline(x=cl["best_k"], line_dash="dash", line_color="#f44336", row=1, col=1)
                fe.add_vline(x=cl["best_k"], line_dash="dash", line_color="#f44336", row=1, col=2)
                fe.update_layout(template="plotly_dark", paper_bgcolor="#161b27", plot_bgcolor="#1a2332",
                                 height=280, font=dict(color="#e0e0e0"))
                st.plotly_chart(fe, use_container_width=True)
                pdf_pages.append({"name": "Clustering (Codo y Silhouette)", "fig": fe})

            with cpca:
                fp = px.scatter(cl["pca_df"], x="PC1", y="PC2", color="Cluster", opacity=.75,
                                template="plotly_dark", color_discrete_sequence=COLORS, title=f"Clusters PCA (k={cl['best_k']})")
                fp.update_layout(paper_bgcolor="#161b27", plot_bgcolor="#1a2332", height=280, font=dict(color="#e0e0e0"))
                st.plotly_chart(fp, use_container_width=True)
                pdf_pages.append({"name": "Clustering PCA", "fig": fp})
                ch_exp["PCA Clusters"] = fp.to_html(include_plotlyjs=True, full_html=False)

            st.markdown('<div class="stitle">Perfiles de Clusters</div>', unsafe_allow_html=True)
            pr = cl["profiles"].round(4).copy()
            pr.index = [f"Cluster {i+1}" for i in pr.index]
            st.dataframe(pr.style.background_gradient(cmap="Blues", axis=0), use_container_width=True)
        else:
            st.info("Se requieren ≥ 2 variables numéricas y ≥ 15 registros para clustering.")

    # TAB 5
with tabs[5]:
        if len(nc) >= 2:
            st.markdown('<div class="stitle">Dispersión entre Numéricas</div>', unsafe_allow_html=True)
            c1,c2,c3 = st.columns(3)
            with c1: vx = st.selectbox("Var X:", nc, 0, key="rx")
            with c2: vy = st.selectbox("Var Y:", nc, min(1,len(nc)-1), key="ry")
            with c3: vc3 = st.selectbox("Color:", ["(ninguno)"]+cat_cols, key="rc")

            pd2 = df[[vx,vy]+([vc3] if vc3!="(ninguno)" else [])].dropna()
            rv, pv = stats.pearsonr(pd2[vx].values, pd2[vy].values)
            fs = px.scatter(pd2, x=vx, y=vy, color=vc3 if vc3!="(ninguno)" else None, trendline="ols",
                            opacity=.65, title=f"{vx} vs {vy}  |  r={rv:.4f}  |  p={pv:.4f}",
                            template="plotly_dark", color_discrete_sequence=COLORS)
            fs.update_layout(paper_bgcolor="#161b27", plot_bgcolor="#1a2332", height=440, font=dict(color="#e0e0e0"))
            st.plotly_chart(fs, use_container_width=True)
            pdf_pages.append({"name": f"Relación {vx} vs {vy}", "fig": fs})

            if cat_cols and nc:
                st.markdown('<div class="stitle">Categoria × Numérica</div>', unsafe_allow_html=True)
                c1,c2 = st.columns(2)
                with c1: cs = st.selectbox("Categórica:", cat_cols, key="cn1")
                with c2: ns = st.selectbox("Numérica:", nc, key="cn2")
                tops = df[cs].value_counts().head(10).index.tolist()
                flt = df[df[cs].isin(tops)][[cs,ns]].dropna()
                fv = px.violin(flt, x=cs, y=ns, box=True, points="all", color=cs,
                               template="plotly_dark", color_discrete_sequence=COLORS)
                fv.update_layout(paper_bgcolor="#161b27", plot_bgcolor="#1a2332", height=420,
                                 showlegend=False, xaxis=dict(tickangle=30), font=dict(color="#e0e0e0"))
                st.plotly_chart(fv, use_container_width=True)
                pdf_pages.append({"name": f"{cs} vs {ns}", "fig": fv})
        else:
            st.info("Se requieren ≥ 2 variables numéricas para relaciones.")

    # TAB 6 EXPORTAR (solo botones)
with tabs[6]:
        st.markdown('<div class="stitle">Exportar Resultados</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        # estado
        if "pdf_bytes" not in st.session_state:
            st.session_state.pdf_bytes = None
        if "pdf_name" not in st.session_state:
            st.session_state.pdf_name = None
        if "html_bytes" not in st.session_state:
            st.session_state.html_bytes = None
        if "html_name" not in st.session_state:
            st.session_state.html_name = None

        with c1:
            if st.button("Generar PDF", use_container_width=True, key="btn_gen_pdf"):
                try:
                    if not pdf_pages:
                        st.session_state.pdf_bytes = None
                        st.session_state.pdf_name = None
                        st.error("No hay gráficos disponibles para exportar.")
                    else:
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                        st.session_state.pdf_bytes = make_pdf_report_plotly(pdf_pages)
                        st.session_state.pdf_name = f"AutoAnalytics_{fname}_{ts}.pdf"
                        st.success("PDF listo. Dale click a 'Descargar PDF'.")
                except Exception as e:
                    st.session_state.pdf_bytes = None
                    st.session_state.pdf_name = None
                    st.error(str(e))

            if st.session_state.pdf_bytes is not None:
                pdf_b64 = base64.b64encode(st.session_state.pdf_bytes).decode("utf-8")
                pdf_name = st.session_state.pdf_name or "AutoAnalytics.pdf"

                st.markdown(
                    f'''
                    <a href="data:application/pdf;base64,{pdf_b64}" download="{pdf_name}"
                       style="display:block;background:linear-gradient(135deg,#1565c0,#0d47a1);
                       color:white;text-align:center;padding:14px;border-radius:8px;
                       text-decoration:none;font-weight:600;margin-top:10px">
                       ⬇️ Descargar PDF
                    </a>
                    ''',
                    unsafe_allow_html=True
            )

        with c2:
            if st.button("Generar HTML", use_container_width=True, key="btn_gen_html"):
                try:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                    html = make_report(df, vt, dstats, ol, cl, corrs, ins, ale, inf, ch_exp, cat_analysis)
                    st.session_state.html_bytes = html.encode("utf-8")
                    st.session_state.html_name = f"AutoAnalytics_{fname}_{ts}.html"
                    st.success("HTML listo. Dale click a 'Descargar HTML'.")
                except Exception as e:
                    st.session_state.html_bytes = None
                    st.session_state.html_name = None
                    st.error(str(e))

            if st.session_state.html_bytes is not None:
                st.download_button(
                    "⬇️ Descargar HTML",
                    data=st.session_state.html_bytes,
                    file_name=st.session_state.html_name or "AutoAnalytics.html",
                    mime="text/html",
                    use_container_width=True,
                    key="dl_html"
                )
if __name__ == "__main__":
    main()
"""
Dashboard de Análisis de Encuestas - Laboratorio de Física
---------------------------------------------------------
Autores: J. I. Peralta & Gemini Pro 3.0
Fecha: 05/12/2025

Descripción:
Este script genera un dashboard interactivo para el análisis de encuestas de satisfacción.
Incluye procesamiento de datos robusto, análisis de sentimiento, visualización estadística
y funciones de gestión de estado para filtros.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Encuesta del laboratorio de Física",
    page_icon="📊",
    layout="wide"
)

# --- GESTIÓN DE ESTADO (Para el botón de reset) ---
if 'lab_filter' not in st.session_state: st.session_state.lab_filter = 'Todos'
if 'car_filter' not in st.session_state: st.session_state.car_filter = 'Todas'
if 'doc_filter' not in st.session_state: st.session_state.doc_filter = 'Todos'

def reset_filters():
    st.session_state.lab_filter = 'Todos'
    st.session_state.car_filter = 'Todas'
    st.session_state.doc_filter = 'Todos'

# --- LISTA OFICIAL Y NORMALIZACIÓN DE DOCENTES ---
DOCENTES_OFICIALES = [
    "Chaparro, Emiliano",
    "Dragone, Esteban",
    "Leone, Emiliano",
    "Merlo, Rafael",
    "Orozco Gil, Estefanía",
    "Oviedo, Carla",
    "Peralta, Juan Ignacio",
    "Romeo, Martín",
    "Vieytes, Mariela",
    "Villalba, Martín"
]

NORMALIZACION_DOCENTES = {
    "Esteban Dragone": "Dragone, Esteban",
    "Rafael Merlo": "Merlo, Rafael",
    "Carla Oviedo": "Oviedo, Carla",
    "Stefanía Orozco": "Orozco Gil, Estefanía",
    "Stefanía Orozco Gil": "Orozco Gil, Estefanía",
    "Estefanía Orozco Gil": "Orozco Gil, Estefanía",
    "Orozco Gil, Stefanía": "Orozco Gil, Estefanía", 
    "Juan Ignacio Peralta": "Peralta, Juan Ignacio",
    "Mariela Vieytes": "Vieytes, Mariela",
    "Martín Villalba": "Villalba, Martín",
    "Emiliano Leone": "Leone, Emiliano",
    "Emiliano Chaparro": "Chaparro, Emiliano",
    "Fabiana Chaparro": "Chaparro, Emiliano", 
    "Martín Romeo": "Romeo, Martín"
}

# --- MAPEO DE PREGUNTAS ---
PREGUNTA_MAP = {
    'Marca temporal': 'Timestamp',
    'Asistí al': 'Laboratorio',
    'Docentes de Laboratorio': 'Docentes',
    '¿Qué tan fáciles de entender le resultaron las guías de laboratorio?': 'Calif_Guias',
    '¿Le parecieron útiles los videos subidos al aula para explicar las guías de laboratorio?': 'Calif_Videos',
    '¿Cómo evaluaría la coordinación entre los TEMAS tratados en las clases teóricas y en el laboratorio?': 'Calif_Coord_Teoria',
    '¿Le resultaron claras las explicaciones de los docentes del laboratorio en las clases presenciales?': 'Calif_Docentes_Expl',
    '¿Le resultaron útiles las correcciones del docente luego de cada informe?': 'Calif_Correcciones',
    '¿Considera que los conocimientos adquiridos en la cursada de laboratorio le sirvieron para entender de una forma más profunda los fenómenos físicos vistos de forma teórica en la materia?': 'Calif_Impacto_Aprendizaje',
    'Escribí tres  palabras que describan tu experiencia en el laboratorio (NO MAS por favor! y no poner artículos ni preposiciones)': 'Palabras_Clave',
    '¿Qué opinión tenés respecto del uso del aula virtual para manejar la cursada de labo y el material multimedia subido en la misma? En lo posible enumerá pros y contras. ': 'Opinion_Aula_Virtual',
    '¿Qué cosas mejorarías, y qué cosas te parecieron buenas de la cursada de laboratorio?': 'Opinion_Mejoras',
    '¿Qué te pareció la idea de dar una charla para el tercer TP en reemplazo del informe?': 'Opinion_Charla'
}

REVERSE_MAP = {v: k for k, v in PREGUNTA_MAP.items()}

# --- EXPLICACIONES TÉCNICAS ---
EXPLICACIONES_CARRERA = {
    'Calif_Guias': "Evalúa la accesibilidad y claridad del material escrito (guías de trabajos prácticos) desde la perspectiva de cada especialidad académica.",
    'Calif_Videos': "Mide la utilidad percibida del material audiovisual de apoyo. Permite identificar si el contenido multimedia satisface las necesidades de estudio de las diferentes carreras.",
    'Calif_Coord_Teoria': "Analiza la sincronización percibida entre los contenidos teóricos de la materia y la práctica de laboratorio para cada plan de estudios.",
    'Calif_Docentes_Expl': "Cuantifica la claridad expositiva del equipo docente en las clases presenciales, desglosado por el perfil académico de los estudiantes.",
    'Calif_Correcciones': "Mide la valoración del feedback y las correcciones recibidas en los informes, indicando si la retroalimentación resulta útil y comprensible para las distintas especialidades.",
    'Calif_Impacto_Aprendizaje': "Indicador de relevancia académica. Refleja en qué medida los estudiantes consideran que el laboratorio contribuyó a su comprensión profunda de los fenómenos físicos.",
}
EXPLICACION_DEFAULT = "Presenta el promedio de satisfacción desagregado por carrera. Permite detectar variaciones en la experiencia educativa según la especialidad del alumno."

# --- FUNCIONES ---

@st.cache_data(ttl=600)
def load_data(file_or_url, is_url=False):
    try:
        df = None
        if is_url:
            url = file_or_url
            if "docs.google.com/spreadsheets" in url:
                if "/edit" in url:
                    url = url.split("/edit")[0] + "/export?format=csv"
                elif "/view" in url:
                    url = url.split("/view")[0] + "/export?format=csv"
            df = pd.read_csv(url)
        else:
            if file_or_url.name.endswith('.csv'):
                df = pd.read_csv(file_or_url)
            else:
                df = pd.read_excel(file_or_url)
        
        # --- FUSIÓN ROBUSTA DE CARRERAS ---
        if 'Carrera (F1)' in df.columns:
            df['Carrera (F1)'] = df['Carrera (F1)'].replace(r'^\s*$', np.nan, regex=True)
        if 'Carrera (F2)' in df.columns:
            df['Carrera (F2)'] = df['Carrera (F2)'].replace(r'^\s*$', np.nan, regex=True)

        if 'Carrera (F1)' in df.columns and 'Carrera (F2)' in df.columns:
            df['Carrera'] = df['Carrera (F1)'].fillna(df['Carrera (F2)'])
        elif 'Carrera (F1)' in df.columns:
            df['Carrera'] = df['Carrera (F1)']
        elif 'Carrera (F2)' in df.columns:
            df['Carrera'] = df['Carrera (F2)']
        
        df.rename(columns={k: v for k, v in PREGUNTA_MAP.items() if k in df.columns}, inplace=True)
        
        if 'Timestamp' in df.columns: 
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)

        for c in ['Laboratorio', 'Carrera', 'Docentes']:
            if c in df.columns: 
                df[c] = df[c].fillna('Sin Especificar').astype(str)

        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

def extract_teachers_from_row(row_str, official_names, mapping_dict):
    """Extrae y normaliza nombres de docentes."""
    found = set()
    row_clean = str(row_str).strip()
    
    # 1. Buscar nombres oficiales exactos
    for t in official_names:
        if t in row_clean: found.add(t)
    
    # 2. Buscar variantes y normalizar
    for variant, official in mapping_dict.items():
        if variant in row_clean: found.add(official)
        
    # 3. Fallback
    if not found and row_clean.lower() not in ['nan', 'sin especificar', '']:
        return [p.strip() for p in row_clean.split(',')]
        
    return sorted(list(found))

def calcular_sentimiento(df_input):
    text_cols = ['Opinion_Mejoras', 'Opinion_Aula_Virtual', 'Opinion_Charla']
    valid = [c for c in text_cols if c in df_input.columns]
    if not valid: return "Sin datos", "#808080"
    
    pos = ['bueno', 'buena', 'excelente', 'útil', 'claro', 'ayuda', 'mejor', 'bien', 'interesante', 'gustó', 'sirvió', 'aprendizaje', 'buenos', 'claras', 'dinamica', 'agil', 'correcta']
    neg = ['malo', 'mala', 'confuso', 'difícil', 'complicado', 'tarde', 'desorganizado', 'problema', 'pésimo', 'no entendí', 'lento', 'poco', 'falta', 'injusto', 'perdido']
    
    score = 0; count = 0
    for _, row in df_input.iterrows():
        txt = " ".join([str(row[c]) for c in valid if pd.notna(row[c])]).lower()
        if len(txt) < 3: continue
        p = sum(1 for w in pos if w in txt)
        n = sum(1 for w in neg if w in txt)
        val = max(min(p - n, 3), -3)
        score += val; count += 1
        
    if count == 0: return "Neutro / Sin Texto", "#808080"
    avg = score / count
    
    if avg > 0.5: return "Muy Positivo 😄", "#28a745"
    elif avg > 0.1: return "Positivo 🙂", "#90EE90"
    elif avg < -0.5: return "Negativo 😟", "#dc3545"
    elif avg < -0.1: return "Algo Negativo 😐", "#ffc107"
    else: return "Neutro 😐", "#6c757d"

# --- UI PRINCIPAL ---

st.markdown("""
<style>
    .header-container {
        display: flex;
        align-items: center;
        padding-bottom: 20px;
        border-bottom: 2px solid #6c757d;
        margin-bottom: 20px;
    }
    .logo-text {
        font-size: 2.5rem;
        font-weight: 800;
        color: #333;
        margin-right: 15px;
        letter-spacing: -1px;
    }
    .sub-text {
        font-size: 1.2rem;
        color: #555;
    }
    .dept-text {
        font-size: 0.9rem;
        color: #777;
        font-style: italic;
    }
</style>
<div class="header-container">
    <div>
        <span class="logo-text">UNTREF</span>
    </div>
    <div style="margin-left: 15px; border-left: 2px solid #ddd; padding-left: 15px;">
        <div class="sub-text">Universidad Nacional de Tres de Febrero</div>
        <div class="dept-text">Departamento de Ciencia y Tecnología | Laboratorio de Física</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.title("📊 Encuesta del Laboratorio de Física")

src = st.radio("Fuente de Datos:", ("Pegar Link de Google Sheet", "Subir Archivo (.xlsx / .csv)"), horizontal=True)
df = None

if src == "Subir Archivo (.xlsx / .csv)":
    up = st.file_uploader("Archivo", type=['csv', 'xlsx'])
    if up: df = load_data(up, False)
else:
    url = st.text_input("Link Público:")
    if url: df = load_data(url, True)

if df is not None:
    if 'Docentes' in df.columns:
        df['Docentes_List'] = df['Docentes'].apply(lambda x: extract_teachers_from_row(x, DOCENTES_OFICIALES, NORMALIZACION_DOCENTES))

    # --- BARRA LATERAL (FILTROS) ---
    st.sidebar.header("Filtros")
    
    if st.sidebar.button("🔄 Borrar Filtros", on_click=reset_filters, type="primary"):
        st.rerun()

    sel_lab = st.sidebar.selectbox(
        "Laboratorio", 
        ['Todos'] + sorted(df['Laboratorio'].unique()), 
        help="Filtra las respuestas según el laboratorio cursado.",
        key='lab_filter'
    )
    sel_car = st.sidebar.selectbox(
        "Carrera", 
        ['Todas'] + sorted(df['Carrera'].unique()), 
        help="Filtra por la carrera declarada por el estudiante.",
        key='car_filter'
    )
    
    sel_doc = st.sidebar.selectbox(
        "Docente (Presente)", 
        ['Todos'] + sorted(DOCENTES_OFICIALES), 
        help="Muestra encuestas donde el docente seleccionado estuvo presente en la comisión.",
        key='doc_filter'
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Visualización")
    viz_mode = st.sidebar.radio("Unidad:", ("Porcentaje (%)", "Cantidad Absoluta/Escala"), index=0, help="Cambia entre vista porcentual (0-100%) o valores absolutos (votos y escala 1-5).")
    is_pct = (viz_mode == "Porcentaje (%)")

    # Aplicar Filtros
    df_f = df.copy()
    if sel_lab != 'Todos': df_f = df_f[df_f['Laboratorio'] == sel_lab]
    if sel_car != 'Todas': df_f = df_f[df_f['Carrera'] == sel_car]
    if sel_doc != 'Todos': df_f = df_f[df_f['Docentes_List'].apply(lambda x: sel_doc in x)]

    # --- MÉTRICAS TOP ---
    tot, filt = len(df), len(df_f)
    pct = (filt/tot*100) if tot>0 else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Encuestas", f"{filt} de {tot}", f"{pct:.1f}% de la muestra total")

    sent_txt, sent_col = calcular_sentimiento(df_f)
    with c2:
        st.markdown(f"<div style='background:{sent_col};color:white;padding:10px;border-radius:5px;text-align:center'><b>{sent_txt}</b></div>", unsafe_allow_html=True)
    with c3:
         st.caption("Termómetro basado en análisis de palabras clave en todos los campos de texto.")
    
    st.divider()

    # --- INDICADORES GENERALES Y COMPARATIVA TOTAL ---
    st.markdown("### 📈 Resumen General y Comparativa")
    
    calif_cols = [c for c in df_f.columns if c.startswith('Calif_')]
    if calif_cols:
        # A. Promedios Globales
        avgs = df_f[calif_cols].mean()
        cols_kpi = st.columns(len(calif_cols))
        for i, col in enumerate(calif_cols):
            val = avgs[col]
            lbl = col.replace('Calif_', '').replace('_', ' ')
            d_val = f"{(val/5)*100:.0f}%" if is_pct else f"{val:.2f}"
            cols_kpi[i].metric(lbl, d_val)
        
        st.write("") 
        
        # B. Comparativa TOTAL por Carrera
        st.markdown("#### 🆚 Comparativa Global de Satisfacción por Carrera")
        
        df_f['Score_Global'] = df_f[calif_cols].mean(axis=1)
        
        career_global = df_f.groupby('Carrera')['Score_Global'].agg(['mean', 'count']).reset_index().sort_values('mean', ascending=True)
        
        if is_pct:
            career_global['Display'] = (career_global['mean'] / 5) * 100
            x_ax = 'Display'
            x_title = "% Satisfacción Global"
            txt_fmt = '.1f'; txt_suf = '%'
            range_x = [0, 110]
        else:
            career_global['Display'] = career_global['mean']
            x_ax = 'Display'
            x_title = "Promedio General (1-5)"
            txt_fmt = '.2f'; txt_suf = ''
            range_x = [1, 5.8] 
            
        career_global['Label'] = career_global.apply(lambda x: f"{x['Carrera']} (N={int(x['count'])})", axis=1)
        
        fig_global = px.bar(career_global, y='Label', x=x_ax, text=x_ax,
                            orientation='h', title="Satisfacción Promedio Unificada por Carrera",
                            color=x_ax, color_continuous_scale='Teal')
        
        fig_global.update_traces(texttemplate='%{text:' + txt_fmt + '}' + txt_suf, textposition='outside')
        fig_global.update_layout(xaxis=dict(range=range_x, title=x_title), yaxis_title=None, height=400)
        
        st.plotly_chart(fig_global, use_container_width=True, key="global_chart")
        st.info("**Interpretación:** Este gráfico condensa todas las calificaciones cuantitativas en un único puntaje promedio. Permite visualizar rápidamente el nivel general de satisfacción de cada carrera con respecto a la cursada completa.", icon="ℹ️")

    st.divider()

    # --- ANÁLISIS DETALLADO ---
    st.markdown("### 📝 Análisis Detallado por Pregunta")
    
    for col in calif_cols:
        title = REVERSE_MAP.get(col, col)
        st.subheader(f"📌 {title}")
        
        c_left, c_right = st.columns([1, 1])
        
        # IZQ: Distribución
        with c_left:
            cnt = df_f[col].value_counts().reindex([1,2,3,4,5], fill_value=0).reset_index()
            cnt.columns = ['Puntaje', 'Valor']
            
            y_val = 'Valor'
            if is_pct:
                tot_v = cnt['Valor'].sum()
                cnt['Pct'] = (cnt['Valor']/tot_v*100).fillna(0)
                y_val = 'Pct'; y_title = "%"; 
                y_max = 115 
                txt_t = '%{y:.1f}%'
            else:
                y_title = "Votos"
                y_max = cnt['Valor'].max() * 1.2 
                txt_t = '%{y}'
            
            fig1 = px.bar(cnt, x='Puntaje', y=y_val, text=y_val, title=f"Distribución ({y_title})",
                          color=y_val, color_continuous_scale='Blues')
            fig1.update_traces(texttemplate=txt_t, textposition='outside')
            fig1.update_layout(yaxis=dict(range=[0, y_max], title=y_title), height=350)
            
            st.plotly_chart(fig1, use_container_width=True, key=f"dist_{col}")
            st.caption("Distribución de los puntajes otorgados (escala 1 a 5) para este aspecto específico.")

        # DER: Comparativa Carrera
        with c_right:
            cg = df_f.groupby('Carrera')[col].agg(['mean', 'count']).reset_index().sort_values('mean', ascending=True)
            
            if is_pct:
                cg['Val'] = (cg['mean']/5)*100
                x_rng = [0, 115]
                fmt = '.1f'; suf = '%'
                t_suf = "(% Satisfacción)"
            else:
                cg['Val'] = cg['mean']
                x_rng = [1, 5.8]
                fmt = '.2f'; suf = ''
                t_suf = "(Escala 1-5)"
                
            cg['Lbl'] = cg.apply(lambda x: f"{x['Carrera']} (N={int(x['count'])})", axis=1)
            
            fig2 = px.bar(cg, y='Lbl', x='Val', text='Val', orientation='h',
                          title=f"Promedio por Carrera {t_suf}",
                          color='Val', color_continuous_scale='Teal')
            fig2.update_traces(texttemplate='%{text:'+fmt+'}'+suf, textposition='outside')
            fig2.update_layout(xaxis=dict(range=x_rng), yaxis_title=None, height=350)
            
            st.plotly_chart(fig2, use_container_width=True, key=f"comp_{col}")
            
            exp = EXPLICACIONES_CARRERA.get(col, EXPLICACION_DEFAULT)
            st.info(f"💡 **Qué mide este gráfico:** {exp}")
            
        st.divider()

    # --- TEXTO: NUBE Y COMENTARIOS ---
    st.markdown("### ☁️ Comentarios y Opiniones")
    
    # Nube de Palabras
    st.markdown("#### Palabras Clave")
    if 'Palabras_Clave' in df_f.columns:
        txt = " ".join(df_f['Palabras_Clave'].dropna().astype(str))
        if len(txt) > 5:
            wc = WordCloud(width=1200, height=400, background_color='white').generate(txt)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.imshow(wc); ax.axis("off")
            plt.close(fig)
            st.pyplot(fig)
        else:
            st.info("No hay suficientes datos para generar la nube.")
    
    st.divider()

    # --- SECCIÓN OPINIONES (3 COLUMNAS TEMÁTICAS) ---
    st.markdown("#### Últimas Opiniones")
    
    text_cols_display = ['Opinion_Mejoras', 'Opinion_Aula_Virtual', 'Opinion_Charla']
    df_comments = df_f.dropna(subset=text_cols_display, how='all')
    
    # Controles de Ordenamiento y Cantidad
    c_sort, c_limit = st.columns([1, 1])
    with c_sort:
        sort_mode = st.selectbox("Ordenar por:", ["Más Recientes", "Longitud (Texto)"])
    with c_limit:
        limit_mode = st.selectbox("Mostrar:", [10, 20, 50, "Todos"], index=0)

    # Lógica de Ordenamiento
    if not df_comments.empty:
        if sort_mode == "Más Recientes" and 'Timestamp' in df_comments.columns:
            df_comments = df_comments.sort_values('Timestamp', ascending=False)
        elif sort_mode == "Longitud (Texto)":
            df_comments['text_len'] = df_comments[text_cols_display].astype(str).sum(axis=1).str.len()
            df_comments = df_comments.sort_values('text_len', ascending=False)

        # Lógica de Limite
        if limit_mode != "Todos":
            df_comments = df_comments.head(int(limit_mode))
        
        # Display en 3 columnas Temáticas
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("##### 💡 Aspectos a Mejorar / Buenos")
            for idx, row in df_comments.iterrows():
                if pd.notna(row.get('Opinion_Mejoras')) and len(str(row['Opinion_Mejoras'])) > 3:
                    with st.container(border=True):
                        st.caption(f"👤 {row.get('Carrera', '')} ({row.get('Laboratorio', '')})")
                        st.markdown(f"{row['Opinion_Mejoras']}")

        with c2:
            st.markdown("##### 💻 Aula Virtual")
            for idx, row in df_comments.iterrows():
                if pd.notna(row.get('Opinion_Aula_Virtual')) and len(str(row['Opinion_Aula_Virtual'])) > 3:
                    with st.container(border=True):
                        st.caption(f"👤 {row.get('Carrera', '')} ({row.get('Laboratorio', '')})")
                        st.markdown(f"{row['Opinion_Aula_Virtual']}")

        with c3:
            st.markdown("##### 🗣️ Charla TP3")
            for idx, row in df_comments.iterrows():
                if pd.notna(row.get('Opinion_Charla')) and len(str(row['Opinion_Charla'])) > 3:
                    with st.container(border=True):
                        st.caption(f"👤 {row.get('Carrera', '')} ({row.get('Laboratorio', '')})")
                        st.markdown(f"{row['Opinion_Charla']}")
    else:
        st.info("No hay comentarios de texto para los filtros seleccionados.")
    
    st.divider()

    # --- DATOS CRUDOS ---
    with st.expander("📂 Ver Base de Datos Completa (Descargable)"):
        st.dataframe(df_f)

elif src == "Subir Archivo (.xlsx / .csv)" and not st.session_state.get('uploaded_file'):
    st.info("Sube un archivo para comenzar.")
elif src == "Pegar Link de Google Sheet":
    st.info("Pega el link arriba.")

st.markdown("---")
# --- FOOTER ---
st.markdown(f"""
<div style="text-align: center; color: grey; padding-top: 20px;">
    <p>Desarrollado por: <b>J. I. Peralta</b> & <b>Gemini Pro 3.0</b> | Fecha: 05/12/2025</p>
    <p>
        <a href="mailto:jperalta@untref.edu.ar" style="color: grey; text-decoration: none;">📧 jperalta@untref.edu.ar</a> | 
        <a href="https://www.linkedin.com/in/juaniperalta/" style="color: grey; text-decoration: none;">🔗 LinkedIn</a>
    </p>
</div>
""", unsafe_allow_html=True)
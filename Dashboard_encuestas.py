"""
Dashboard de Análisis de Encuestas - Laboratorio de Física
---------------------------------------------------------
Autores: J. I. Peralta & Gemini Pro 3.0
Fecha: 05/12/2025

Descripción:
Aplicación web interactiva desarrollada con Streamlit para el análisis de encuestas
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURACIÓN DE DATOS POR DEFECTO ---
LINK_OFICIAL_ENCUESTA = "https://docs.google.com/spreadsheets/d/1xiz_2A3bWK5vAd6MkCIC0dIXfiMcYqs3UpCQvRtZ1Mg/edit?usp=sharing" 

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Resultados Encuesta del laboratorio de Física",
    page_icon="📊",
    layout="wide"
)

# --- GESTIÓN DE ESTADO ---
if 'lab_filter' not in st.session_state: st.session_state.lab_filter = 'Todos'
if 'car_filter' not in st.session_state: st.session_state.car_filter = 'Todas'
if 'doc_filter' not in st.session_state: st.session_state.doc_filter = 'Todos'

def reset_filters():
    st.session_state.lab_filter = 'Todos'
    st.session_state.car_filter = 'Todas'
    st.session_state.doc_filter = 'Todos'

# --- LISTAS Y DICCIONARIOS DE CONFIGURACIÓN ---

DOCENTES_OFICIALES = [
    "Chaparro, Fabiana", "Dragone, Esteban", "Leone, Emiliano", "Merlo, Rafael", 
    "Orozco Gil, Stefanía", "Oviedo, Carla", "Peralta, Juan Ignacio", 
    "Romeo, Martín", "Vieytes, Mariela", "Villalba, Martín"
]

NORMALIZACION_DOCENTES = {
    "Esteban Dragone": "Dragone, Esteban", "Rafael Merlo": "Merlo, Rafael",
    "Carla Oviedo": "Oviedo, Carla", "Stefanía Orozco": "Orozco Gil, Stefanía",
    "Stefanía Orozco Gil": "Orozco Gil, Stefanía", "Estefanía Orozco Gil": "Orozco Gil, Stefanía",
    "Orozco Gil, Estefanía": "Orozco Gil, Stefanía", "Juan Ignacio Peralta": "Peralta, Juan Ignacio",
    "Mariela Vieytes": "Vieytes, Mariela", "Martín Villalba": "Villalba, Martín",
    "Emiliano Leone": "Leone, Emiliano", "Emiliano Chaparro": "Chaparro, Fabiana",
    "Fabiana Chaparro": "Chaparro, Fabiana", "Martín Romeo": "Romeo, Martín",
    "Stefanía Orozco": "Orozco Gil, Stefanía"
}

# Mapa de preguntas (Base para buscar columnas conocidas)
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

EXPLICACIONES_CARRERA = {
    'Calif_Guias': "Evalúa la accesibilidad y claridad del material escrito.",
    'Calif_Videos': "Mide la utilidad percibida del material audiovisual de apoyo.",
    'Calif_Coord_Teoria': "Analiza la sincronización percibida entre teoría y práctica.",
    'Calif_Docentes_Expl': "Cuantifica la claridad expositiva del equipo docente.",
    'Calif_Correcciones': "Mide la valoración del feedback recibido en los informes.",
    'Calif_Impacto_Aprendizaje': "Indicador de relevancia académica y comprensión profunda.",
}
EXPLICACION_DEFAULT = "Promedio de satisfacción desagregado por carrera."

# --- FUNCIONES DE CARGA Y PROCESAMIENTO ---

@st.cache_data(ttl=600)
def load_data(file_or_url, is_url=False):
    """Carga datos y normaliza nombres de columnas de forma segura."""
    try:
        df = None
        if is_url:
            url = file_or_url
            if "docs.google.com/spreadsheets" in url:
                if "/edit" in url: url = url.split("/edit")[0] + "/export?format=csv"
                elif "/view" in url: url = url.split("/view")[0] + "/export?format=csv"
            df = pd.read_csv(url)
        else:
            if file_or_url.name.endswith('.csv'): df = pd.read_csv(file_or_url)
            else: df = pd.read_excel(file_or_url)
        
        # 1. Fusión Robusta de Carreras
        # Reemplazar espacios vacíos por NaN para asegurar funcionamiento de fillna
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
        elif 'Carrera' not in df.columns:
            carrera_candidates = [c for c in df.columns if 'Carrera' in c]
            if carrera_candidates:
                df['Carrera'] = df[carrera_candidates[0]]
            else:
                df['Carrera'] = 'Sin Especificar'

        # Limpieza: Eliminar las columnas originales F1/F2 para que no ensucien el análisis de texto
        cols_to_drop = [c for c in ['Carrera (F1)', 'Carrera (F2)'] if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

        # 2. Renombrado Seguro
        df.rename(columns={k: v for k, v in PREGUNTA_MAP.items() if k in df.columns}, inplace=True)
        
        # 3. Limpieza de Metadata
        if 'Timestamp' in df.columns: 
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)

        for c in ['Laboratorio', 'Carrera', 'Docentes']:
            if c not in df.columns: df[c] = 'Sin Especificar'
            else: df[c] = df[c].fillna('Sin Especificar').astype(str)

        return df
    except Exception as e:
        if is_url and not file_or_url: return None
        st.error(f"Error al cargar datos: {e}")
        return None

# JIP: Método que mira la varianza y tipo de contenido
def identify_columns(df):
    numeric_cols = []
    text_cols = []
    
    ignore_cols = [
        'Timestamp', 'Laboratorio', 'Carrera', 'Docentes', 
        'Docentes_List', 'Score_Global', 'Carrera (F1)', 'Carrera (F2)'
    ]

    for col in df.columns:
        if col in ignore_cols: continue
        
        # Numérico (Ratings)
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].max() <= 10: numeric_cols.append(col)
        
        # Texto (Opiniones)
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # 1. Filtramos nulos
            valid_data = df[col].dropna().astype(str)
            if len(valid_data) == 0: continue
            
            # 2. Criterio de "Riqueza":
            # Si el 80% de las respuestas son únicas, probablemente es texto libre.
            # Si hay pocas respuestas únicas (ej: "Si", "No"), es una categoría.
            n_unique = valid_data.nunique()
            ratio_unique = n_unique / len(valid_data)
            
            # 3. Criterio de Longitud (reaseguro)
            # Solo si la longitud promedio es decente (>5 chars) Y son variadas
            mean_len = valid_data.str.len().mean()
            
            if ratio_unique > 0.5 and mean_len > 10:
                text_cols.append(col)
                
    return numeric_cols, text_cols


def extract_teachers_from_row(row_str, official_names, mapping_dict):
    found = set()
    row_clean = str(row_str).strip()
    for t in official_names:
        if t in row_clean: found.add(t)
    for variant, official in mapping_dict.items():
        if variant in row_clean: found.add(official)
    if not found and row_clean.lower() not in ['nan', 'sin especificar', '', '0']:
        return [p.strip() for p in row_clean.split(',')]
    return sorted(list(found))


def clean_text_for_wordcloud(text):
    """
    Limpieza ligera específica para la nube de palabras de este laboratorio.
    Normaliza género, elimina conectores y unifica términos comunes.
    """
    if not isinstance(text, str): return ""
    
    # 1. Minúsculas y limpieza básica
    text = text.lower()
    # Reemplazar puntuación por espacios
    for char in [',', '.', '-', ';', '(', ')', '/']:
        text = text.replace(char, ' ')
    
    words = text.split()
    
    # 2. Lista de Stopwords (Palabras a ignorar)
    stopwords = {
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',  'tenía', 
        'y', 'e', 'ni', 'o', 'u', 'de', 'del', 'a', 'al', 'con', 
        'sin', 'por', 'para', 'en', 'sobre', 'que', 'mi', 'tu', 'su',
        'fue', 'muy', 'mas', 'más', 'pero', 'todo', 'laboratorio', 'labo'
    }
    
    # 3. Diccionario de Unificación (Corrección manual ligera)
    # Mapea: Variante -> Palabra Raíz
    replacements = {
        'divertida': 'divertido', 'divertidas': 'divertido',
        'entretenida': 'entretenido', 'entretenidas': 'entretenido',
        'cansadora': 'cansador',
        'estresante': 'estresante', # estres es estresante
        'confusa': 'confuso', 'confusas': 'confuso', 'confusion': 'confuso',
        'buena': 'bueno', 'buenas': 'bueno',
        'utiles': 'util', 'útil': 'util', 'útiles': 'util',
        'prácticos': 'practico', 'práctica': 'practico', 'practica': 'practico',
        'dinamica': 'dinamico', 'dinámica': 'dinamico',
        'interesante': 'interesante', 'interesantes': 'interesante',
        'exigente': 'exigente',
        'organizacion': 'organizado', 'desorden': 'desorganizado',
        'dificil': 'difícil'
    }
    
    cleaned_words = []
    for w in words:
        if w in stopwords or len(w) < 2: continue
        # Aplicar reemplazo si existe, sino dejar la palabra original
        w_clean = replacements.get(w, w)
        cleaned_words.append(w_clean)
        
    return " ".join(cleaned_words)


# JIP: Versión que tiene en cuenta las negaciones
def calcular_sentimiento(df_input, text_columns):
    """Calcula sentimiento detectando negaciones antes de adjetivos."""
    if not text_columns: return "Sin datos", "#808080"
    
    # Listas ampliadas
    pos = {'bueno', 'buena', 'buenos', 'buenas', 'excelente', 'excelentes', 'util', 'utiles', 'claro', 
           'clara', 'claras', 'claros', 'mejor', 'mejores', 'bien', 'gusto', 'gustó', 'sirvio', 'sirvió', 
           'aprendizaje', 'dinamica', 'dinámico', 'dinamico', 'correcto', 'correcta', 'interesante', 
           'interesantes', 'llevadero', 'llevadera'}
    neg = {'malo', 'mala', 'malos', 'malas', 'confuso', 'confusa', 'dificil', 'difícil', 'complicado', 
           'complicada', 'tarde', 'desorganizado', 'pesimo', 'pésimo', 'lento', 'lenta', 'poco', 
           'injusto', 'perdido', 'aburrido', 'aburrida', 'tedioso', 'pesado', 'pesada'}
    negators = {'no', 'nunca', 'jamás', 'poco', 'menos', 'nada'}
    
    score = 0; count = 0
    
    for _, row in df_input.iterrows():
        # Unir y limpiar
        full_text = " ".join([str(row[c]) for c in text_columns if c in row and pd.notna(row[c])]).lower()
        # Quitar puntuación básica para tokenizar bien
        for char in ['.', ',', ';', '!', '?']: full_text = full_text.replace(char, ' ')
        
        words = full_text.split()
        row_score = 0
        
        for i, w in enumerate(words):
            val = 0
            if w in pos: val = 1
            elif w in neg: val = -1
            
            # Chequear negación en la palabra anterior (ventana de 1)
            if val != 0 and i > 0 and words[i-1] in negators:
                val *= -1.5 # Invierte y da peso (ej: "no bueno" -> malo)
            
            row_score += val
            
        # Normalizar el score de la fila entre -3 y 3
        row_score = max(min(row_score, 3), -3)
        if row_score != 0: # Solo contar si hubo algo de sentimiento detectado
            score += row_score
            count += 1
        
    if count == 0: return "Neutro / Sin Texto", "#808080"
    
    avg = score / count
    
    # Umbrales ajustados
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

st.title("📊 Resultados Encuesta del laboratorio de Física")

opciones_fuente = ["📊 Datos Oficiales (Cargados)", "🔗 Pegar Link de Google Sheet", "📂 Subir Archivo (.xlsx / .csv)"]
src = st.radio("Fuente de Datos:", opciones_fuente, horizontal=True, index=0)

df = None

if src == "📊 Datos Oficiales (Cargados)":
    if LINK_OFICIAL_ENCUESTA:
        df = load_data(LINK_OFICIAL_ENCUESTA, is_url=True)
    else:
        st.warning("⚠️ No se ha configurado el link oficial.")
elif src == "📂 Subir Archivo (.xlsx / .csv)":
    up = st.file_uploader("Archivo", type=['csv', 'xlsx'])
    if up: df = load_data(up, False)
else:
    url = st.text_input("Link Público:")
    if url: df = load_data(url, True)

if df is not None:
    # 1. Identificar columnas dinámicamente
    rating_cols, text_cols = identify_columns(df)
    
    # Forzar la inclusión de Palabras_Clave para el sentimiento, aunque sea texto corto
    if 'Palabras_Clave' in df.columns and 'Palabras_Clave' not in text_cols:
        text_cols.append('Palabras_Clave')

    # 2. Procesar Docentes
    if 'Docentes' in df.columns:
        df['Docentes_List'] = df['Docentes'].apply(lambda x: extract_teachers_from_row(x, DOCENTES_OFICIALES, NORMALIZACION_DOCENTES))

    # --- FILTROS ---
    st.sidebar.header("Filtros")
    if st.sidebar.button("🔄 Borrar Filtros", on_click=reset_filters, type="primary"): st.rerun()

    sel_lab = st.sidebar.selectbox("Laboratorio", ['Todos'] + sorted(df['Laboratorio'].unique()), key='lab_filter')
    sel_car = st.sidebar.selectbox("Carrera", ['Todas'] + sorted(df['Carrera'].unique()), key='car_filter')
    
    all_docs = set()
    if 'Docentes_List' in df.columns:
        for l in df['Docentes_List']: all_docs.update(l)
    sel_doc = st.sidebar.selectbox("Docente (Presente)", ['Todos'] + sorted(list(all_docs)), key='doc_filter')

    st.sidebar.markdown("---")
    st.sidebar.header("Visualización")
    viz_mode = st.sidebar.radio("Unidad:", ("Porcentaje (%)", "Cantidad Absoluta/Escala"), index=0)
    is_pct = (viz_mode == "Porcentaje (%)")

    # --- APLICAR FILTROS ---
    df_f = df.copy()
    if sel_lab != 'Todos': df_f = df_f[df_f['Laboratorio'] == sel_lab]
    if sel_car != 'Todas': df_f = df_f[df_f['Carrera'] == sel_car]
    if sel_doc != 'Todos': df_f = df_f[df_f['Docentes_List'].apply(lambda x: sel_doc in x)]

    # --- KPI & SENTIMIENTO ---
    tot, filt = len(df), len(df_f)
    pct = (filt/tot*100) if tot>0 else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Encuestas", f"{filt} de {tot}", f"{pct:.1f}% muestra")

    # sent_txt, sent_col = calcular_sentimiento(df_f, text_cols) # JIP: detección automática
    # Ejemplo manual:
    text_cols_manual = ['Opinion_Mejoras', 'Palabras_Clave']
    sent_txt, sent_col = calcular_sentimiento(df_f, text_cols_manual)

    with c2:
        st.markdown(f"<div style='background:{sent_col};color:white;padding:10px;border-radius:5px;text-align:center'><b>{sent_txt}</b></div>", unsafe_allow_html=True)
    with c3:
        fuentes_sentimiento = [c.replace('Opinion_', '').replace('_', ' ') for c in text_cols_manual]
        st.caption(f"**Fuentes del análisis:** Preg. {' + '.join(fuentes_sentimiento)}.")
    
    st.divider()

    # --- 1. RESUMEN GENERAL ---
    st.markdown("### 📈 Resumen General y Comparativa")
    
    if rating_cols:
        avgs = df_f[rating_cols].mean()
        for i in range(0, len(rating_cols), 4):
            cols_kpi = st.columns(4)
            for j, col in enumerate(rating_cols[i:i+4]):
                with cols_kpi[j]:
                    val = avgs[col]
                    lbl = col.replace('Calif_', '').replace('_', ' ') if col.startswith('Calif_') else col
                    d_val = f"{(val/5)*100:.0f}%" if is_pct else f"{val:.2f}"
                    st.metric(lbl, d_val)
        
        st.write("") 
        
        # Comparativa Global por Carrera
        st.markdown("#### 🆚 Comparativa Global de Satisfacción por Carrera")
        df_f['Score_Global'] = df_f[rating_cols].mean(axis=1)
        career_global = df_f.groupby('Carrera')['Score_Global'].agg(['mean', 'count']).reset_index().sort_values('mean', ascending=True)
        
        if is_pct:
            career_global['Display'] = (career_global['mean'] / 5) * 100
            x_ax = 'Display'; x_title = "% Satisfacción Global"; txt_fmt = '.1f'; txt_suf = '%'
            range_x = [0, 110]
        else:
            career_global['Display'] = career_global['mean']
            x_ax = 'Display'; x_title = "Promedio General (1-5)"; txt_fmt = '.2f'; txt_suf = ''
            range_x = [1, 5.8] 
            
        career_global['Label'] = career_global.apply(lambda x: f"{x['Carrera']} (N={int(x['count'])})", axis=1)
        
        fig_global = px.bar(career_global, y='Label', x=x_ax, text=x_ax, orientation='h', 
                            color=x_ax, color_continuous_scale='Teal')
        fig_global.update_traces(texttemplate='%{text:' + txt_fmt + '}' + txt_suf, textposition='outside')
        fig_global.update_layout(xaxis=dict(range=range_x, title=x_title), yaxis_title=None, height=400)
        st.plotly_chart(fig_global, use_container_width=True, key="global_chart")

    st.divider()

    # --- 2. DETALLE POR PREGUNTA ---
    st.markdown("### 📝 Análisis Detallado por Pregunta")
    
    for col in rating_cols:
        title = REVERSE_MAP.get(col, col)
        st.subheader(f"📌 {title}")
        
        c_left, c_right = st.columns([1, 1])
        
        with c_left:
            cnt = df_f[col].value_counts().reindex(sorted(df_f[col].unique()), fill_value=0).reset_index()
            cnt.columns = ['Puntaje', 'Valor']
            
            y_val = 'Valor'
            if is_pct:
                tot_v = cnt['Valor'].sum()
                cnt['Pct'] = (cnt['Valor']/tot_v*100).fillna(0)
                y_val = 'Pct'; y_title = "%"; y_max = 115; txt_t = '%{y:.1f}%'
            else:
                y_title = "Votos"; y_max = cnt['Valor'].max() * 1.2; txt_t = '%{y}'
            
            fig1 = px.bar(cnt, x='Puntaje', y=y_val, text=y_val, title=f"Distribución ({y_title})",
                          color=y_val, color_continuous_scale='Blues')
            fig1.update_traces(texttemplate=txt_t, textposition='outside')
            fig1.update_layout(yaxis=dict(range=[0, y_max], title=y_title), height=350)
            st.plotly_chart(fig1, use_container_width=True, key=f"dist_{col}")

        with c_right:
            cg = df_f.groupby('Carrera')[col].agg(['mean', 'count']).reset_index().sort_values('mean', ascending=True)
            
            if is_pct:
                cg['Val'] = (cg['mean']/5)*100
                x_rng = [0, 115]; fmt = '.1f'; suf = '%'; t_suf = "(% Satisfacción)"
            else:
                cg['Val'] = cg['mean']
                x_rng = [1, 5.8]; fmt = '.2f'; suf = ''; t_suf = "(Escala 1-5)"
                
            cg['Lbl'] = cg.apply(lambda x: f"{x['Carrera']} (N={int(x['count'])})", axis=1)
            
            fig2 = px.bar(cg, y='Lbl', x='Val', text='Val', orientation='h',
                          title=f"Promedio por Carrera {t_suf}",
                          color='Val', color_continuous_scale='Teal')
            fig2.update_traces(texttemplate='%{text:'+fmt+'}'+suf, textposition='outside')
            fig2.update_layout(xaxis=dict(range=x_rng), yaxis_title=None, height=350)
            st.plotly_chart(fig2, use_container_width=True, key=f"comp_{col}")
            
            exp = EXPLICACIONES_CARRERA.get(col, EXPLICACION_DEFAULT)
            st.info(f"💡 {exp}")
            
        st.divider()

    # --- 3. COMENTARIOS ---
    st.markdown("### ☁️ Comentarios y Opiniones")
    
    # Nube (Exclusiva de Palabras Clave)
    cloud_col = 'Palabras_Clave'
    if cloud_col in df_f.columns:
        raw_txt = " ".join(df_f[cloud_col].dropna().astype(str))
        txt_cloud = clean_text_for_wordcloud(raw_txt) # JIP: Limpieza
        if len(txt_cloud) > 10:
            st.markdown("#### Palabras Clave Globales")
            st.caption("Fuente: Pregunta 'Escribí tres palabras...'")
            wc = WordCloud(width=1200, height=400, background_color='white', colormap='viridis', regexp=r"\w+", random_state=42).generate(txt_cloud)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.imshow(wc); ax.axis("off")
            plt.close(fig)
            st.pyplot(fig)
    
    st.divider()

    # Opiniones en 3 Columnas Dinámicas
    st.markdown("#### Últimas Opiniones")
    
    # Identificar columnas de texto que NO son palabras clave
    display_text_cols = [c for c in text_cols if c != 'Palabras_Clave']
    
    if display_text_cols:
        # Filtramos filas vacías
        df_comments = df_f.dropna(subset=display_text_cols, how='all')
        
        c_sort, c_limit = st.columns([1, 1])
        with c_sort: sort_mode = st.selectbox("Ordenar por:", ["Más Recientes", "Longitud (Texto)"])
        with c_limit: limit_mode = st.selectbox("Mostrar:", [10, 20, 50, "Todos"], index=0)

        if not df_comments.empty:
            if sort_mode == "Más Recientes" and 'Timestamp' in df_comments.columns:
                df_comments = df_comments.sort_values('Timestamp', ascending=False)
            elif sort_mode == "Longitud (Texto)":
                df_comments['text_len'] = df_comments[display_text_cols].astype(str).sum(axis=1).str.len()
                df_comments = df_comments.sort_values('text_len', ascending=False)

            if limit_mode != "Todos": df_comments = df_comments.head(int(limit_mode))
            
            # --- LAYOUT DE 3 COLUMNAS ADAPTATIVO ---
            # Tomamos hasta 3 columnas de texto. Si hay 2, usa 2. Si hay 4, usa las primeras 3.
            # Esto evita errores si falta una columna o hay nuevas.
            cols_to_render = display_text_cols[:3]
            cols = st.columns(len(cols_to_render))
            
            for i, col_name in enumerate(cols_to_render):
                header = REVERSE_MAP.get(col_name, col_name)
                header = header.replace('Opinion_', '').replace('_', ' ').title()
                
                with cols[i]:
                    st.markdown(f"##### 🗣️ {header}")
                    for idx, row in df_comments.iterrows():
                        val = row.get(col_name)
                        if pd.notna(val) and len(str(val)) > 3:
                            with st.container(border=True):
                                st.caption(f"👤 {row.get('Carrera', '')} ({row.get('Laboratorio', '')})")
                                st.markdown(f"{val}")
        else:
            st.info("No hay comentarios disponibles.")
    else:
        st.info("No se detectaron columnas de comentarios en este archivo.")

    st.divider()
    

    with st.expander("📂 Ver Base de Datos (Filtros actuales)"):
        st.dataframe(df_f)

elif src == "📂 Subir Archivo (.xlsx / .csv)" and not st.session_state.get('uploaded_file'):
    st.info("Sube un archivo del formulario para comenzar.")
elif src == "🔗 Pegar Link de Google Sheet":
    st.info("Pega el link arriba.")

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: grey; padding-top: 20px;">
    <p>Desarrollado por: <b>J. I. Peralta</b> & <b>Gemini Pro 3.0</b> | Fecha: 05/12/2025</p>
    <p>
        <a href="mailto:jperalta@untref.edu.ar" style="color: grey; text-decoration: none;">📧 jperalta@untref.edu.ar</a> | 
        <a href="https://www.linkedin.com/in/juaniperalta/" style="color: grey; text-decoration: none;">🔗 LinkedIn</a>
    </p>
</div>
""", unsafe_allow_html=True)
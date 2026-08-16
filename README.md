---
title: Dashboard Untref Laboratorio
emoji: 📊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.39.0
app_file: Dashboard_encuestas.py
pinned: false
---

![Vistas](https://visitor-badge.laobi.icu/badge?page_id=grapako.dashboard-untref-laboratorio)

# 📊 Resultados de las encuestas del Laboratorio de Física (UNTREF)

Este repositorio contiene el código fuente de una aplicación web interactiva desarrollada para analizar las encuestas de satisfacción de los estudiantes del Laboratorio de Física.

La herramienta permite al equipo docente y directivo visualizar métricas clave, leer opiniones cualitativas y detectar áreas de mejora en tiempo real.

## 🚀 Acceso a la Aplicación

El dashboard está disponible en dos versiones equivalentes, con el mismo diseño y funcionalidades:

- **App Streamlit (Python):** 👉 **[Abrir en Streamlit](https://encuestas-laboratorio-untref.streamlit.app/)**
- **Versión web estática (HTML/JS):** 👉 **[Abrir versión web](https://grapako.github.io/dashboard-untref-laboratorio/)** — corre 100% en el navegador, sin backend.

## ✨ Funcionalidades Principales

- **Carga de Datos:** Conexión directa con Google Sheets o carga de archivos CSV/Excel.
- **Filtros Dinámicos:** Segmentación por Materia, Laboratorio y Carrera.
- **Indicadores (KPIs):** Promedios de satisfacción general y comparativas, en porcentaje o escala 1-5.
- **Visualización:**
  - Distribuciones de respuestas por pregunta (Escala 1-5).
  - Comparativa de satisfacción global y por pregunta, desagregada por Carrera.
- **Análisis de Texto:**
  - Nube de palabras clave, con selector de paleta de color.
  - Clasificación automática de sentimiento (Positivo/Neutro/Negativo) a partir de las respuestas abiertas.
  - Explorador de opiniones por pregunta, con orden y límite configurables.
- **Modo claro / oscuro**, con soporte para seguir la preferencia del sistema operativo.

## 🛠️ Tecnologías Utilizadas

**App Streamlit:**
- **Python**, **Streamlit**, **Pandas**, **Plotly**, **WordCloud**, **Matplotlib**

**Versión web estática** (`docs/`):
- **HTML / CSS / JavaScript** puro (sin frameworks ni build step)
- **Plotly.js**, **PapaParse** (CSV), **SheetJS** (Excel), **wordcloud2.js**, todo vía CDN

## 👥 Créditos

**Desarrollado por:**
- **Juan Ignacio Peralta** (Departamento de Ciencia y Tecnología - UNTREF)
- Asistencia técnica de IA (Claude Sonnet 5, Anthropic)

---
*Universidad Nacional de Tres de Febrero (UNTREF)*
/*
 * Versión estática del tablero. Todo el procesamiento ocurre en el navegador:
 * no se suben archivos ni respuestas a un servidor.
 */

// const OFFICIAL_SHEET_URL = "https://docs.google.com/spreadsheets/d/1xiz_2A3bWK5vAd6MkCIC0dIXfiMcYqs3UpCQvRtZ1Mg/edit?usp=sharing";
const OFFICIAL_SHEET_URL = "https://docs.google.com/spreadsheets/d/14wzNg71_McoDbpw-yVrYBXQlB9NMuH1W_ATytBw4cJI/edit?usp=sharing"
const SHOW_TEACHER_FILTER = false;

const QUESTION_MAP = {
  "Marca temporal": "Timestamp",
  "Asistí al": "Laboratorio",
  "Docentes de Laboratorio": "Docentes",
  "¿Qué tan fáciles de entender le resultaron las guías de laboratorio?": "Calif_Guias",
  "¿Le parecieron útiles los videos subidos al aula para explicar las guías de laboratorio?": "Calif_Videos",
  "¿Cómo evaluaría la coordinación entre los TEMAS tratados en las clases teóricas y en el laboratorio?": "Calif_Coord_Teoria",
  "¿Le resultaron claras las explicaciones de los docentes del laboratorio en las clases presenciales?": "Calif_Docentes_Expl",
  "¿Le resultaron útiles las correcciones del docente luego de cada informe?": "Calif_Correcciones",
  "¿Considera que los conocimientos adquiridos en la cursada de laboratorio le sirvieron para entender de una forma más profunda los fenómenos físicos vistos de forma teórica en la materia?": "Calif_Impacto_Aprendizaje",
  "Escribí tres  palabras que describan tu experiencia en el laboratorio (NO MAS por favor! y no poner artículos ni preposiciones)": "Palabras_Clave",
  "¿Qué opinión tenés respecto del uso del aula virtual para manejar la cursada de labo y el material multimedia subido en la misma? En lo posible enumerá pros y contras. ": "Opinion_Aula_Virtual",
  "¿Qué cosas mejorarías, y qué cosas te parecieron buenas de la cursada de laboratorio?": "Opinion_Mejoras",
  "¿Qué te pareció la idea de dar una charla para el tercer TP en reemplazo del informe?": "Opinion_Charla"
};

const REVERSE_MAP = Object.fromEntries(Object.entries(QUESTION_MAP).map(([key, value]) => [value, key]));
const KPI_NAMES = {
  Calif_Guias: "Claridad de las Guías",
  Calif_Videos: "Utilidad de los Videos",
  Calif_Coord_Teoria: "Coordinación con la Teoría",
  Calif_Docentes_Expl: "Explicación de los Docentes",
  Calif_Correcciones: "Valor de las Correcciones",
  Calif_Impacto_Aprendizaje: "Impacto en el Aprendizaje"
};
const CAREER_EXPLANATIONS = {
  Calif_Guias: "Indica si las guías permiten la autonomía del estudiante o si representan una dificultad que requiere gran intervención docente para interpretar la consigna.",
  Calif_Videos: "Un valor alto sugiere que el material asincrónico libera tiempo de clase presencial resolviendo dudas previamente.",
  Calif_Coord_Teoria: "Percepción de sincronía entre teoría y práctica. Valores bajos indican que el alumno asiste sin el marco conceptual necesario.",
  Calif_Docentes_Expl: "Percepción sobre la capacidad del equipo docente para transmitir conceptos complejos y resolver dudas en clase.",
  Calif_Correcciones: "¿El alumno utiliza la corrección para mejorar el siguiente trabajo, o la percibe solo como una penalización?",
  Calif_Impacto_Aprendizaje: "Percepción de si el Laboratorio cumple su función de consolidar los conceptos teóricos, o se percibe como una materia aislada."
};
const SCALE_REFERENCES = {
  Calif_Guias: ["Muy difíciles", "Muy claras"],
  Calif_Videos: ["Nada útiles", "Excelentes"],
  Calif_Coord_Teoria: ["Los temas no habían sido vistos en las clases teóricas", "Todos los temas fueron vistos en las clases teóricas"],
  Calif_Docentes_Expl: ["Nunca se entendía nada", "Eran siempre claras"],
  Calif_Correcciones: ["No servían para nada", "Fueron muy útiles"],
  Calif_Impacto_Aprendizaje: ["Para nada", "Absolutamente"]
};

const OFFICIAL_TEACHERS = [
  "Chaparro, Fabiana", "Dragone, Esteban", "Leone, Emiliano", "Merlo, Rafael",
  "Orozco Gil, Stefanía", "Oviedo, Carla", "Peralta, Juan Ignacio",
  "Romeo, Martín", "Vieytes, Mariela", "Villalba, Martín"
];
const TEACHER_NORMALIZATION = {
  "Esteban Dragone": "Dragone, Esteban", "Rafael Merlo": "Merlo, Rafael",
  "Carla Oviedo": "Oviedo, Carla", "Stefanía Orozco": "Orozco Gil, Stefanía",
  "Stefanía Orozco Gil": "Orozco Gil, Stefanía", "Estefanía Orozco Gil": "Orozco Gil, Stefanía",
  "Orozco Gil, Estefanía": "Orozco Gil, Stefanía", "Juan Ignacio Peralta": "Peralta, Juan Ignacio",
  "Mariela Vieytes": "Vieytes, Mariela", "Martín Villalba": "Villalba, Martín",
  "Emiliano Leone": "Leone, Emiliano", "Emiliano Chaparro": "Chaparro, Fabiana",
  "Fabiana Chaparro": "Chaparro, Fabiana", "Martín Romeo": "Romeo, Martín"
};

const STOP_WORDS = new Set([
  "el", "la", "los", "las", "un", "una", "unos", "unas", "tenia", "algun", "y", "e", "ni", "o", "u", "de", "del", "a", "al", "con", "os", "alumnos", "sin", "por", "para", "en", "sobre", "que", "mi", "tu", "su", "parte", "fue", "muy", "mas", "pero", "todo", "laboratorio", "labo", "año", "profesores"
]);
const REPLACEMENTS = {
  acelerada: ["rapido", "rapida", "aceleracion", "frenetica"], aprendizaje: ["aprendizage", "conocimiento"], buena: ["bueno", "buenas", "buen", "bien"], cansadora: ["cansador"], colaborativa: ["grupal", "equipo", "colaboracion"], confusa: ["confuso", "confusas", "confusos", "confusion"], desafiante: ["desafio"], desincronizada: ["desarticulada"], desorganizada: ["desorden", "desorganizado", "desorganizadas", "caos", "erratica"], "didáctica": ["didactico"], "difícil": ["complejo", "compleja", "complicado", "complicada"], "dinámica": ["dinamico", "dinamicos", "dinamismo", "fluida"], divertida: ["divertido", "divertidas", "divertidos"], esclarecedora: ["esclarecedor", "clara", "aclarador", "entendible", "escalrecedor", "escalrecedora", "explicativo", "ilustrativa", "reveladora"], entretenida: ["entretenido", "entretenidas", "entretenidos"], enriquecedora: ["enriquecera", "enriquecedor"], estresante: ["estrs", "estres"], exigente: ["estricta", "intenso", "esfuerzo", "presion", "laboriosa", "laborioso"], integradora: ["integrador"], llevadera: ["llevado", "llevadero"], "útil": ["utiles"], pesada: ["pesado"], "práctica": ["practica", "practicos", "practico", "practicidad", "aplicativo"], precisa: ["precision", "preciso"], organizado: ["organizacion", "organizada", "organizadas"], "técnica": ["tecnica", "tecnico"], satisfactoria: ["recompensante"], "versátil": []
};

const readPreference = (key, fallback) => {
  try { return localStorage.getItem(key) || fallback; }
  catch { return fallback; }
};
const savePreference = (key, value) => {
  try { localStorage.setItem(key, value); }
  catch { /* El tablero sigue funcionando aunque el navegador bloquee el almacenamiento local. */ }
};
const WORD_CLOUD_PALETTES = {
  Pastel: ["#7c3aed", "#a21caf", "#be185d", "#c2410c", "#a16207", "#4d7c0f", "#15803d", "#0f766e", "#0369a1", "#4338ca"],
  Viridis: ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#d4e21a"],
  Tropical: ["#e63946", "#f77f00", "#fcbf49", "#06a77d", "#118ab2", "#073b4c", "#d62828", "#f77f00", "#fcbf49", "#06a77d"]
};
const state = {
  source: "official",
  data: null,
  columns: [],
  ratingColumns: [],
  textColumns: [],
  loading: true,
  error: "",
  filters: { materia: "Todas", laboratorio: "Todos", carrera: "Todas", docente: "Todos" },
  isPercentage: true,
  commentSort: "Últimos",
  commentLimit: "Todos",
  sidebarOpen: readPreference("untref-sidebar-open", "true") !== "false",
  themeMode: readPreference("untref-theme-mode", "system"),
  cloudPalette: readPreference("untref-cloud-palette", "Pastel")
};

const app = document.querySelector("#app");
const isDarkTheme = () => state.themeMode === "dark" || (state.themeMode === "system" && window.matchMedia?.("(prefers-color-scheme: dark)").matches);
const applyTheme = () => {
  const root = document.documentElement;
  if (!root) return;
  if (state.themeMode === "system") delete root.dataset.theme;
  else root.dataset.theme = state.themeMode;
};
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
const isBlank = (value) => value === null || value === undefined || String(value).trim() === "";
const asText = (value) => value instanceof Date ? value.toLocaleString("es-AR") : String(value ?? "");
const numberOrNull = (value) => {
  if (isBlank(value)) return null;
  const normalized = typeof value === "string" ? value.trim().replace(",", ".") : value;
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
};
const uniqueSorted = (items) => [...new Set(items)].sort((a, b) => String(a).localeCompare(String(b), "es"));
const average = (values) => {
  const valid = values.filter((value) => Number.isFinite(value));
  return valid.length ? valid.reduce((total, value) => total + value, 0) / valid.length : null;
};
const prettify = (column) => KPI_NAMES[column] || REVERSE_MAP[column] || column.replace("Opinion_", "").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

function googleCsvUrl(url) {
  const match = String(url).match(/docs\.google\.com\/spreadsheets\/d\/([^/]+)/);
  return match ? `https://docs.google.com/spreadsheets/d/${match[1]}/export?format=csv` : String(url).trim();
}

function parseTimestamp(value) {
  if (value instanceof Date && !Number.isNaN(value.getTime())) return value.getTime();
  const text = String(value ?? "").trim();
  const match = text.match(/^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})(?:[ ,T]+(\d{1,2}):(\d{2})(?::(\d{2}))?)?/);
  if (match) {
    const year = Number(match[3].length === 2 ? `20${match[3]}` : match[3]);
    return new Date(year, Number(match[2]) - 1, Number(match[1]), Number(match[4] || 0), Number(match[5] || 0), Number(match[6] || 0)).getTime();
  }
  const time = Date.parse(text);
  return Number.isNaN(time) ? 0 : time;
}

function extractTeachers(value) {
  const cleaned = String(value ?? "").trim();
  const found = new Set();
  OFFICIAL_TEACHERS.forEach((teacher) => { if (cleaned.includes(teacher)) found.add(teacher); });
  Object.entries(TEACHER_NORMALIZATION).forEach(([variant, official]) => { if (cleaned.includes(variant)) found.add(official); });
  if (!found.size && !["nan", "sin especificar", "", "0"].includes(cleaned.toLowerCase())) return cleaned.split(",").map((name) => name.trim()).filter(Boolean);
  return [...found].sort((a, b) => a.localeCompare(b, "es"));
}

function normalizeDataset(rawRows) {
  if (!Array.isArray(rawRows) || !rawRows.length) throw new Error("El archivo no contiene filas de datos.");
  const metadata = new Set(["Timestamp", "Laboratorio", "Carrera", "Docentes", "Materia", "Docentes_List", "Score_Global", "Carrera (F1)", "Carrera (F2)"]);
  const data = rawRows.map((raw) => {
    const f1 = raw["Carrera (F1)"];
    const f2 = raw["Carrera (F2)"];
    const row = {};
    Object.entries(raw).forEach(([key, value]) => { if (key !== "Carrera (F1)" && key !== "Carrera (F2)") row[QUESTION_MAP[key] || key] = value; });
    if (!isBlank(f1) && !isBlank(f2)) { row.Carrera = f1; row.Materia = "Física I"; }
    else if (!isBlank(f1)) { row.Carrera = f1; row.Materia = "Física I"; }
    else if (!isBlank(f2)) { row.Carrera = f2; row.Materia = "Física II"; }
    else if (isBlank(row.Carrera)) {
      const candidate = Object.keys(raw).find((key) => key.includes("Carrera"));
      row.Carrera = candidate ? raw[candidate] : "Sin Especificar";
      row.Materia = "Sin Especificar";
    } else if (isBlank(row.Materia)) row.Materia = "Sin Especificar";
    ["Laboratorio", "Carrera", "Docentes", "Materia"].forEach((key) => { row[key] = isBlank(row[key]) ? "Sin Especificar" : asText(row[key]); });
    row.Laboratorio = row.Laboratorio.replace("Laboratorio de ", "").trim();
    return row;
  });
  const columns = [...new Set(data.flatMap((row) => Object.keys(row)))];
  columns.forEach((column) => {
    if (metadata.has(column)) return;
    const populated = data.map((row) => row[column]).filter((value) => !isBlank(value));
    if (populated.length && populated.every((value) => numberOrNull(value) !== null)) data.forEach((row) => { if (!isBlank(row[column])) row[column] = numberOrNull(row[column]); });
  });
  data.forEach((row) => { row.Docentes_List = extractTeachers(row.Docentes); });
  const finalColumns = [...new Set(data.flatMap((row) => Object.keys(row)))];
  const ratings = [];
  const texts = [];
  finalColumns.forEach((column) => {
    if (metadata.has(column)) return;
    const values = data.map((row) => row[column]).filter((value) => !isBlank(value));
    if (!values.length) return;
    if (values.every((value) => Number.isFinite(value)) && Math.max(...values) <= 10) ratings.push(column);
    else {
      const strings = values.map(asText);
      if ((new Set(strings).size / strings.length) > 0.5 && average(strings.map((value) => value.length)) > 10) texts.push(column);
    }
  });
  if (finalColumns.includes("Palabras_Clave") && !texts.includes("Palabras_Clave")) texts.push("Palabras_Clave");
  return { data, columns: finalColumns, ratings, texts };
}

function setDataset(rawRows) {
  const normalized = normalizeDataset(rawRows);
  state.data = normalized.data;
  state.columns = normalized.columns;
  state.ratingColumns = normalized.ratings;
  state.textColumns = normalized.texts;
  state.filters = { materia: "Todas", laboratorio: "Todos", carrera: "Todas", docente: "Todos" };
  state.error = "";
  state.loading = false;
  render();
}

async function loadFromUrl(url) {
  const destination = googleCsvUrl(url);
  if (!destination) return;
  state.loading = true; state.error = ""; render();
  try {
    const response = await fetch(destination);
    if (!response.ok) throw new Error(`No se pudo descargar el archivo (respuesta ${response.status}).`);
    const csv = await response.text();
    const parsed = Papa.parse(csv, { header: true, skipEmptyLines: "greedy" });
    if (parsed.errors.length && !parsed.data.length) throw new Error(parsed.errors[0].message);
    setDataset(parsed.data);
  } catch (error) {
    state.loading = false;
    state.error = `No se pudieron cargar los datos: ${error.message} Si el enlace no admite acceso desde el navegador, descargá el archivo y usá “Subir archivo”.`;
    render();
  }
}

async function loadFromFile(file) {
  if (!file) return;
  state.loading = true; state.error = ""; render();
  try {
    let rows;
    if (/\.csv$/i.test(file.name)) {
      const content = await file.text();
      const parsed = Papa.parse(content, { header: true, skipEmptyLines: "greedy" });
      if (parsed.errors.length && !parsed.data.length) throw new Error(parsed.errors[0].message);
      rows = parsed.data;
    } else if (/\.xlsx$/i.test(file.name)) {
      const workbook = XLSX.read(await file.arrayBuffer(), { type: "array", cellDates: true });
      const worksheet = workbook.Sheets[workbook.SheetNames[0]];
      rows = XLSX.utils.sheet_to_json(worksheet, { defval: null, raw: true });
    } else throw new Error("Elegí un archivo CSV o XLSX.");
    setDataset(rows);
  } catch (error) {
    state.loading = false; state.error = `No se pudo leer el archivo: ${error.message}`; render();
  }
}

function filteredData() {
  if (!state.data) return [];
  return state.data.filter((row) =>
    (state.filters.materia === "Todas" || row.Materia === state.filters.materia) &&
    (state.filters.laboratorio === "Todos" || row.Laboratorio === state.filters.laboratorio) &&
    (state.filters.carrera === "Todas" || row.Carrera === state.filters.carrera) &&
    (state.filters.docente === "Todos" || row.Docentes_List.includes(state.filters.docente))
  );
}

function groupedMean(rows, column) {
  const groups = new Map();
  rows.forEach((row) => {
    const value = numberOrNull(row[column]);
    if (value === null) return;
    const key = row.Carrera || "Sin Especificar";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(value);
  });
  return [...groups.entries()].map(([career, values]) => ({ career, mean: average(values), count: values.length })).sort((a, b) => a.mean - b.mean);
}

function globalCareerMeans(rows) {
  const values = rows.map((row) => ({ ...row, score: average(state.ratingColumns.map((column) => numberOrNull(row[column]))) })).filter((row) => row.score !== null);
  const groups = new Map();
  values.forEach((row) => { if (!groups.has(row.Carrera)) groups.set(row.Carrera, []); groups.get(row.Carrera).push(row.score); });
  return [...groups.entries()].map(([career, scores]) => ({ career, mean: average(scores), count: scores.length })).sort((a, b) => a.mean - b.mean);
}

function removeAccents(text) { return String(text).normalize("NFD").replace(/[\u0300-\u036f]/g, ""); }
const WORD_LOOKUP = (() => {
  const lookup = {};
  Object.entries(REPLACEMENTS).forEach(([root, variants]) => {
    variants.forEach((variant) => { lookup[variant] = root; });
    const unaccented = removeAccents(root);
    if (unaccented !== root) lookup[unaccented] = root;
  });
  return lookup;
})();

function cleanTextForCloud(text) {
  return String(text ?? "").toLowerCase().replace(/[,.;()\/!?'"-]/g, " ").split(/\s+/).filter(Boolean).map((word) => {
    const clean = removeAccents(word);
    if (word.length < 2 || STOP_WORDS.has(clean)) return null;
    return WORD_LOOKUP[clean] || word;
  }).filter(Boolean);
}

function calculateSentiment(rows) {
  const pos = new Set(["bueno", "buena", "buenos", "buenas", "excelente", "excelentes", "util", "utiles", "claro", "clara", "claras", "claros", "mejor", "mejores", "bien", "gusto", "gustó", "sirvio", "sirvió", "aprendizaje", "dinamica", "dinámico", "dinamico", "correcto", "correcta", "interesante", "interesantes", "llevadero", "llevadera"]);
  const neg = new Set(["malo", "mala", "malos", "malas", "confuso", "confusa", "dificil", "difícil", "complicado", "complicada", "tarde", "desorganizado", "pesimo", "pésimo", "lento", "lenta", "poco", "injusto", "perdido", "aburrido", "aburrida", "tedioso", "pesado", "pesada"]);
  const negators = new Set(["no", "nunca", "jamás", "poco", "menos", "nada"]);
  let score = 0; let count = 0;
  rows.forEach((row) => {
    const words = [row.Opinion_Mejoras, row.Palabras_Clave].filter((text) => !isBlank(text)).join(" ").toLowerCase().replace(/[.,;!?]/g, " ").split(/\s+/).filter(Boolean);
    let rowScore = 0;
    words.forEach((word, index) => {
      let value = pos.has(word) ? 1 : neg.has(word) ? -1 : 0;
      if (value && index && negators.has(words[index - 1])) value *= -1.5;
      rowScore += value;
    });
    rowScore = Math.max(-3, Math.min(3, rowScore));
    if (rowScore) { score += rowScore; count += 1; }
  });
  if (!count) return { label: "Neutro / Sin Texto", color: "#808080" };
  const avg = score / count;
  if (avg > .5) return { label: "Muy Positivo 😄", color: "#28a745" };
  if (avg > .1) return { label: "Positivo 🙂", color: "#90be6d" };
  if (avg < -.5) return { label: "Negativo 😟", color: "#dc3545" };
  if (avg < -.1) return { label: "Algo Negativo 😐", color: "#ffc107" };
  return { label: "Neutro 😐", color: "#6c757d" };
}

function selectOptions(options, selected) {
  return options.map((option) => `<option value="${escapeHtml(option)}" ${option === selected ? "selected" : ""}>${escapeHtml(option)}</option>`).join("");
}

function sourceMarkup() {
  const radio = (value, label) => `<label class="source-option"><input type="radio" name="source" value="${value}" ${state.source === value ? "checked" : ""}>${label}</label>`;
  const detail = state.source === "official" ? `<div class="source-detail"><small>Se cargan los datos oficiales desde la planilla configurada.</small></div>` :
                 state.source === "link" ? `<div class="source-detail"><div class="inline-form"><input id="sheet-url" type="text" placeholder="Pegá un enlace público de Google Sheets o un CSV" aria-label="Enlace público"><button id="load-url">Cargar datos</button></div><small>La planilla debe ser pública. Si el proveedor bloquea el acceso desde el navegador, descargala y subila como archivo.</small></div>` :
                 state.source === "file" ? `<div class="source-detail"><input id="file-input" type="file" accept=".csv,.xlsx"><small>El archivo se procesa localmente en tu navegador: no se envía a ningún servidor.</small></div>` : "";
  return `<section class="source-panel"><p>Fuente de datos</p><div class="source-options">${radio("official", "📊 Datos oficiales cargados")}${radio("link", "🔗 Pegar enlace de Google Sheet")}${radio("file", "📂 Subir archivo (.xlsx / .csv)")}</div>${detail}</section>`;
}

function filterMarkup() {
  const materialOptions = ["Todas", ...uniqueSorted(state.data.map((row) => row.Materia))];
  const laboratoryOptions = ["Todos", ...uniqueSorted(state.data.map((row) => row.Laboratorio))];
  const careerOptions = ["Todas", ...uniqueSorted(state.data.map((row) => row.Carrera))];
  const teacher = SHOW_TEACHER_FILTER ? `<label>Docente presente<select id="filter-docente">${selectOptions(["Todos", ...uniqueSorted(state.data.flatMap((row) => row.Docentes_List))], state.filters.docente)}</select></label>` : "";
  return `<section class="filters-panel"><h2>Filtros</h2><div class="filter-grid"><label>Materia<select id="filter-materia">${selectOptions(materialOptions, state.filters.materia)}</select></label><label>Laboratorio<select id="filter-laboratorio">${selectOptions(laboratoryOptions, state.filters.laboratorio)}</select></label><label>Carrera<select id="filter-carrera">${selectOptions(careerOptions, state.filters.carrera)}</select></label>${teacher}<div class="viz-control"><span>Visualización</span><div class="control-options"><label class="radio-option"><input type="radio" name="unit" value="percentage" ${state.isPercentage ? "checked" : ""}>Porcentaje (%)</label><label class="radio-option"><input type="radio" name="unit" value="absolute" ${!state.isPercentage ? "checked" : ""}>Cantidad absoluta / escala</label></div></div></div><button id="reset-filters" class="secondary">🔄 Borrar filtros</button></section>`;
}

function metricsMarkup(rows) {
  const total = state.data.length;
  const percentage = total ? (rows.length / total) * 100 : 0;
  const sentiment = calculateSentiment(rows);
  return `<div class="metrics-row"><article class="metric"><div class="metric-label">Encuestas</div><div class="metric-value">${rows.length} de ${total}</div><div class="metric-delta">${percentage.toFixed(1)}% de la muestra</div></article><article class="metric"><div class="sentiment-badge" style="background:${sentiment.color}">${sentiment.label}</div><p class="sentiment-caption">Fuentes del análisis: Preguntas Mejoras + Palabras Clave.</p></article></div>`;
}

function overviewMarkup(rows) {
  if (!state.ratingColumns.length) return `<section class="section"><div class="section-heading"><h2>📈 Resultado general y comparativa</h2></div><div class="message info">No se detectaron columnas de calificación numérica entre 1 y 10.</div></section>`;
  const cards = state.ratingColumns.map((column) => {
    const value = average(rows.map((row) => numberOrNull(row[column])));
    const display = value === null ? "Sin datos" : state.isPercentage ? `${Math.round(value / 5 * 100)}%` : value.toFixed(2);
    return `<article class="kpi-card"><div class="metric-label">${escapeHtml(prettify(column))}</div><div class="metric-value">${display}</div></article>`;
  }).join("");
  return `<section class="section"><div class="section-heading"><h2>📈 Resultado general y comparativa</h2><p>Promedios de satisfacción total (escala 1 a 5).</p></div><div class="result-columns"><div class="result-left"><div class="kpi-grid">${cards}</div></div><div class="result-right"><div class="chart-card"><div id="plot-global" class="plot"></div></div></div></div></section>`;
}

function questionMarkup() {
  if (!state.ratingColumns.length) return "";
  return `<section class="section"><div class="section-heading"><h2>📝 Resultado detallado por pregunta</h2></div>${state.ratingColumns.map((column) => {
    const reference = SCALE_REFERENCES[column];
    const explanation = CAREER_EXPLANATIONS[column] || "Promedio de satisfacción desagregado por carrera.";
    return `<article class="question"><h3 style="font-size: 1.2rem; margin-bottom: 12px; font-weight: 700;">📌 ${escapeHtml(REVERSE_MAP[column] || column)}</h3><p style="margin: 0 0 10px; padding: 0; border: none; background: transparent; color: #667085; font-size: 0.9rem;">💡 ${escapeHtml(explanation)}</p>${reference ? `<p class="scale-reference"><strong>Extremos:</strong> <span class="low">1</span>: ${escapeHtml(reference[0])} &nbsp;⟶&nbsp; <span class="high">5</span>: ${escapeHtml(reference[1])}</p>` : ""}<div class="two-charts"><div class="chart-card"><div id="plot-dist-${escapeHtml(column)}" class="plot"></div></div><div class="chart-card"><div id="plot-career-${escapeHtml(column)}" class="plot"></div></div></div></article>`;
  }).join("")}</section>`;
}

function commentsMarkup(rows) {
  const cloudWords = cleanTextForCloud(rows.map((row) => row.Palabras_Clave).filter((value) => !isBlank(value)).join(" "));
  const displayColumns = state.textColumns.filter((column) => column !== "Palabras_Clave");
  const cloud = cloudWords.length ? `<div class="word-cloud-card"><div class="word-cloud-heading"><div><h3>Palabras clave globales</h3><p class="caption">Consigna: "Escribí tres palabras que describan tu experiencia en el laboratorio".</p></div><label class="cloud-palette-control">Colores<select id="cloud-palette">${selectOptions(Object.keys(WORD_CLOUD_PALETTES), state.cloudPalette)}</select></label></div><canvas id="word-cloud" width="1200" height="400" aria-label="Nube de palabras clave"></canvas></div>` : "";
  if (!displayColumns.length) return `<section class="section"><div class="section-heading"><h2>☁️ Comentarios y opiniones</h2></div>${cloud}<div class="message info">No se detectaron columnas de comentarios en este archivo.</div></section>`;
  const columnsStyle = `--comment-columns: ${displayColumns.length};`;
  const eligible = rows.filter((row) => displayColumns.some((column) => !isBlank(row[column]))).slice();
  if (state.commentSort === "Últimos") eligible.sort((a, b) => parseTimestamp(b.Timestamp) - parseTimestamp(a.Timestamp));
  else eligible.sort((a, b) => displayColumns.reduce((sum, column) => sum + asText(a[column]).length, 0) - displayColumns.reduce((sum, column) => sum + asText(b[column]).length, 0)).reverse();
  const visible = state.commentLimit === "Todos" ? eligible : eligible.slice(0, Number(state.commentLimit));
  const header = displayColumns.map((column) => `<div class="comment-question">${escapeHtml(prettify(column))}</div>`).join("");
  const commentRows = visible.map((row) => `<article class="comment-row"><div class="comment-meta">👤 <strong>${escapeHtml(row.Carrera)}</strong> <span> | ${escapeHtml(row.Laboratorio)}</span></div><div class="comment-content" style="${columnsStyle}">${displayColumns.map((column) => !isBlank(row[column]) && asText(row[column]).trim().length > 1 ? `<div class="comment-answer" data-question="${escapeHtml(prettify(column))}">“${escapeHtml(asText(row[column]).trim())}”</div>` : `<div class="no-answer">[No responde.]</div>`).join("")}</div></article>`).join("");
  return `<section class="section"><div class="section-heading"><h2>☁️ Comentarios y opiniones</h2></div>${cloud}<h3 style="margin-top:20px">Opiniones</h3><div class="comments-controls"><label>Ordenar por<select id="comment-sort">${selectOptions(["Últimos", "Longitud (Texto)"], state.commentSort)}</select></label><label>Mostrar<select id="comment-limit">${selectOptions(["10", "20", "Todos"], state.commentLimit)}</select></label></div>${visible.length ? `<p class="comment-scroll-note"><em>(Desplácese verticalmente por la tabla para recorrer los comentarios)</em></p><div class="comments-card"><div class="comment-head" style="${columnsStyle}">${header}</div><div class="comment-scroll">${commentRows}</div></div>` : `<div class="message info">No hay comentarios disponibles.</div>`}</section>`;
}

function dataMarkup(rows) {
  const headers = state.columns.map((column) => `<th>${escapeHtml(prettify(column))}</th>`).join("");
  const body = rows.map((row) => `<tr>${state.columns.map((column) => `<td title="${escapeHtml(Array.isArray(row[column]) ? row[column].join(", ") : asText(row[column]))}">${escapeHtml(Array.isArray(row[column]) ? row[column].join(", ") : asText(row[column]))}</td>`).join("")}</tr>`).join("");
  return `<section class="section"><details class="data-card"><summary>📂 Ver base de datos (filtros actuales)</summary><div class="table-scroll"><table><thead><tr>${headers}</tr></thead><tbody>${body}</tbody></table></div></details></section>`;
}

function footerMarkup() {
  return `<div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--line); text-align: center; color: var(--muted); font-size: 0.86rem;"><p>Desarrollado por: <strong>J. I. Peralta</strong> & <strong>🤖 LLMs</strong> | Última edición: 13/08/2026</p><p><a href="mailto:jperalta@untref.edu.ar" style="color: var(--muted); text-decoration: none;">📧 jperalta@untref.edu.ar</a> · <a href="https://www.linkedin.com/in/juaniperalta/" style="color: var(--muted); text-decoration: none;" target="_blank" rel="noreferrer">🔗 LinkedIn</a></p></div>`;
}

function dashboardMarkup() {
  const rows = filteredData();
  const collapsed = state.sidebarOpen ? "" : " sidebar-collapsed";
  const sidebarLabel = state.sidebarOpen ? "Ocultar filtros" : "Mostrar filtros";
  const sidebarIcon = state.sidebarOpen ? "◀" : "▶";
  return `<div class="dashboard-layout${collapsed}"><aside class="sidebar"><button id="toggle-sidebar" class="sidebar-toggle" type="button" aria-expanded="${state.sidebarOpen}" aria-label="${sidebarLabel}" title="${sidebarLabel}"><span aria-hidden="true">${sidebarIcon}</span><span class="sidebar-toggle-label">${state.sidebarOpen ? "Ocultar" : "Filtros"}</span></button>${state.sidebarOpen ? filterMarkup() : ""}</aside><div class="dashboard-content">${metricsMarkup(rows)}${overviewMarkup(rows)}${questionMarkup()}${commentsMarkup(rows)}${dataMarkup(rows)}${footerMarkup()}</div></div>`;
}

function render() {
  applyTheme();
  let content = `<div class="title-row"><h1>📊 Resultados de la Encuesta del Laboratorio de Física</h1><label class="theme-control">Apariencia<select id="theme-mode" aria-label="Apariencia del tablero">${selectOptions(["system", "light", "dark"], state.themeMode).replace(">system<", ">Sistema<").replace(">light<", ">Claro<").replace(">dark<", ">Oscuro<")}</select></label></div>${sourceMarkup()}`;
  if (state.error) content += `<div class="message error">${escapeHtml(state.error)}</div>`;
  if (state.loading) content += `<div class="loading-card">Cargando y procesando datos…</div>`;
  else if (state.data) content += dashboardMarkup();
  else if (state.source === "file") content += `<div class="empty-card">Subí un archivo del formulario para comenzar.</div>`;
  else if (state.source === "link") content += `<div class="empty-card">Pegá un enlace público arriba para comenzar.</div>`;
  app.innerHTML = content;
  bindEvents();
  if (!state.loading && state.data) requestAnimationFrame(drawVisualizations);
}

function bindEvents() {
  document.querySelectorAll('input[name="source"]').forEach((radio) => radio.addEventListener("change", () => {
    state.source = radio.value; state.error = "";
    if (state.source === "official") loadFromUrl(OFFICIAL_SHEET_URL);
    else { state.data = null; state.loading = false; render(); }
  }));
  document.querySelector("#load-url")?.addEventListener("click", () => loadFromUrl(document.querySelector("#sheet-url").value));
  document.querySelector("#file-input")?.addEventListener("change", (event) => loadFromFile(event.target.files[0]));
  ["materia", "laboratorio", "carrera", "docente"].forEach((filter) => document.querySelector(`#filter-${filter}`)?.addEventListener("change", (event) => { state.filters[filter] = event.target.value; render(); }));
  document.querySelectorAll('input[name="unit"]').forEach((radio) => radio.addEventListener("change", (event) => { state.isPercentage = event.target.value === "percentage"; render(); }));
  document.querySelector("#reset-filters")?.addEventListener("click", () => { state.filters = { materia: "Todas", laboratorio: "Todos", carrera: "Todas", docente: "Todos" }; render(); });
  document.querySelector("#comment-sort")?.addEventListener("change", (event) => { state.commentSort = event.target.value; render(); });
  document.querySelector("#comment-limit")?.addEventListener("change", (event) => { state.commentLimit = event.target.value; render(); });
  document.querySelector("#cloud-palette")?.addEventListener("change", (event) => { state.cloudPalette = event.target.value; savePreference("untref-cloud-palette", state.cloudPalette); drawWordCloud(filteredData()); });
  document.querySelector("#theme-mode")?.addEventListener("change", (event) => { state.themeMode = event.target.value; savePreference("untref-theme-mode", state.themeMode); render(); });
  document.querySelector("#toggle-sidebar")?.addEventListener("click", () => { state.sidebarOpen = !state.sidebarOpen; savePreference("untref-sidebar-open", String(state.sidebarOpen)); render(); });
}

function plotBar(id, { x, y, text, orientation = "h", title, xTitle, xRange, valueColors }) {
  const element = document.querySelector(`#${CSS.escape(id)}`);
  if (!element || !window.Plotly) return;
  const dark = isDarkTheme();
  const background = dark ? "#17212b" : "#ffffff";
  const ink = dark ? "#e6edf3" : "#1f2933";
  const grid = dark ? "#334657" : "#e6edf2";
  Plotly.newPlot(element, [{ type: "bar", orientation, x, y, text, texttemplate: "%{text}", textposition: "outside", cliponaxis: false, marker: { color: valueColors, colorscale: [[0, "#bce8e5"], [1, "#006d77"]], showscale: false }, hovertemplate: "%{y}<br><b>%{x}</b><extra></extra>" }], {
    title: { text: title, font: { size: 15 } }, margin: { t: 48, r: 65, b: 55, l: orientation === "h" ? 185 : 55 }, height: 350,
    paper_bgcolor: background, plot_bgcolor: background, font: { color: ink }, xaxis: { title: xTitle, range: xRange, fixedrange: false, gridcolor: grid, zerolinecolor: grid }, yaxis: { title: orientation === "v" ? xTitle : "", fixedrange: false, gridcolor: grid, zerolinecolor: grid }, showlegend: false
  }, { responsive: true, displaylogo: false });
}

function drawVisualizations() {
  const rows = filteredData();
  if (state.ratingColumns.length) {
    const global = globalCareerMeans(rows);
    const globalValues = global.map((item) => state.isPercentage ? item.mean / 5 * 100 : item.mean);
    plotBar("plot-global", { x: globalValues, y: global.map((item) => `${item.career} (N=${item.count})`), text: globalValues.map((value) => state.isPercentage ? `${value.toFixed(1)}%` : value.toFixed(2)), title: "Satisfacción global por carrera", xTitle: state.isPercentage ? "% satisfacción global" : "Promedio general (1-5)", xRange: state.isPercentage ? [0, 110] : [1, 5.8], valueColors: globalValues });
    state.ratingColumns.forEach((column) => {
      const distribution = [1, 2, 3, 4, 5].map((score) => rows.filter((row) => numberOrNull(row[column]) === score).length);
      const total = distribution.reduce((sum, value) => sum + value, 0);
      const values = state.isPercentage ? distribution.map((value) => total ? value / total * 100 : 0) : distribution;
      plotBar(`plot-dist-${column}`, { x: ["1", "2", "3", "4", "5"], y: values, text: values.map((value) => state.isPercentage ? `${value.toFixed(1)}%` : String(value)), orientation: "v", title: `Distribución (${state.isPercentage ? "%" : "votos"})`, xTitle: "Puntaje", xRange: undefined, valueColors: values });
      const careers = groupedMean(rows, column);
      const careerValues = careers.map((item) => state.isPercentage ? item.mean / 5 * 100 : item.mean);
      plotBar(`plot-career-${column}`, { x: careerValues, y: careers.map((item) => `${item.career} (N=${item.count})`), text: careerValues.map((value) => state.isPercentage ? `${value.toFixed(1)}%` : value.toFixed(2)), title: `Promedio por carrera (${state.isPercentage ? "% satisfacción" : "escala 1-5"})`, xTitle: state.isPercentage ? "% satisfacción" : "Promedio", xRange: state.isPercentage ? [0, 115] : [1, 5.8], valueColors: careerValues });
    });
  }
  drawWordCloud(rows);
}

function drawWordCloud(rows) {
  const canvas = document.querySelector("#word-cloud");
  if (!canvas || !window.WordCloud) return;
  const containerWidth = canvas.parentElement?.clientWidth || 1200;
  const width = Math.max(320, Math.min(1200, Math.floor(containerWidth - 32)));
  canvas.width = width;
  canvas.height = Math.round(width / 3);
  const counts = new Map();
  cleanTextForCloud(rows.map((row) => row.Palabras_Clave).filter((value) => !isBlank(value)).join(" ")).forEach((word) => counts.set(word, (counts.get(word) || 0) + 1));
  const words = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 300);
  if (!words.length) return;
  const palette = WORD_CLOUD_PALETTES[state.cloudPalette] || WORD_CLOUD_PALETTES.Viridis;
  const max = words[0][1];
  const min = words[words.length - 1][1];
  WordCloud(canvas, {
    list: words,
    gridSize: Math.max(4, Math.round(canvas.width / 180)),
    weightFactor: (count) => 20 + ((count - min) / Math.max(1, max - min)) * 76,
    fontFamily: "Inter, Arial, sans-serif",
    fontWeight: 600,
    color: (_word, _weight, fontSize) => palette[Math.min(palette.length - 1, Math.max(0, Math.round((fontSize - 20) / 76 * (palette.length - 1))))],
    backgroundColor: "transparent",
    rotateRatio: 0,
    shuffle: true,
    shape: "square",
    drawOutOfBound: false
  });
}

render();
loadFromUrl(OFFICIAL_SHEET_URL);
window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener("change", () => { if (state.themeMode === "system") render(); });
window.addEventListener("resize", () => { if (state.data) drawWordCloud(filteredData()); });

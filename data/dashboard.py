import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import warnings

# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="Dashboard de Predicción (Rápido)")
warnings.filterwarnings('ignore')

# --- Constantes ---
RUTA_HISTORICO = 'data/datos_finales_listos_para_modelo.csv'
# Asumimos que así se llaman tus 3 archivos de predicciones
RUTAS_PREDICCIONES = {
    'ARIMA': 'data/predicciones_precalculadas (ARIMA)_sin_decimales.csv',
    'Holt-Winters': 'data/predicciones_precalculadas (holt_winters)_sin_decimales.csv',
    'Red Neuronal (LSTM)': 'data/predicciones_lstm_sin_decimales.csv'
}
COL_PRODUCTO = 'Producto - Descripción'
COL_CLIENTE = 'Cliente - Descripción'
COL_FECHA = 'Fecha'
METRICAS = ['Pedido_piezas', 'Pedido_MXN', 'Factura_piezas', 'Factura_MXN']

# --- Funciones de Carga de Datos (Cacheada) ---
@st.cache_data
def cargar_datos_completos():
    """
    Carga el histórico y todos los archivos de predicciones.
    Une las predicciones en un solo DataFrame con una columna 'Modelo'.
    """
    try:
        # Cargar histórico
        df_hist = pd.read_csv(RUTA_HISTORICO)
        df_hist[COL_FECHA] = pd.to_datetime(df_hist[COL_FECHA], format='%d/%m/%Y')
        
        # Cargar y unir predicciones
        lista_dfs_pred = []
        for nombre_modelo, ruta in RUTAS_PREDICCIONES.items():
            df_pred = pd.read_csv(ruta)
            df_pred[COL_FECHA] = pd.to_datetime(df_pred[COL_FECHA], format='%d/%m/%Y')
            df_pred['Modelo'] = nombre_modelo # ¡Clave! Añadimos la columna del modelo
            lista_dfs_pred.append(df_pred)
            
        df_pred_total = pd.concat(lista_dfs_pred)
        
        return df_hist, df_pred_total
        
    except FileNotFoundError as e:
        st.error(f"Error fatal: No se encontró el archivo {e.filename}.")
        st.error("Asegúrate de que los 4 archivos (1 histórico, 3 de predicción) estén en la carpeta 'data/' de GitHub.")
        return None, None
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None, None

# --- Cargar TODOS los datos al inicio ---
df_hist, df_pred = cargar_datos_completos()

if df_hist is None:
    st.stop()

# --- Listas de Filtros (del histórico) ---
productos_lista_completa = df_hist[COL_PRODUCTO].unique().tolist()
clientes_lista_completa = df_hist[COL_CLIENTE].unique().tolist()
modelos_lista_completa = df_pred['Modelo'].unique().tolist()

# --- Barra Lateral (Filtros) ---
st.sidebar.header("⚙️ Configuración del Dashboard")

# --- Inicializar "Memoria" (Session State) ---
if 'productos_seleccionados' not in st.session_state:
    st.session_state.productos_seleccionados = productos_lista_completa
if 'clientes_seleccionados' not in st.session_state:
    st.session_state.clientes_seleccionados = clientes_lista_completa

# --- Funciones de Callback ---
def callback_select_all():
    st.session_state.productos_seleccionados = productos_lista_completa
    st.session_state.clientes_seleccionados = clientes_lista_completa

def callback_deselect_all():
    st.session_state.productos_seleccionados = []
    st.session_state.clientes_seleccionados = []

# --- Botones de Control ---
st.sidebar.write("Control de Filtros:")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.button("Seleccionar Todos", on_click=callback_select_all, use_container_width=True)
with col2:
    st.button("Limpiar Todo", on_click=callback_deselect_all, use_container_width=True)
st.sidebar.divider()

# --- Filtros (Conectados a la memoria) ---
productos_seleccionados = st.sidebar.multiselect(
    "Selecciona Productos:", 
    options=productos_lista_completa, 
    key='productos_seleccionados' 
)
clientes_seleccionados = st.sidebar.multiselect(
    "Selecciona Clientes:", 
    options=clientes_lista_completa, 
    key='clientes_seleccionados' 
)

# --- Filtros de Métrica y Modelo ---
metrica_seleccionada = st.sidebar.selectbox("Selecciona la Métrica:", METRICAS)
modelo_seleccionado = st.sidebar.selectbox("Selecciona el Modelo:", modelos_lista_completa)

st.title("📈 Dashboard de Predicción (Versión Pre-calculada)")
st.info("Esta versión es instantánea. Los modelos se pre-calcularon offline.")

# --- Lógica Principal (SIN MODELOS) ---
if not productos_seleccionados or not clientes_seleccionados:
    st.warning("Por favor, selecciona al menos un producto y un cliente.")
else:
    # 1. Filtrar los DataFrames (¡Instantáneo!)
    df_hist_filtrado = df_hist[
        (df_hist[COL_PRODUCTO].isin(productos_seleccionados)) &
        (df_hist[COL_CLIENTE].isin(clientes_seleccionados))
    ]
    
    df_pred_filtrado = df_pred[
        (df_pred[COL_PRODUCTO].isin(productos_seleccionados)) &
        (df_pred[COL_CLIENTE].isin(clientes_seleccionados)) &
        (df_pred['Modelo'] == modelo_seleccionado) # <-- Filtrar por el modelo elegido
    ]
    
    if df_hist_filtrado.empty or df_pred_filtrado.empty:
        st.warning("No se encontraron datos históricos o predicciones para la selección actual.")
    else:
        # 2. Preparar datos para el gráfico (LA SUMA)
        ts_hist_sum = df_hist_filtrado.groupby(COL_FECHA)[metrica_seleccionada].sum()
        
        # Seleccionar solo la métrica correcta para la predicción
        ts_pred_sum = df_pred_filtrado.groupby(COL_FECHA)[metrica_seleccionada].sum()

        # 3. Crear el Gráfico (del TOTAL)
        st.subheader(f"Gráfico de Predicción ({modelo_seleccionado}) - {metrica_seleccionada}")
        fig = go.Figure()

        # Histórico
        fig.add_trace(go.Scatter(
            x=ts_hist_sum.index, y=ts_hist_sum.values,
            mode='lines+markers', name='Datos Históricos (Total)'
        ))
        
        # Predicción
        fig.add_trace(go.Scatter(
            x=ts_pred_sum.index, y=ts_pred_sum.values,
            mode='lines', name=f'Predicción ({modelo_seleccionado})',
            line=dict(color='red', width=3, dash='dash')
        ))
        
        fig.update_layout(xaxis_title="Fecha", yaxis_title=metrica_seleccionada, legend_title="Series")
        st.plotly_chart(fig, use_container_width=True)

        # 4. Crear la Tabla (DESGLOSADA)
        st.subheader(f"Tabla de Predicciones Desglosada ({modelo_seleccionado})")
        
        # Seleccionar solo las columnas que importan
        columnas_tabla = [COL_PRODUCTO, COL_CLIENTE, COL_FECHA, metrica_seleccionada]
        df_tabla = df_pred_filtrado[columnas_tabla]
        
        st.dataframe(
            df_tabla.style.format({
                metrica_seleccionada: "{:,.0f}"
            }),
            height=400,
            use_container_width=True
        )


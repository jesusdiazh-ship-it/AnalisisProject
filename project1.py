import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, BarycentricInterpolator, CubicSpline

# Configuration de la página
st.set_page_config(page_title="Interpolación - Instrumentación", layout="wide")

# ==========================================
# DATOS DE LABORATORIO (PREDETERMINADOS)
# ==========================================
datos_defecto = pd.DataFrame({
    'Temperatura (°C)': [10.0, 25.0, 40.0, 65.0, 80.0, 100.0],
    'Resistencia (kOhm)': [4.50, 3.20, 2.10, 1.30, 0.80, 0.40]
})

# ==========================================
# MENÚ LATERAL (NAVEGACIÓN)
# ==========================================
st.sidebar.title("Navegación del Proyecto")
opcion = st.sidebar.radio("Ir a:", [
    "🏠 Inicio / Acerca de", 
    "📊 Datos de Laboratorio", 
    "📈 Métodos de Interpolación", 
    "📘 Manual de Usuario"
])

# ==========================================
# SECCIÓN 1: INICIO / ACERCA DE
# ==========================================
if opcion == "🏠 Inicio / Acerca de":
    st.title("Proyecto Final: Análisis de Técnicas Numéricas")
    st.subheader("Aplicación Real de Interpolación en Instrumentación de Laboratorio")
    
    st.markdown("""
    ---
    ### 📝 Descripción del Proyecto
    En el ámbito de la ingeniería y la instrumentación, los sensores físicos no entregan lecturas directas de las variables, sino magnitudes eléctricas (voltaje, corriente, resistencia). Para traducir estas lecturas a variables de ingeniería (como la temperatura), se requiere un proceso de **Calibración**.
    
    Esta aplicación implementa y compara de manera crítica distintos métodos numéricos de **Interpolación** para modelar el comportamiento de un **Termistor NTC** a partir de datos discretos obtenidos en laboratorio, garantizando la aproximación de valores intermedios con control de error.
    """)
    
    st.info("""
    ### 👥 Integrantes y Detalles ("Acerca de")
    * **Asignatura:** Análisis de Técnicas Numéricas
    * **Docente:** Ing. Carlos Cohen M.
    * **Institución:** Corporación Universitaria del Caribe (CECAR)
    * **Facultad:** Ciencias Básicas, Ingeniería y Arquitectura
    * **Autores:** [Samuel Sanches, Jesús Díaz]
    * **Semestre:** 2026-I
    """)

# ==========================================
# SECCIÓN 2: DATOS DE LABORATORIO
# ==========================================
elif opcion == "📊 Datos de Laboratorio":
    st.title("🧪 Gestión de Datos del Experimento")
    st.write("A continuación se presentan los datos capturados mediante los instrumentos de medida en el laboratorio:")
    
    # Permitir al usuario modificar o ver los datos
    modo_datos = st.selectbox("Seleccione el set de datos:", ["Datos de calibración predeterminados (Termistor NTC)", "Ingresar datos manualmente"])
    
    if modo_datos == "Datos de calibración predeterminados (Termistor NTC)":
        df = datos_defecto
    else:
        st.warning("Ingrese los datos separados por comas (,)")
        col_x = st.text_input("Valores de Variable Independiente X (Ej: Resistencia en kOhm):", "4.50, 3.20, 2.10, 1.30, 0.80, 0.40")
        col_y = st.text_input("Valores de Variable Dependiente Y (Ej: Temperatura en °C):", "10.0, 25.0, 40.0, 65.0, 80.0, 100.0")
        
        try:
            x_arr = [float(i) for i in col_x.split(",")]
            y_arr = [float(i) for i in col_y.split(",")]
            df = pd.DataFrame({'Resistencia (kOhm)': x_arr, 'Temperatura (°C)': y_arr})
        except:
            st.error("Error en el formato de los datos. Asegúrate de usar números separados por comas.")
            df = datos_defecto

    st.dataframe(df.style.highlight_max(axis=0))
    
    # Gráfica de dispersión de los puntos experimentales
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], color='red', marker='o', s=80, label='Puntos de Calibración')
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.set_title("Puntos Discretos obtenidos por Instrumentación")
    ax.grid(True, linestyle='--')
    ax.legend()
    st.pyplot(fig)

# ==========================================
# SECCIÓN 3: MÉTODOS DE INTERPOLACIÓN (MÓDULO CENTRAL)
# ==========================================
elif opcion == "📈 Métodos de Interpolación":
    st.title("📐 Análisis Comparativo de Técnicas Numéricas")
    
    # Recuperamos los datos elegidos
    X_med = np.array(datos_defecto.iloc[:, 1]) # Usamos Resistencia como entrada para estimar Temp
    Y_med = np.array(datos_defecto.iloc[:, 0]) 
    
    # Ordenar los datos por si las moscas para que los algoritmos funcionen bien
    idx = np.argsort(X_med)
    X_med, Y_med = X_med[idx], Y_med[idx]
    
    st.sidebar.markdown("### 🎛️ Controles de Interpolación")
    valor_test = st.sidebar.slider("Valor de Resistencia a estimar ($k\Omega$):", float(X_med.min()), float(X_med.max()), float(np.median(X_med)))
    
    # --- IMPLEMENTACIÓN DE LOS MÉTODOS NUMÉRICOS ---
    # 1. Lineal
    f_lineal = interp1d(X_med, Y_med, kind='linear')
    # 2. Lagrange / Newton (Generan el mismo polinomio único, usamos interpolador baricéntrico para estabilidad)
    f_polinomial = BarycentricInterpolator(X_med, Y_med)
    # 3. Splines Cúbicos
    f_spline = CubicSpline(X_med, Y_med)
    
    # Predicciones puntuales
    pred_lin = f_lineal(valor_test)
    pred_pol = f_polinomial(valor_test)
    pred_spl = f_spline(valor_test)
    
    # --- CÁLCULO DE ERRORES (Leave-One-Out Cross-Validation para Interpolación) ---
    errores = []
    for i in range(len(X_med)):
        X_sub = np.delete(X_med, i)
        Y_sub = np.delete(Y_med, i)
        
        # Evaluar qué tan bien predice el punto eliminado cada método
        # Lineal
        try:
            f_l_sub = interp1d(X_sub, Y_sub, kind='linear', fill_value="extrapolate")
            err_l = abs(f_l_sub(X_med[i]) - Y_med[i])
        except: err_l = np.nan
        
        # Polinomial
        f_p_sub = BarycentricInterpolator(X_sub, Y_sub)
        err_p = abs(f_p_sub(X_med[i]) - Y_med[i])
        
        # Spline
        try:
            f_s_sub = CubicSpline(X_sub, Y_sub, extrapolate=True)
            err_s = abs(f_s_sub(X_med[i]) - Y_med[i])
        except: err_s = np.nan
        
        errores.append([err_l, err_p, err_s])
        
    df_err = pd.DataFrame(errores, columns=['Error Lineal', 'Error Polinomial (Lagrange/Newton)', 'Error Trazador Cúbico'])
    err_medios = df_err.mean()

 # --- MOSTRAR ESTIMACIÓN PUNTUAL EN TARJETAS ESTÉTICAS (CORREGIDO) ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Interpolación Lineal", f"{pred_lin:.3f} °C", f"Error Promedio: {err_medios['Error Lineal']:.3f}")
    col2.metric("Polinomio Lagrange/Newton", f"{pred_pol:.3f} °C", f"Error Promedio: {err_medios['Error Lineal']:.3f}", delta_color="inverse")
    col3.metric("Trazador Cúbico (Spline)", f"{pred_spl:.3f} °C", f"Error Promedio: {err_medios['Error Trazador Cúbico']:.3f}", delta_color="off")

    # --- PANELES VISUALES (GRÁFICAS INDIVIDUALES Y CONJUNTAS) ---
    st.markdown("---")
    visualizacion = st.radio("Seleccione el tipo de visualización gráfica:", ["Gráfica Conjunta (Superpuesta para Comparación)", "Gráficas Individuales por Método"])
    
    # Generación de curva continua densa para graficar los modelos
    X_denso = np.linspace(X_med.min(), X_med.max(), 300)
    
    if visualizacion == "Gráfica Conjunta (Superpuesta para Comparación)":
        fig_conj, ax_conj = plt.subplots(figsize=(10, 5))
        ax_conj.scatter(X_med, Y_med, color='black', zorder=5, s=60, label='Datos de Laboratorio')
        
        ax_conj.plot(X_denso, f_lineal(X_denso), label='Lineal', linestyle='--', color='blue', alpha=0.8)
        ax_conj.plot(X_denso, f_polinomial(X_denso), label='Polinomial de grado 5 (Lagrange)', linestyle='-.', color='orange', alpha=0.8)
        ax_conj.plot(X_denso, f_spline(X_denso), label='Trazador Cúbico (Spline)', linestyle='-', color='green', alpha=0.9)
        
        # Indicador del punto testeado
        ax_conj.scatter(valor_test, pred_spl, color='red', marker='X', s=150, zorder=6, label=f'Estimación en {valor_test} k$\Omega$')
        
        ax_conj.set_title("Visualización Multimétodo Superpuesta (Análisis Crítico)")
        ax_conj.set_xlabel("Resistencia ($k\Omega$)")
        ax_conj.set_ylabel("Temperatura (°C)")
        ax_conj.legend()
        ax_conj.grid(True, alpha=0.3)
        st.pyplot(fig_conj)
        
        st.success("""
        **Análisis de Resultados:** El método de **Trazadores Cúbicos** demuestra ser el más eficiente para esta aplicación física. Evita las oscilaciones artificiales del polinomio global de Lagrange (Grado 5) en los extremos y suaviza los quiebres abruptos de la interpolación lineal, emulando con alta precisión la física real de descarga térmica del termistor.
        """)

    else:
        # Pestañas para las gráficas individuales
        p_lin, p_lag, p_spl = st.tabs(["Líneal", "Lagrange / Newton", "Spline Cúbico"])
        
        with p_lin:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.scatter(X_med, Y_med, color='black', label='Datos')
            ax.plot(X_denso, f_lineal(X_denso), color='blue')
            ax.set_title("Interpolación Lineal Tramo a Tramo")
            ax.grid(True)
            st.pyplot(fig)
            
        with p_lag:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.scatter(X_med, Y_med, color='black', label='Datos')
            ax.plot(X_denso, f_polinomial(X_denso), color='orange')
            ax.set_title("Polinomio de Grado Máximo (Lagrange/Newton)")
            ax.grid(True)
            st.pyplot(fig)
            
        with p_spl:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.scatter(X_med, Y_med, color='black', label='Datos')
            ax.plot(X_denso, f_spline(X_denso), color='green')
            ax.set_title("Splines Cúbicos (Trazadores)")
            ax.grid(True)
            st.pyplot(fig)

# ==========================================
# SECCIÓN 4: MANUAL DE USUARIO
# ==========================================
elif opcion == "📘 Manual de Usuario":
    st.title("📘 Manual de Usuario y Guía de Despliegue")
    
    st.markdown("""
    ### 🔌 Requisitos de Instalación
    Para ejecutar este software de manera local, es necesario disponer de un entorno de Python e instalar las siguientes dependencias mediante la terminal:
    
    ```bash
    pip install streamlit numpy pandas matplotlib scipy
    ```
    
    ### 🚀 Instrucciones de Uso
    1. **Ejecutar la aplicación:** Abra la consola en la carpeta raíz del archivo `app.py` y ejecute:
       ```bash
       streamlit run app.py
       ```
    2. **Exploración de datos:** Diríjase a la sección *'Datos de Laboratorio'* para observar la tabla base obtenida por los instrumentos de medición.
    3. **Interpolación interactiva:** En el módulo *'Métodos de Interpolación'*, utilice la barra deslizable lateral para simular una lectura en kiloohms del sensor. El sistema calculará en tiempo real las temperaturas según cada método y actualizará los gráficos individuales y conjuntos de forma inmediata.
    """)
import streamlit as st
import numpy as np

# Configuración de la página y estilos
st.set_page_config(
    page_title="¡Hola neurona!",
    page_icon="🧠",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-top: 10px;
        padding-bottom: 10px;
        padding-left: 20px;
        padding-right: 20px;
        font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db; /* Cambiado a un azul más claro */
        color: white;
        border-radius: 5px 5px 0px 0px;
        font-weight: bold;
    }
    .main {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .neuron-container {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin: 10px 0;
    }
    .title {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
    }
    .button {
        background-color: #1f618d;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        width: 100%;
    }
    .output-container {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        text-align: center;
        border: 2px solid #3498db;
    }
    .output-text {
        color: #2c3e50;
        font-size: 24px;
        font-weight: bold;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Función para calcular la suma ponderada de la neurona
def calculate_weighted_sum(inputs, weights, bias=0):
    # Convertir a numpy arrays para asegurar el cálculo correcto
    inputs = np.array(inputs)
    weights = np.array(weights)
    
    # Calcular la suma ponderada
    weighted_sum = np.dot(inputs, weights) + bias
    
    return weighted_sum

# Título principal con estilo personalizado
st.markdown('<h1 class="title">¡Hola neuronillas! 🧠</h1>', unsafe_allow_html=True)

# Crear pestañas
tab1, tab2, tab3 = st.tabs(["Una entrada", "Dos entradas", "Tres entradas y sesgo"])

# Pestaña 1: Una entrada
with tab1:
    st.markdown("### Una neurona con una entrada y un peso")
    
    # Contenedor para el peso
    st.markdown("#### Peso")
    weight = st.slider("", 
                      min_value=0.0, 
                      max_value=5.0, 
                      value=1.0, 
                      key="weight1",
                      help="Ajusta el peso de la neurona")
    
    # Contenedor para la entrada
    st.markdown("#### Entrada")
    input_value = st.number_input("", 
                                value=0.0,
                                key="input1",
                                help="Introduce el valor de entrada")
    
    if st.button("Calcular la salida", key="calc1"):
        weighted_sum = calculate_weighted_sum([input_value], [weight])
        st.write(f"**Salida de la Neurona:** {weighted_sum}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Pestaña 2: Dos entradas
with tab2:
    st.markdown("### Una neurona con dos entradas y dos pesos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Peso w₀")
        weight0 = st.slider("", min_value=0.0, max_value=5.0, value=1.0, key="weight2_0")
        st.markdown("#### Entrada x₀")
        input0 = st.number_input("", value=0.0, key="input2_0")
    
    with col2:
        st.markdown("#### Peso w₁")
        weight1 = st.slider("", min_value=0.0, max_value=5.0, value=1.0, key="weight2_1")
        st.markdown("#### Entrada x₁")
        input1 = st.number_input("", value=0.0, key="input2_1")
    
    if st.button("Calcular la salida", key="calc2"):
        weighted_sum = calculate_weighted_sum([input0, input1], [weight0, weight1])
        st.write(f"**Resultado de la Neurona:** {weighted_sum}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Pestaña 3: Tres entradas y sesgo
with tab3:
    st.markdown("### Una neurona con tres entradas y sesgo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Peso w₀")
        weight0 = st.slider("", min_value=0.0, max_value=5.0, value=1.0, key="weight3_0")
        st.markdown("#### Entrada x₀")
        input0 = st.number_input("", value=0.0, key="input3_0")
    
    with col2:
        st.markdown("#### Peso w₁")
        weight1 = st.slider("", min_value=0.0, max_value=5.0, value=1.0, key="weight3_1")
        st.markdown("#### Entrada x₁")
        input1 = st.number_input("", value=0.0, key="input3_1")
    
    with col3:
        st.markdown("#### Peso w₂")
        weight2 = st.slider("", min_value=0.0, max_value=5.0, value=1.0, key="weight3_2")
        st.markdown("#### Entrada x₂")
        input2 = st.number_input("", value=0.0, key="input3_2")
    
    st.markdown("#### Sesgo")
    bias = st.number_input("", value=0.0, key="bias3", help="Introduce el valor del sesgo")
    
    if st.button("Calcular la salida", key="calc3"):
        weighted_sum = calculate_weighted_sum([input0, input1, input2], [weight0, weight1, weight2], bias)
        st.write(f"**Suma Ponderada:** {weighted_sum}")
    
    st.markdown('</div>', unsafe_allow_html=True)

st.write("Realizado por Marcos García Estévez")

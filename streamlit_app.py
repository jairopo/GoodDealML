from model import GoodDealModel
from procesa_imagen import getDataFromImage
import streamlit as st

# Título, descripción e imagen de la aplicación
st.title("GoodDealML")
st.write("GoodDealML es un clasficador de oferta que se encarga de predecir si una oferta es Mala, Regular o Buena. Para ello, debe introducir los datos de su producto o bien subir una foto de la que se sacarán los datos necesarios de forma automática.")
st.image("img/logo.jpeg", use_container_width=True)
# Creación de las pestañas para el escáner y para los datos manuales
tab1, tab2 = st.tabs(["FotoScan", "Manual"])

@st.cache_resource
def initial_load():
    """Método para cargar el módelo y la key de la api únicamente al principio"""
    return GoodDealModel(), st.secrets['GEMINI_API_KEY']

# Carga el modelo
goodDealModel, api_key = initial_load()
# Pestaña 1 con el escáner de las fotos
with tab1:
    # Carga la imagen
    uploaded_file = st.file_uploader("Sube una imagen del producto", type=["png", "jpg", "jpeg"])
    # Comprueba que esté la imagen
    if uploaded_file is not None:
        # Muestra la imagen
        st.image(uploaded_file, caption="Imagen procesada", use_container_width=True)
        try:
            # Recoge los datos de la imagen procesada
            datos_procesados = getDataFromImage(uploaded_file, api_key)
            # Muestra los datos recogidos
            titulo = st.text_input("Título del producto", datos_procesados["titulo"])
            empresa = st.text_input("Nombre de la empresa", datos_procesados["empresa"])
            precio_anterior = st.number_input("Precio anterior", value=datos_procesados["precio_anterior"], min_value=0.01)
            precio_actual = st.number_input("Precio actual", value=datos_procesados["precio_actual"], min_value=0.01)
            # Calcula la oferta y los otros datos al pulsar el botón
            if st.button("Calcular oferta", key="boton_tab1"):
                # Comprueba que los datos estén rellenos
                if not titulo:
                    st.error("El título del producto es obligatorio.")
                elif not empresa:
                    st.error("El nombre de la empresa es obligatorio.")
                elif precio_anterior <= 0:
                    st.error("El precio anterior debe ser mayor a 0.")
                elif precio_actual <= 0:
                    st.error("El precio actual debe ser mayor a 0.")
                else:
                    with st.spinner("Calculando..."):
                        # Recoge los datos de la oferta
                        oferta, diferencia_precios, porcentaje_descuento, ratio_precios = goodDealModel.predict(titulo, empresa, precio_anterior, precio_actual)
                        # Muestra la lista de datos
                        st.markdown("""
                        ### Resultados del cálculo:
                        - **Diferencia de precios:** {:.2f}
                        - **Porcentaje de descuento:** {:.2f}%
                        - **Ratio de precios:** {:.2f}
                        """.format(diferencia_precios, porcentaje_descuento, ratio_precios), unsafe_allow_html=True)
                        st.markdown(f"<h3>Oferta: {oferta}</h3>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error procesando la imagen: {e}")
# Pestaña 2 con los datos manuales
with tab2: 
    # Recoge los datos del producto
    titulo = st.text_input("Título del producto").strip()
    empresa = st.text_input("Nombre de la empresa").strip()
    precio_anterior = st.number_input("Precio anterior", min_value=0.01)
    precio_actual = st.number_input("Precio actual", min_value=0.01)
    # Comprueba los campos y calcula la oferta al pulsar el botón
    if st.button("Calcular oferta", key="boton_tab2"):
        # Comprueba que los datos estén rellenos
        if not titulo:
            st.error("El título del producto es obligatorio.")
        elif not empresa:
            st.error("El nombre de la empresa es obligatorio.")
        elif precio_anterior <= 0:
            st.error("El precio anterior debe ser mayor a 0.")
        elif precio_actual <= 0:
            st.error("El precio actual debe ser mayor a 0.")
        else:
            with st.spinner("Calculando..."):
                # Recoge los datos de la oferta
                oferta, diferencia_precios, porcentaje_descuento, ratio_precios = goodDealModel.predict(titulo, empresa, precio_anterior, precio_actual)
                # Muestra la lista de datos
                st.markdown("""
                ### Resultados del cálculo:
                - **Diferencia de precios:** {:.2f}
                - **Porcentaje de descuento:** {:.2f}%
                - **Ratio de precios:** {:.2f}
                """.format(diferencia_precios, porcentaje_descuento, ratio_precios), unsafe_allow_html=True)
                st.markdown(f"<h3>Oferta: {oferta}</h3>", unsafe_allow_html=True)

    
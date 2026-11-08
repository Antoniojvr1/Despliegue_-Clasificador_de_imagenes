
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Configuración de la página
st.set_page_config(page_title="Clasificador de Gatos y Perros", page_icon="🐾")
st.title("🐾 Clasificador de Gatos y Perros")
st.write("Sube una imagen y el modelo preentrenado MobileNetV2 te dirá si es un gato o un perro.")

# Cargar el modelo de forma eficiente (usando caché para no recargar en cada interacción)
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model('modelo_gatos_perros.h5')

try:
    modelo = cargar_modelo()
    st.success("¡Modelo cargado correctamente!")
except Exception as e:
    st.error(f"No se pudo cargar el modelo. Asegúrate de haber ejecutado la celda para guardar 'modelo_gatos_perros.h5'. Error: {e}")

# Subir archivo de imagen
archivo_subido = st.file_uploader("Elige una imagen de un gato o perro...", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    # Mostrar la imagen seleccionada
    imagen = Image.open(archivo_subido)
    st.image(imagen, caption='Imagen subida', use_column_width=True)
    st.write("Clasificando...")

    # Preprocesar la imagen para que coincida con el entrenamiento (224x224, normalizada)
    img_rgb = imagen.convert("RGB")
    img_array = np.array(img_rgb).astype(float) / 255.0
    img_resized = cv2.resize(img_array, (224, 224))
    img_batch = img_resized.reshape(-1, 224, 224, 3)

    # Realizar predicción
    prediccion = modelo.predict(img_batch)
    clase_detectada = np.argmax(prediccion[0], axis=-1)
    probabilidad = np.max(prediccion[0]) * 100

    # Mostrar resultados (ajusta los nombres de las clases según el orden de tu generator)
    # Por defecto, flow_from_directory ordena las clases alfabéticamente: ['Gatos', 'Perros']
    clases = ["Gato 🐱", "Perro 🐶"]
    resultado = clases[clase_detectada]

    st.metric(label="Predicción de la Inteligencia Artificial:", value=resultado)
    st.write(f"Confianza de la predicción: {probabilidad:.2f}%")

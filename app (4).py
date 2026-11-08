
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# =========================================================
NOMBRE_ESTUDIANTE = "Antonio Josue Vasquez Rojas"
NUMERO_CUENTA = "20221900346"
# =========================================================

# Configuración de la página
st.set_page_config(page_title="Clasificador de Gatos y Perros", page_icon="🐾", layout="centered")

# Barra lateral con información del estudiante
with st.sidebar:
    st.header("🎓 Datos del Estudiante")
    st.markdown(f"**Nombre:** {Antonio Vasquez}")
    st.markdown(f"**Cuenta:** {20221900346}")
    st.write("---")
    st.info("Esta aplicación utiliza el modelo MobileNetV2 preentrenado y ajustado para clasificar imágenes de Gatos y Perros.")

# Contenido principal
st.title("🐾 Clasificador de Gatos y Perros")
st.write("Sube una imagen o usa tu cámara web para que la Inteligencia Artificial determine si es un gato o un perro.")

# Cargar el modelo de forma eficiente
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model('modelo_gatos_perros.h5')

try:
    modelo = cargar_modelo()
    st.success("¡Modelo cargado correctamente!")
except Exception as e:
    st.error(f"No se pudo cargar el modelo. Asegúrate de tener el archivo 'modelo_gatos_perros.h5' en el mismo directorio. Error: {e}")

# Opciones de entrada: Subir archivo o usar cámara
opcion_entrada = st.radio("Selecciona el método de entrada:", ("Subir una foto desde el dispositivo", "Tomar foto con la cámara (Webcam)"))

archivo_subido = None

if opcion_entrada == "Subir una foto desde el dispositivo":
    archivo_subido = st.file_uploader("Elige una imagen de un gato o perro...", type=["jpg", "jpeg", "png"])
else:
    archivo_subido = st.camera_input("Captura una foto con tu cámara")

if archivo_subido is not None:
    # Mostrar la imagen seleccionada
    imagen = Image.open(archivo_subido)
    st.image(imagen, caption='Imagen a clasificar', use_column_width=True)
    st.write("Clasificando...")

    # Preprocesar la imagen para que coincida con el entrenamiento (224x224, normalizada)
    img_rgb = imagen.convert("RGB")
    img_array = np.array(img_rgb).astype(float) / 255.0
    img_resized = cv2.resize(img_array, (224, 224))
    img_batch = img_resized.reshape(-1, 224, 224, 3)

    # Realizar predicción
    prediccion = modelo.predict(img_batch, verbose=0)
    clase_detectada = np.argmax(prediccion[0], axis=-1)
    probabilidad = np.max(prediccion[0]) * 100

    # Clases
    clases = ["Gato 🐱", "Perro 🐶"]
    resultado = clases[clase_detectada]

    # Mostrar resultados en pantalla de forma atractiva
    st.markdown("### Resultado del Análisis")
    st.metric(label="Predicción:", value=resultado)
    st.write(f"**Confianza del modelo:** {probabilidad:.2f}%")

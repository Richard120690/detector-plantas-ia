import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
import os
import streamlit as st

# 1. Descargar modelo del Release de GitHub
MODEL_URL = 'https://github.com/Richard120690/detector-plantas-ia/releases/download/v1.0/modelo_tomates.pth'
local_filename = 'modelo_tomates.pth'

if not os.path.exists(local_filename):
    response = requests.get(MODEL_URL)
    with open(local_filename, 'wb') as f:
        f.write(response.content)

# 2. Cargar modelo
device = torch.device("cpu")
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(local_filename, map_location=device))
model.eval()

# Configuración de página ancha para aprovechar el espacio
st.set_page_config(layout="wide")

# Estilo personalizado para el diseño oscuro
st.markdown("""
    <style>
    .stApp { background-color: #1a1a1a; color: white; }
    .card { background-color: #2d2d2d; padding: 20px; border-radius: 10px; margin-bottom: 10px; }
    h1 { color: #2e8b57; }
    </style>
""", unsafe_allow_html=True)

# Encabezado
st.markdown("<h1 style='text-align: center;'>Plant Doctor IA</h1>", unsafe_allow_html=True)

# Crear dos columnas principales
col_info, col_diag = st.columns([1, 1])

with col_info:
    st.subheader("Beneficios de las hojas")
    # Creamos las "tarjetas" informativas
    beneficios = [("🌿 INSECTICIDA NATURAL", "Contienen solanina, repelente natural contra plagas."),
                  ("☕ INFUSIÓN MEDICINAL", "Antiinflamatorio y alivio gastrointestinal."),
                  ("🖊️ CICATRIZACIÓN", "Propiedades para la recuperación de heridas."),
                  ("🌱 FERTILIZANTE", "Aportan nutrientes esenciales.")]
    for titulo, desc in beneficios:
        st.markdown(f"<div class='card'><b>{titulo}</b><br>{desc}</div>", unsafe_allow_html=True)

with col_diag:
    st.subheader("Diagnóstico al Instante")
    # Tu lógica de carga de archivo aquí...
    uploaded_file = st.file_uploader("Sube una foto de la hoja", type=['jpg', 'png'])
    if uploaded_file:
        # Aquí llamarías a tu función de predicción
        st.success("Analizando...")

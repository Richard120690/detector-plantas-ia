import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
import os

# Configuración de página
st.set_page_config(layout="wide")

# 1. Cargar Modelo (con caché para que no se descargue cada vez)
@st.cache_resource
def load_model():
    MODEL_URL = 'https://github.com/Richard120690/detector-plantas-ia/releases/download/v1.0/modelo_tomates.pth'
    local_filename = 'modelo_tomates.pth'
    if not os.path.exists(local_filename):
        response = requests.get(MODEL_URL)
        with open(local_filename, 'wb') as f:
            f.write(response.content)
    
    device = torch.device("cpu")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(local_filename, map_location=device))
    model.eval()
    return model

model = load_model()

# 2. Estilo CSS
st.markdown("""
    <style>
    .stApp { background-color: #1a1a1a; color: white; }
    .card { 
            background-color: #2d2d2d; 
            padding: 10px;        /* Reducimos el relleno interno (antes era 20px) */
            border-radius: 8px;   /* Bordes ligeramente menos redondeados */
            margin-bottom: 8px;   /* Menos espacio entre tarjetas */
            border: 1px solid #444; 
            font-size: 14px;
    }
    .card b { font-size: 13px; color: #7cfc00; }
    h1 { color: #2e8b57; }
    </style>
""", unsafe_allow_html=True)

# 3. Interfaz
st.markdown("<h1 style='text-align: center;'>Plant Doctor IA</h1>", unsafe_allow_html=True)

col_info, col_diag = st.columns([1, 1])

with col_info:
    st.subheader("Beneficios de las hojas")
    beneficios = [("🌿 INSECTICIDA", "Solanina, repelente natural."), ("☕ INFUSIÓN", "Antiinflamatorio."), ("🖊️ CICATRIZACIÓN", "Recuperación cutánea."), ("🌱 FERTILIZANTE", "Nutrientes esenciales.")]
    for t, d in beneficios:
        st.markdown(f"<div class='card'><b>{t}</b><br>{d}</div>", unsafe_allow_html=True)

with col_diag:
    st.subheader("Diagnóstico al Instante")
    uploaded_file = st.file_uploader("Sube una foto de la hoja", type=['jpg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Imagen subida', use_column_width=True)
        
        # Preprocesamiento y Predicción
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        input_tensor = transform(image).unsqueeze(0)
        
        with st.spinner('Analizando...'):
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                clases = ['Enfermo', 'Sano']
                resultado = clases[predicted.item()]
        
        if resultado == 'Enfermo':
            st.error(f"⚠️ Resultado: {resultado}")
        else:
            st.success(f"✅ Resultado: {resultado}")

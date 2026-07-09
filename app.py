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
    /* Fondo oscuro global */
    .stApp { background-color: #121212; color: #e0e0e0; }
    
    /* Títulos principales */
    h1 { color: #4CAF50 !important; text-align: center; margin-bottom: 30px; }
    h3 { color: #ffffff !important; }

    /* Tarjetas de beneficios */
    .card { 
        background-color: #1e1e1e; 
        padding: 15px; 
        border-radius: 12px; 
        margin-bottom: 12px; 
        border-left: 5px solid #4CAF50;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    }
    
    /* Botones verdes profesionales */
    div.stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        border: none !important;
        width: 100%;
        padding: 10px;
        font-weight: bold;
        border-radius: 8px;
    }
    
    /* Ajuste del Uploader para que se vea bien en oscuro */
    [data-testid="stFileUploader"] {
        background-color: #1e1e1e !important;
        border: 2px dashed #4CAF50 !important;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Interfaz
st.markdown("<h1>Plant Doctor IA</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2])

with col_1:
    st.subheader("Beneficios de las hojas")
    beneficios = [("🌿 INSECTICIDA NATURAL", "Contienen solanina, repelente natural contra plagas."), ("☕ INFUSIÓN MEDICINAL", "Antiinflamatorio y alivio gastrointestinal."), ("🖊️ CICATRIZACIÓN", "Sus propiedades naturales ayudan en compresas para acelerar la recuperación de heridas cutáneas leves."), ("🌱 FERTILIZANTE ORGANICO", "Integradas en composta, aportan nutrientes esenciales para fortalecer el sustrato de tu huerto.")]
    for t, d in beneficios:
        st.markdown(f"<div class='card'><b>{t}</b><br>{d}</div>", unsafe_allow_html=True)

with col_2:
    st.subheader("Diagnóstico al Instante")
    st.markdown("<p style='color: #7cfc00; font-weight: bold;'>¡Selecciona o arrastra una foto de la hoja para analizar!</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=['jpg', 'png'])
    
    # 1. El botón de acción
    boton_analizar = st.button("Analizar Hoja")
    
    # 2. Toda la lógica ocurre SOLO cuando hay archivo Y presionas el botón
    if uploaded_file is not None and boton_analizar:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Mostramos la imagen justo al analizar
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
        
        # Resultado final
        if resultado == 'Enfermo':
            st.error(f"⚠️ Resultado: {resultado}")
        else:
            st.success(f"✅ Resultado: {resultado}")

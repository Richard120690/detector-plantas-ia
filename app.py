import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
import os

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

# 3. Interfaz de Streamlit
st.title("Detector de Salud de Plantas")
uploaded_file = st.file_uploader("Sube una foto de la hoja", type=['jpg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Imagen subida', use_column_width=True)
    
    # Preprocesamiento
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    input_tensor = transform(image).unsqueeze(0)
    
    # Predicción
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        clases = ['enfermo', 'sano']
        st.write(f"Resultado: {clases[predicted.item()]}")

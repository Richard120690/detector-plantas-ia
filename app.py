import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

app = FastAPI()
# Estas dos líneas le dicen a FastAPI dónde están tus archivos:
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
# ... (mantén tus mounts y templates igual)

# 1. Definir nombres de clases (ya no necesitas leer clases.txt si quieres, 
# puedes definirlos directamente para evitar errores de orden)
lista_clases = ['enfermo', 'sano'] # Asegúrate que el orden coincida con tu carpeta de entrenamiento

# 2. Arquitectura del modelo (RESNET18)
device = torch.device("cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2) # ¡IMPORTANTE! 2 clases, no num_clases de 1000
model.load_state_dict(torch.load('modelo_tomates.pth', map_location=device))
model.eval()

# 3. Transformaciones para INFERENCIA (sin cosas aleatorias)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- ESTA RUTA ES LA QUE TE FALTA ---
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

from fastapi import HTTPException

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Validación de seguridad en Backend
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="Formato no válido. Por favor, sube solo imágenes (jpg, png, etc.)."
        )
    
    try:
        # 2. Procesamiento seguro
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_t)
            probabilidades = torch.nn.functional.softmax(output[0], dim=0)
            confianza, predicted = torch.max(probabilidades, 0)
            
        nombre_estado = lista_clases[predicted.item()]
        
        return {
            "estado": nombre_estado,
            "confianza": f"{confianza.item()*100:.2f}%"
        }
    except Exception as e:
        # Captura cualquier error técnico (ej. archivo corrupto)
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")




if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000) 
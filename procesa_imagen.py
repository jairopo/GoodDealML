from google.ai.generativelanguage_v1beta.types import content
import PIL.Image
import google.generativeai as genai
import PIL.Image
import json

def getDataFromImage(image, api_key):
    sample_file_1 = PIL.Image.open(image)
    genai.configure(api_key=api_key)
    # Crea la configuración del modelo
    generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_schema": content.Schema(
        type = content.Type.OBJECT,
        enum = [],
        required = ["titulo", "precio_anterior", "precio_actual", "empresa"],
        properties = {
        "titulo": content.Schema(
            type = content.Type.STRING,
        ),
        "precio_anterior": content.Schema(
            type = content.Type.NUMBER,
        ),
        "precio_actual": content.Schema(
            type = content.Type.NUMBER,
        ),
        "empresa": content.Schema(
            type = content.Type.STRING,
        ),
        },
    ),
    "response_mime_type": "application/json",
    }
    # Crea el modelo
    model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config)
    # Establece el prompt para el modelo
    prompt = "Eres un experto en la extracción de títulos de producto, precios y empresa de la imagen proporcionada. Contamos con una lista de empresas (Adidas, AliExpress, Amazon, Asos, Carrefour, Decathlon, El Corte Inglés, Fnac, Game, Lidl, MediaMarkt, Miravia, Outlet PC, PcComponentes, Privalia, Privé by Zalando, Sports Direct, Women'secret, Xiaomi, Zalando, Otras), en caso de desconocerla, colócale \"otras\" ."
    # Recoge la respuesta proporcionando la imagen y el prompt
    response = model.generate_content([sample_file_1, prompt])
    # Devuelve la respuesta en formato json
    return json.loads(response.text)
    
    
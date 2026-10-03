
from src.prediction.service import predict_price_and_time

features = {
    "descricao": "Dragao oriental",
    "estilo": "oriental",
    "tamanho_cm": 25,
    "vermelho": 1,
    "preto": 1,
    "local": "braco",
}

result = predict_price_and_time(features)

print(result)
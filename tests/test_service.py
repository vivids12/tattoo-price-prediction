
from src.prediction.service import predict_price_and_time


def test_predict_price_and_time():
    features = {
        "descricao": "Dragao oriental",
        "estilo": "oriental",
        "tamanho_cm": 25,
        "vermelho": 1,
        "preto": 1,
        "local": "braco",
    }

    result = predict_price_and_time(features)

    assert "preco_sugerido" in result
    assert "tempo_sugerido" in result

    assert isinstance(result["preco_sugerido"], float)
    assert isinstance(result["tempo_sugerido"], float)

    assert result["preco_sugerido"] >= 0
    assert result["tempo_sugerido"] >= 0
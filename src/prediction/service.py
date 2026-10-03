
from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd


# Caminho da pasta raiz do projeto
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

PRICE_MODEL_PATH = MODELS_DIR / "price_model.joblib"
TIME_MODEL_PATH = MODELS_DIR / "time_model.joblib"

FEATURES = [
    "descricao",
    "estilo",
    "tamanho_cm",
    "vermelho",
    "preto",
    "local",
]


@lru_cache(maxsize=1)
def load_models():
    """
    Carrega os modelos treinados uma única vez
    e os mantém disponíveis para as previsões.
    """
    if not PRICE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo de preço não encontrado: {PRICE_MODEL_PATH}"
        )

    if not TIME_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo de tempo não encontrado: {TIME_MODEL_PATH}"
        )

    price_model = joblib.load(PRICE_MODEL_PATH)
    time_model = joblib.load(TIME_MODEL_PATH)

    return price_model, time_model


def predict_price_and_time(features: dict) -> dict:
    """
    Recebe as características de uma tatuagem
    e retorna o preço e o tempo previstos.
    """

    # Verifica se todas as entradas foram informadas
    missing = [key for key in FEATURES if key not in features]

    if missing:
        raise ValueError(
            f"Campos obrigatórios ausentes: {missing}"
        )

    # Prepara os dados no formato esperado pelo modelo
    raw_data = pd.DataFrame(
        [{key: features[key] for key in FEATURES}]
    )

    # Validação básica dos dados
    if raw_data["descricao"].isna().any() or not isinstance(
        features["descricao"], str
    ) or not features["descricao"].strip():
        raise ValueError("A descrição deve ser um texto não vazio.")

    if pd.isna(features["tamanho_cm"]) or features["tamanho_cm"] <= 0:
        raise ValueError("O tamanho deve ser maior que zero.")

    for color in ["vermelho", "preto"]:
        if features[color] not in (0, 1, False, True):
            raise ValueError(f"{color} deve ser 0 ou 1.")

    if not isinstance(features["local"], str) or not features["local"].strip():
        raise ValueError("O local deve ser informado.")

    if not isinstance(features["estilo"], str) or not features["estilo"].strip():
        raise ValueError("O estilo deve ser informado.")

    # Carrega os modelos treinados
    price_model, time_model = load_models()

    # Executa as previsões
    predicted_price = price_model.predict(raw_data)[0]
    predicted_time = time_model.predict(raw_data)[0]

    # Retorna os resultados
    return {
        "preco_sugerido": round(max(0.0, float(predicted_price)), 2),
        "tempo_sugerido": round(max(0.0, float(predicted_time)), 2),
    }
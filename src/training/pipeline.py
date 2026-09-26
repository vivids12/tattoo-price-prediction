from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from src.preprocessing.preprocessor import create_preprocessor


FEATURES = [
    "descricao",
    "tamanho_cm",
    "vermelho",
    "preto",
    "local",
]

TARGET_PRICE = "preco_sugerido"
TARGET_TIME = "tempo_sugerido"


DATA_PATH = Path("data/tatuagens.csv")
MODELS_PATH = Path("models")


def run_training_pipeline():
    # 1. Carrega os dados
    data = pd.read_csv(DATA_PATH)

    # 2. Separa entradas
    X = data[FEATURES]

    # 3. Separa os resultados que queremos prever
    y_price = data[TARGET_PRICE]
    y_time = data[TARGET_TIME]

    # 4. Divide os dados em treinamento e teste
    X_train, X_test, y_price_train, y_price_test, y_time_train, y_time_test = (
        train_test_split(
            X,
            y_price,
            y_time,
            test_size=0.2,
            random_state=42,
        )
    )

    # 5. Cria o pré-processador
    preprocessor = create_preprocessor()

    # 6. Modelo para previsão de preço
    price_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                ),
            ),
        ]
    )

    # 7. Treina o modelo de preço
    price_model.fit(X_train, y_price_train)

    # 8. Faz previsões de preço
    price_predictions = price_model.predict(X_test)

    # 9. Avalia o modelo de preço
    price_error = mean_absolute_error(
        y_price_test,
        price_predictions,
    )

    # 10. Cria o modelo para previsão de tempo
    time_preprocessor = create_preprocessor()

    time_model = Pipeline(
        steps=[
            ("preprocessor", time_preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                ),
            ),
        ]
    )

    # 11. Treina o modelo de tempo
    time_model.fit(X_train, y_time_train)

    # 12. Faz previsões de tempo
    time_predictions = time_model.predict(X_test)

    # 13. Avalia o modelo de tempo
    time_error = mean_absolute_error(
        y_time_test,
        time_predictions,
    )

    # 14. Cria a pasta de modelos
    MODELS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 15. Salva os modelos treinados
    joblib.dump(
        price_model,
        MODELS_PATH / "price_model.joblib",
    )

    joblib.dump(
        time_model,
        MODELS_PATH / "time_model.joblib",
    )

    print("Treinamento concluído!")
    print(f"Erro médio do preço: R$ {price_error:.2f}")
    print(f"Erro médio do tempo: {time_error:.2f} horas")

    return {
        "price_model": price_model,
        "time_model": time_model,
        "price_error": price_error,
        "time_error": time_error,
    }

if __name__ == "__main__":
    run_training_pipeline()
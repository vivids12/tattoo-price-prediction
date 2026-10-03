from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


TEXT_FEATURE = "descricao"
CATEGORICAL_FEATURES = ["local", "estilo"]
NUMERIC_FEATURES = [
    "tamanho_cm",
    "vermelho",
    "preto",
]


def create_preprocessor():
    """
    Cria o pipeline responsável por transformar
    os dados brutos em dados numéricos.
    """

    return ColumnTransformer(
        transformers=[
            (
                "descricao",
                TfidfVectorizer(),
                TEXT_FEATURE,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",
    )
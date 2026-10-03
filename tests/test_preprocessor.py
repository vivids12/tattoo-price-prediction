import pandas as pd
import numpy as np

from src.preprocessing.preprocessor import create_preprocessor


def test_preprocessor():
    data = pd.DataFrame(
        [
            {
                "descricao": "Dragão oriental",
                "estilo": "oriental",
                "tamanho_cm": 25,
                "vermelho": 1,
                "preto": 1,
                "local": "braco",
            },
            {
                "descricao": "Rosa pequena",
                "estilo": "minimalista",
                "tamanho_cm": 8,
                "vermelho": 1,
                "preto": 0,
                "local": "pulso",
            },
        ]
    )

    """
    Testa se o preprocessor transforma os dados brutos em dados numéricos, mantendo o mesmo número de linhas.
    """

    preprocessor = create_preprocessor()
    transformed = preprocessor.fit_transform(data)

    assert transformed.shape[0] == 2

def test_preprocessor_transforms_features_correctly():
    data = pd.DataFrame(
        [
            {
                "descricao": "Dragão oriental",
                "estilo": "oriental",
                "tamanho_cm": 25,
                "vermelho": 1,
                "preto": 1,
                "local": "braco",
            },
            {
                "descricao": "Rosa pequena",
                "estilo": "minimalista",
                "tamanho_cm": 8,
                "vermelho": 1,
                "preto": 0,
                "local": "pulso",
            },
        ]
    )

    preprocessor = create_preprocessor()
    transformed = preprocessor.fit_transform(data)

    # Mantém a quantidade de registros
    assert transformed.shape[0] == len(data)

    # A saída possui colunas geradas pelo texto, local e números
    assert transformed.shape[1] > data.shape[1]

    # A saída contém apenas valores numéricos finitos
    dense_transformed = np.asarray(transformed)

    assert np.issubdtype(dense_transformed.dtype, np.number)
    assert np.isfinite(dense_transformed).all()

    # Verifica se as features esperadas foram criadas
    feature_names = preprocessor.get_feature_names_out()

    assert any("descricao" in name for name in feature_names)
    assert any("local" in name for name in feature_names)
    assert any("estilo" in name for name in feature_names)
    assert any("tamanho_cm" in name for name in feature_names)
    assert any("vermelho" in name for name in feature_names)
    assert any("preto" in name for name in feature_names)
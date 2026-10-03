
from flask import Flask, request, jsonify
from src.prediction.service import predict_price_and_time

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    dados = request.get_json()

    if not dados:
        return jsonify({
            "erro": "O corpo da requisição deve conter um JSON."
        }), 400

    campos_obrigatorios = [
        "descricao",
        "estilo",
        "tamanho_cm",
        "vermelho",
        "preto",
        "local"
    ]

    faltando = [
        campo for campo in campos_obrigatorios
        if campo not in dados
    ]

    if faltando:
        return jsonify({
            "erro": "Campos obrigatórios ausentes.",
            "campos": faltando
        }), 400

    try:
        features = {
            "descricao": str(dados["descricao"]),
            "estilo": str(dados["estilo"]),
            "tamanho_cm": float(dados["tamanho_cm"]),
            "vermelho": int(dados["vermelho"]),
            "preto": int(dados["preto"]),
            "local": str(dados["local"])
        }

        if features["tamanho_cm"] <= 0:
            return jsonify({
                "erro": "O tamanho deve ser maior que zero."
            }), 400

        if features["vermelho"] not in (0, 1) or \
           features["preto"] not in (0, 1):
            return jsonify({
                "erro": "As cores devem ser representadas por 0 ou 1."
            }), 400

        previsao = predict_price_and_time(features)

        return jsonify(previsao), 200

    except (TypeError, ValueError):
        return jsonify({
            "erro": "Os dados enviados possuem formatos inválidos."
        }), 400

    except Exception:
        app.logger.exception("Erro ao realizar a previsão")
        return jsonify({
            "erro": "Não foi possível realizar a previsão."
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
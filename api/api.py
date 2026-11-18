"""
API REST do Argos (IA) - Flask.

Expõe os dois modelos treinados nos notebooks 03 e 04:
  GET  /api/v1/saude          -> {"status": "ok"}
  POST /api/v1/produtividade  -> {"kg_ha_previsto": 959.2, "modelo": "linreg_v1"}
  POST /api/v1/risco-quebra   -> {"risco": "ALTO", "probabilidade": 0.62, "modelo": "logreg_v1"}

Contrato com o front: claude/CONTRATO_IA_FRONT.md (porta 5000, prefixo /api/v1, CORS liberado).
Rodar:  python api.py   (de dentro da pasta api/, com a venv ativada)
"""
import math
from pathlib import Path

import joblib
import sklearn
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

PASTA_MODELOS = Path(__file__).resolve().parent.parent / "modelos"


def carregar_pacote(nome_arquivo):
    pacote = joblib.load(PASTA_MODELOS / nome_arquivo)
    if pacote["versao_sklearn"] != sklearn.__version__:
        print(f"ATENÇÃO: {nome_arquivo} foi treinado com scikit-learn {pacote['versao_sklearn']}, "
              f"mas esta API está com {sklearn.__version__}. Use o requirements.txt.")
    print(f"Carregado {nome_arquivo}: {pacote['nome']} | colunas: {pacote['colunas']}")
    return pacote


# Os modelos são carregados UMA vez, quando a API sobe (fora das rotas).
PACOTE_PRODUTIVIDADE = carregar_pacote("regressao_v1.joblib")
PACOTE_RISCO = carregar_pacote("risco_quebra_v1.joblib")

app = Flask(__name__)
app.json.ensure_ascii = False      # acentos legíveis no JSON de erro
CORS(app)                          # libera o front (localhost:5173) para chamar a API


class ErroDeEntrada(Exception):
    pass


def montar_linha(dados, colunas):
    """Confere os campos que o modelo usa e devolve a linha na ordem do treino."""
    if not isinstance(dados, dict):
        raise ErroDeEntrada("O corpo precisa ser um JSON com os campos do clima")
    linha = []
    for coluna in colunas:
        if coluna not in dados:
            raise ErroDeEntrada(f"Falta o campo {coluna}")
        valor = dados[coluna]
        # bool é subclasse de int em Python: true/false não podem passar como número
        if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not math.isfinite(valor):
            raise ErroDeEntrada(f"O campo {coluna} precisa ser número")
        if coluna == "regiao_sul" and valor not in (0, 1):
            raise ErroDeEntrada("O campo regiao_sul precisa ser 0 (Nordeste) ou 1 (Sul)")
        linha.append(float(valor))
    return [linha]


def classificar_risco(probabilidade, cortes):
    if probabilidade >= cortes["ALTO"]:
        return "ALTO"
    if probabilidade >= cortes["MEDIO"]:
        return "MEDIO"
    return "BAIXO"


@app.route("/api/v1/saude", methods=["GET"])
def saude():
    return jsonify({"status": "ok"})


@app.route("/api/v1/produtividade", methods=["POST"])
def produtividade():
    pacote = PACOTE_PRODUTIVIDADE
    linha = montar_linha(request.get_json(silent=True), pacote["colunas"])
    linha_escalada = pacote["scaler"].transform(linha)
    previsto = float(pacote["modelo"].predict(linha_escalada)[0])
    return jsonify({"kg_ha_previsto": round(previsto, 1), "modelo": pacote["nome"]})


@app.route("/api/v1/risco-quebra", methods=["POST"])
def risco_quebra():
    pacote = PACOTE_RISCO
    linha = montar_linha(request.get_json(silent=True), pacote["colunas"])
    linha_escalada = pacote["scaler"].transform(linha)
    probabilidade = float(pacote["modelo"].predict_proba(linha_escalada)[0, 1])
    return jsonify({
        "risco": classificar_risco(probabilidade, pacote["cortes"]),
        "probabilidade": round(probabilidade, 2),
        "modelo": pacote["nome"],
    })


# ---- Erros sempre em JSON, sem stack trace ----
@app.errorhandler(ErroDeEntrada)
def erro_de_entrada(erro):
    return jsonify({"erro": str(erro)}), 400


@app.errorhandler(404)
def nao_encontrado(erro):
    return jsonify({"erro": "Rota não encontrada. Use /api/v1/saude, /api/v1/produtividade ou /api/v1/risco-quebra"}), 404


@app.errorhandler(405)
def metodo_nao_permitido(erro):
    return jsonify({"erro": "Método não permitido nesta rota"}), 405


@app.errorhandler(Exception)
def erro_interno(erro):
    if isinstance(erro, HTTPException):
        return jsonify({"erro": erro.description}), erro.code
    print("Erro interno:", repr(erro))      # aparece só no terminal da API
    return jsonify({"erro": "Erro interno ao calcular a previsão"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

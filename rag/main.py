#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulator.py
============

Simulador de provas Microsoft Azure AI Fundamentals (AI-901) com Flask.

Usa a API do DeepSeek V4 Flash (modelo `deepseek-v4-flash`) para gerar
questoes de multipla escolha no estilo da prova. O usuario responde clicando:
  - acertou  -> a alternativa fica verde
  - errou    -> a alternativa fica vermelha, mostra qual era a correta (verde)
                e exibe a explicacao logo abaixo.

Cada resposta e gravada em `history.csv` (pergunta, alternativas, correta,
resposta do usuario, acertou/errou, etc.) e essas perguntas NUNCA se repetem.

Gera apenas 5 perguntas por vez para nao sobrecarregar o modelo.

Fonte de dados:
  1. RAG com ChromaDB (base vetorizada com seus documentos de estudo).
  2. Caso o ChromaDB não retorne contexto, o modelo usa seu conhecimento interno.
"""

import os
import csv
import json
import random
import time
import threading
from datetime import datetime

import requests
from flask import Flask, request, jsonify, render_template

# --------------------------------------------------------------------------- #
# Configuracao
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_CSV = os.path.join(BASE_DIR, "history.csv")

from retriever import Retriever  # importa o recuperador ChromaDB
_retriever = None  # será inicializado depois

# =========================================================================== #

from dotenv import load_dotenv
load_dotenv()
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-flash"

EXAME = "ai-901"
QUESTIONS_PER_BATCH = 5  # gera 5 questões por vez

CSV_FIELDS = [
    "timestamp", "topico", "pergunta",
    "opcao_a", "opcao_b", "opcao_c", "opcao_d", "opcao_e",
    "correta", "resposta_usuario", "acertou", "explicacao",
]

# Lock para escrita concorrente no CSV / set de perguntas
_lock = threading.Lock()

# Conjunto em memoria com as perguntas ja vistas (normalizadas), para garantir
# que o agente nunca repita. E populado a partir do CSV no startup e recebe as
# novas perguntas assim que sao geradas.
_perguntas_vistas = set()

app = Flask(__name__)

# --------------------------------------------------------------------------- #
# Topicos oficiais do AI-901 (benchmark do exame atual)
# Distribuicao aproximada de skills medidas pela Microsoft:
#   - Workloads e consideracoes de IA (incl. IA Responsavel)         15-20%
#   - Principios de machine learning no Azure                        20-25%
#   - Computer Vision no Azure                                       15-20%
#   - Natural Language Processing (NLP) no Azure                     15-20%
#   - IA Generativa no Azure                                         15-20%
# --------------------------------------------------------------------------- #
TOPICOS_AI901 = [
    "Workloads de IA: tipos de cargas (visao, NLP, fala, decisao, generativa)",
    "IA Responsavel: imparcialidade (fairness), confiabilidade e seguranca, "
    "privacidade e seguranca, inclusao, transparencia e responsabilidade",
    "Machine Learning: regressao, classificacao e clustering",
    "Aprendizado supervisionado vs nao supervisionado; features e labels",
    "Azure Machine Learning: Automated ML, Designer, endpoints e dados",
    "Metricas de avaliacao de modelos (accuracy, precision, recall, MAE, R2)",
    "Computer Vision: classificacao de imagem, deteccao de objetos, OCR",
    "Azure AI Vision e Custom Vision; analise facial (Face)",
    "NLP: analise de sentimento, extracao de frases-chave, reconhecimento "
    "de entidades (NER), deteccao de idioma",
    "Azure AI Language e Conversational Language Understanding (CLU)",
    "Azure AI Speech: reconhecimento de fala, sintese de fala, traducao",
    "Document Intelligence (Form Recognizer) e Knowledge Mining (Azure AI Search)",
    "IA Generativa: modelos de linguagem, prompts, capacidades e limitacoes",
    "Azure OpenAI Service, Microsoft Copilot e cenarios de IA generativa",
    "Conceitos de modelos generativos: tokens, fundamentacao (grounding), "
    "alucinacoes e engenharia de prompt",
]

# --------------------------------------------------------------------------- #
# RAG com ChromaDB
# --------------------------------------------------------------------------- #
def inicializar_chromadb():
    """Inicializa o retriever ChromaDB."""
    global _retriever
    try:
        _retriever = Retriever(
            persist_dir="./chroma_db",
            collection_name="meus_documentos"  # ajuste conforme seu ingest
        )
        print("[RAG] ChromaDB carregado com sucesso.")
        return True
    except Exception as e:
        print(f"[RAG] Erro ao carregar ChromaDB: {e}")
        return False


def recuperar_contexto(consulta, k=4):
    """Retorna os k chunks mais relevantes do ChromaDB."""
    if _retriever is None:
        return ""
    try:
        docs = _retriever.retrieve(consulta, top_k=k)
        if not docs:
            return ""
        return "\n\n---\n\n".join(doc["texto"] for doc in docs)
    except Exception as e:
        print(f"[RAG] Erro na recuperação: {e}")
        return ""


# --------------------------------------------------------------------------- #
# Historico (CSV) e controle de perguntas ja vistas
# --------------------------------------------------------------------------- #
def _normalizar(texto):
    return " ".join((texto or "").lower().split())


def garantir_csv():
    if not os.path.isfile(HISTORY_CSV):
        with open(HISTORY_CSV, "w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def carregar_perguntas_vistas():
    """Popula o set de perguntas ja respondidas a partir do CSV (persistencia)."""
    if not os.path.isfile(HISTORY_CSV):
        return
    try:
        with open(HISTORY_CSV, "r", newline="", encoding="utf-8-sig") as f:
            for linha in csv.DictReader(f):
                p = linha.get("pergunta")
                if p:
                    _perguntas_vistas.add(_normalizar(p))
        print(f"[HIST] {len(_perguntas_vistas)} pergunta(s) ja respondidas carregadas do CSV.")
    except Exception as e:
        print(f"[HIST] Nao foi possivel ler o historico ({e}).")


def salvar_resposta(registro):
    """Grava uma resposta no CSV."""
    with _lock:
        garantir_csv()
        with open(HISTORY_CSV, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writerow({campo: registro.get(campo, "") for campo in CSV_FIELDS})
        _perguntas_vistas.add(_normalizar(registro.get("pergunta", "")))


def estatisticas():
    acertos = erros = 0
    if os.path.isfile(HISTORY_CSV):
        try:
            with open(HISTORY_CSV, "r", newline="", encoding="utf-8-sig") as f:
                for linha in csv.DictReader(f):
                    if str(linha.get("acertou", "")).lower() in ("1", "true", "sim"):
                        acertos += 1
                    else:
                        erros += 1
        except Exception:
            pass
    total = acertos + erros
    return {"acertos": acertos, "erros": erros, "total": total}


def topicos_menos_vistos():
    contagem = {t: 0 for t in TOPICOS_AI901}
    if os.path.isfile(HISTORY_CSV):
        try:
            with open(HISTORY_CSV, "r", newline="", encoding="utf-8-sig") as f:
                for linha in csv.DictReader(f):
                    topico = linha.get("topico", "")
                    for t in TOPICOS_AI901:
                        if topico in t:
                            contagem[t] += 1
        except Exception as e:
            print(f"[HIST] Erro ao calcular tópicos menos vistos ({e}). Usando amostra aleatória.")
            return random.sample(TOPICOS_AI901, k=5)
    return sorted(contagem, key=contagem.get)[:5]


# --------------------------------------------------------------------------- #
# Geracao de perguntas via DeepSeek V4 Flash
# --------------------------------------------------------------------------- #
def montar_prompt(evitar, contexto):
    """Monta as mensagens (system + user) para o modelo."""
    topicos_escolhidos = topicos_menos_vistos()
    topicos_txt = "\n".join(f"- {t}" for t in topicos_escolhidos)

    # Lista de perguntas a evitar (limitada para nao estourar contexto)
    evitar_amostra = evitar[-120:] if len(evitar) > 120 else evitar
    evitar_txt = "\n".join(f"- {p}" for p in evitar_amostra) if evitar_amostra else "(nenhuma ainda)"

    bloco_contexto = ""
    if contexto:
        rotulo = "CONTEÚDO EXTRAÍDO DOS SEUS DOCUMENTOS DE ESTUDO (use como base principal)."
        bloco_contexto = f"\n\n{rotulo}\n\"\"\"\n{contexto}\n\"\"\""
    else:
        bloco_contexto = "\n\n(Nenhum material de referência fornecido. Use apenas seu conhecimento.)"

    system = (
        "Voce eh um especialista certificador da Microsoft que cria questoes "
        "realistas para o exame AI-901 (Azure AI Fundamentals). As questoes "
        "devem refletir o edital atual: focadas fortemente no Microsoft Foundry, "
        "desenvolvimento de agentes simples, aplicacoes leves com SDK Python, modelos "
        "multimodais, Azure Content Understanding e principios de IA Responsavel. "
        "As questoes devem ser conceituais e baseadas em cenarios praticos (quando e "
        "como usar os recursos do Foundry). Evite pegadinhas com numeros ou decoreba; "
        "foque na capacidade de escolher e aplicar a ferramenta certa do ecossistema "
        "Foundry. Responda SEMPRE em inglês e SEMPRE em JSON valido."
    )

    user = f"""Gere EXATAMENTE {QUESTIONS_PER_BATCH} questoes de multipla escolha para o exame AI-901.

Regras:
- Cada questao tem 5 alternativas (A, B, C, D e E) e APENAS UMA correta.
- Varie os topicos. Priorize estes nesta rodada:
{topicos_txt}
- Linguagem clara, estilo oficial Microsoft, em inglês.
- Distribua as questões em 3 níveis: 2 fáceis, 2 intermediárias, 1 difícil.
- A explicacao deve justificar por que a correta esta certa e, brevemente, por que as outras nao.
- NAO repita nenhuma destas perguntas ja utilizadas (nem variacoes equivalentes):
{evitar_txt}
{bloco_contexto}
- As perguntas DEVEM ser baseadas EXCLUSIVAMENTE no conteúdo fornecido no bloco de contexto (se houver).
- Se o contexto não contiver informações suficientes sobre um tópico, não invente; simplesmente não gere perguntas sobre ele.
- Use o contexto para formular cenários práticos e específicos.

Responda APENAS com um objeto JSON neste formato exato (sem texto fora do JSON):
{{
  "questions": [
    {{
      "topico": "string (um dos dominios do AI-901)",
      "pergunta": "string",
      "opcoes": ["alternativa A", "alternativa B", "alternativa C", "alternativa D", "alternativa E"],
      "indice_correto": 0,
      "explicacao": "string explicando a resposta correta"
    }}
  ]
}}
O campo "indice_correto" e o indice (0 a 4) da alternativa correta dentro de "opcoes".
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def chamar_deepseek(messages, tentativas=3):
    """Chama a API do DeepSeek V4 Flash e retorna o conteudo de texto (JSON)."""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "Variavel de ambiente DEEPSEEK_API_KEY nao definida. "
            "Defina sua chave antes de iniciar o simulador."
        )
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.9,
        "max_tokens": 4000,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    for i in range(tentativas):
        try:
            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.HTTPError as e:
            if e.response.status_code in (429, 502, 503) and i < tentativas - 1:
                time.sleep(2 ** i)
                continue
            raise


def gerar_perguntas():
    """Gera um lote de perguntas inéditas usando ChromaDB como fonte de contexto."""
    evitar = sorted(_perguntas_vistas)

    # 1) Tenta recuperar do ChromaDB
    contexto = ""
    if _retriever is not None:
        topicos_consulta = topicos_menos_vistos()
        consulta = " ".join(topicos_consulta[:3])
        contexto = recuperar_contexto(consulta, k=6)
        if contexto:
            print("[RAG] Usando contexto do ChromaDB.")

    # 2) Monta o prompt com o contexto (pode ser vazio)
    messages = montar_prompt(evitar, contexto)
    conteudo = chamar_deepseek(messages)

    # Parse robusto do JSON retornado
    try:
        parsed = json.loads(conteudo)
    except json.JSONDecodeError:
        inicio = conteudo.find("{")
        fim = conteudo.rfind("}")
        if inicio == -1 or fim == -1:
            raise RuntimeError("O modelo nao retornou JSON valido.")
        parsed = json.loads(conteudo[inicio:fim + 1])

    brutas = parsed.get("questions", []) if isinstance(parsed, dict) else []
    perguntas = []
    for q in brutas:
        pergunta = (q.get("pergunta") or "").strip()
        opcoes = q.get("opcoes") or []
        idx = q.get("indice_correto")
        explicacao = (q.get("explicacao") or "").strip()
        topico = (q.get("topico") or "AI-901").strip()

        if not pergunta or not isinstance(opcoes, list) or len(opcoes) != 5:
            continue
        if not isinstance(idx, int) or idx < 0 or idx > 4:
            continue
        if _normalizar(pergunta) in _perguntas_vistas:
            continue

        opcoes = [str(o).strip() for o in opcoes]
        perguntas.append({
            "topico": topico,
            "pergunta": pergunta,
            "opcoes": opcoes,
            "indice_correto": idx,
            "explicacao": explicacao,
        })
        _perguntas_vistas.add(_normalizar(pergunta))

    return perguntas


# --------------------------------------------------------------------------- #
# Rotas
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    fonte = "chromadb" if _retriever is not None else "nenhuma"
    return render_template("index.html", fonte=fonte, exame=EXAME)


@app.route("/api/questions")
def api_questions():
    try:
        perguntas = gerar_perguntas()
        if not perguntas:
            return jsonify({"erro": "Nenhuma pergunta nova foi gerada. Tente novamente."}), 502
        return jsonify({
            "questions": perguntas,
            "fonte": "chromadb" if _retriever is not None else "nenhuma"
        })
    except requests.HTTPError as e:
        detalhe = ""
        try:
            detalhe = e.response.text[:300]
        except Exception:
            pass
        return jsonify({"erro": f"Erro da API DeepSeek: {e}. {detalhe}"}), 502
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route("/api/answer", methods=["POST"])
def api_answer():
    dados = request.get_json(force=True, silent=True) or {}
    opcoes = dados.get("opcoes") or ["", "", "", "", ""]
    idx_correto = dados.get("indice_correto", -1)
    idx_usuario = dados.get("resposta_usuario", -1)

    try:
        acertou = int(idx_usuario) == int(idx_correto)
    except (TypeError, ValueError):
        acertou = False

    def safe(i):
        return opcoes[i] if isinstance(opcoes, list) and 0 <= i < len(opcoes) else ""

    registro = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "topico": dados.get("topico", ""),
        "pergunta": dados.get("pergunta", ""),
        "opcao_a": safe(0),
        "opcao_b": safe(1),
        "opcao_c": safe(2),
        "opcao_d": safe(3),
        "opcao_e": safe(4),
        "correta": safe(idx_correto) if isinstance(idx_correto, int) else "",
        "resposta_usuario": safe(idx_usuario) if isinstance(idx_usuario, int) else "",
        "acertou": "1" if acertou else "0",
        "explicacao": dados.get("explicacao", ""),
    }
    salvar_resposta(registro)
    return jsonify({"ok": True, "acertou": acertou, "stats": estatisticas()})


@app.route("/api/stats")
def api_stats():
    return jsonify(estatisticas())


# --------------------------------------------------------------------------- #
# Inicializacao
# --------------------------------------------------------------------------- #
def inicializar():
    garantir_csv()
    carregar_perguntas_vistas()
    if not inicializar_chromadb():
        print("[RAG] ChromaDB indisponível. Usando conhecimento interno do modelo.")


if __name__ == "__main__":
    inicializar()
    print(f"Simulador {EXAME.upper()} — ")
    print("Abra http://127.0.0.1:5000 no navegador.")
    app.run(host="127.0.0.1", port=5000, debug=False)
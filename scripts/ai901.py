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

Fonte de dados (em ordem de prioridade):
  1. Web scraping da pagina oficial da certificacao no Microsoft Learn
     (configurado via CERT_URL). Extrai e injeta o conteudo da pagina como
     contexto para o modelo gerar questoes fieis ao exame.
  2. RAG opcional: se existir uma base vetorizada em `vectors/index.pkl`
     (gerada pelo `vectorizator.py` a partir de arquivos .txt na pasta
     `docs/`), ela e usada como fallback.
  3. Se nenhuma das anteriores estiver disponivel, o modelo gera questoes
     usando apenas seu conhecimento interno.

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
VECTORS_PATH = os.path.join(BASE_DIR, "vectors", "index.pkl")
HISTORY_CSV = os.path.join(BASE_DIR, "history.csv")

# =========================================================================== #

from dotenv import load_dotenv
load_dotenv()
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-flash"


EXAME = "ai-901"
CERT_URL = "https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/ai-901"

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

# Cache do conteudo web (evita re-fetch a cada lote)
_web_cache = {"conteudo": "", "timestamp": 0}

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
# RAG opcional (carregado so se a base vetorizada existir)
# --------------------------------------------------------------------------- #
_rag = {"ativo": False, "vectorizer": None, "matriz": None, "chunks": None}


def carregar_rag():
    """Tenta carregar a base vetorizada. Falha silenciosa -> simulador sem RAG."""
    if not os.path.isfile(VECTORS_PATH):
        print("[RAG] Nenhuma base vetorizada encontrada. Rodando sem RAG.")
        return
    try:
        import pickle  # lazy import
        with open(VECTORS_PATH, "rb") as f:
            indice = pickle.load(f)
        _rag["vectorizer"] = indice["vectorizer"]
        _rag["matriz"] = indice["matriz"]
        _rag["chunks"] = indice["chunks"]
        _rag["ativo"] = True
        print(f"[RAG] Base vetorizada carregada: {len(indice['chunks'])} chunks. RAG ATIVO.")
    except Exception as e:  # noqa: BLE001 - qualquer falha = seguir sem RAG
        print(f"[RAG] Falha ao carregar base vetorizada ({e}). Rodando sem RAG.")
        _rag["ativo"] = False


def recuperar_contexto(consulta, k=4):
    """Retorna os k chunks mais relevantes para a consulta (ou '' se sem RAG)."""
    if not _rag["ativo"]:
        return ""
    try:
        from sklearn.metrics.pairwise import cosine_similarity  # lazy import
        vetor_consulta = _rag["vectorizer"].transform([consulta])
        sims = cosine_similarity(vetor_consulta, _rag["matriz"])[0]
        melhores = sims.argsort()[::-1][:k]
        trechos = [_rag["chunks"][i] for i in melhores if sims[i] > 0.0]
        return "\n\n---\n\n".join(trechos)
    except Exception as e:  # noqa: BLE001
        print(f"[RAG] Erro na recuperacao ({e}). Ignorando contexto.")
        return ""


def buscar_conteudo_web(url):
    """Busca e extrai o conteudo textual de uma URL (cache de 1 hora)."""
    agora = time.time()
    if _web_cache["conteudo"] and (agora - _web_cache["timestamp"]) < 3600:
        return _web_cache["conteudo"]

    try:
        from bs4 import BeautifulSoup  # lazy import
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/130.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        # Tenta capturar a area principal do conteudo
        main = soup.find("main") or soup.find("article") or soup.find("body")
        texto = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

        # Filtra linhas curtas (navegacao, menus) e normaliza
        linhas = [l.strip() for l in texto.split("\n")
                  if l.strip() and len(l.strip()) > 20]
        conteudo = "\n".join(linhas)

        # Limita tamanho para nao estourar o prompt do modelo
        if len(conteudo) > 12000:
            conteudo = conteudo[:12000] + "\n\n... (conteudo truncado)"

        _web_cache["conteudo"] = conteudo
        _web_cache["timestamp"] = agora
        print(f"[WEB] Conteudo extraido de {url} ({len(conteudo)} caracteres).")
        return conteudo
    except ImportError:
        print("[WEB] beautifulsoup4 nao instalado. Rode: pip install beautifulsoup4")
        return ""
    except Exception as e:
        print(f"[WEB] Falha ao acessar {url}: {e}")
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
    except Exception as e:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            pass
    total = acertos + erros
    return {"acertos": acertos, "erros": erros, "total": total}


def topicos_menos_vistos():
    contagem = {t: 0 for t in TOPICOS_AI901}
    if os.path.isfile(HISTORY_CSV):
        try:  # ← adicionar
            with open(HISTORY_CSV, "r", newline="", encoding="utf-8-sig") as f:
                for linha in csv.DictReader(f):
                    topico = linha.get("topico", "")
                    for t in TOPICOS_AI901:
                        if topico in t:
                            contagem[t] += 1
        except Exception as e:
            print(f"[HIST] Erro ao calcular tópicos menos vistos ({e}). Usando amostra aleatória.")
            return random.sample(TOPICOS_AI901, k=5)  # fallback seguro
    return sorted(contagem, key=contagem.get)[:5]

# --------------------------------------------------------------------------- #
# Geracao de perguntas via DeepSeek V4 Flash
# --------------------------------------------------------------------------- #
def montar_prompt(evitar, contexto, usando_web=False):
    """Monta as mensagens (system + user) para o modelo."""
    topicos_escolhidos = topicos_menos_vistos()
    topicos_txt = "\n".join(f"- {t}" for t in topicos_escolhidos)

    # Lista de perguntas a evitar (limitada para nao estourar contexto)
    evitar_amostra = evitar[-120:] if len(evitar) > 120 else evitar
    evitar_txt = "\n".join(f"- {p}" for p in evitar_amostra) if evitar_amostra else "(nenhuma ainda)"

    bloco_contexto = ""
    if contexto:
        if usando_web:
            rotulo = (
                "CONTEUDO OFICIAL DA PAGINA DE CERTIFICACAO (Microsoft Learn). "
                "Use como base principal para gerar as questoes, garantindo "
                "fidelidade ao que e cobrado no exame real:"
            )
        else:
            rotulo = (
                "MATERIAL DE ESTUDO FORNECIDO PELO USUARIO (use como base "
                "preferencial, mas mantenha o estilo oficial da prova):"
            )
        bloco_contexto = f"\n\n{rotulo}\n\"\"\"\n{contexto}\n\"\"\""

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
        "temperature": 0.9,             # variedade nas questoes
        "max_tokens": 4000,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},  # modo rapido (permite temperature)
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
                time.sleep(2 ** i)  # espera 1s, 2s, 4s...
                continue
            raise


def gerar_perguntas():
    """Gera um lote de perguntas inéditas (dedup contra _perguntas_vistas)."""
    evitar = sorted(_perguntas_vistas)  # so para passar ao modelo como referencia

    usando_web = False
    contexto = ""

    # 1) Tenta usar conteudo da pagina oficial da certificacao (web)
    if CERT_URL:
        contexto_web = buscar_conteudo_web(CERT_URL)
        if contexto_web:
            contexto = contexto_web
            usando_web = True

    # 2) Fallback: RAG com arquivos locais (docs/)
    if not contexto:
        consulta_rag = " ".join(random.sample(TOPICOS_AI901, k=min(3, len(TOPICOS_AI901))))
        contexto = recuperar_contexto(consulta_rag, k=4)

    messages = montar_prompt(evitar, contexto, usando_web=usando_web)
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

        # Validacoes basicas
        if not pergunta or not isinstance(opcoes, list) or len(opcoes) != 5:
            continue
        if not isinstance(idx, int) or idx < 0 or idx > 4:
            continue
        if _normalizar(pergunta) in _perguntas_vistas:
            continue  # garante que nunca repete

        opcoes = [str(o).strip() for o in opcoes]
        perguntas.append({
            "topico": topico,
            "pergunta": pergunta,
            "opcoes": opcoes,
            "indice_correto": idx,
            "explicacao": explicacao,
        })
        # Marca como vista imediatamente para nao repetir em lotes seguintes
        _perguntas_vistas.add(_normalizar(pergunta))

    return perguntas


# --------------------------------------------------------------------------- #
# Rotas
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    fonte = "nenhuma"
    if CERT_URL and _web_cache["conteudo"]:
        fonte = "web"
    elif _rag["ativo"]:
        fonte = "rag"
    return render_template("index.html", fonte=fonte, exame=EXAME)  # ← só isso muda


@app.route("/api/questions")
def api_questions():
    try:
        perguntas = gerar_perguntas()
        if not perguntas:
            return jsonify({"erro": "Nenhuma pergunta nova foi gerada. Tente novamente."}), 502
        return jsonify({"questions": perguntas, "fonte": "web" if _web_cache["conteudo"] else ("rag" if _rag["ativo"] else "nenhuma")})
    except requests.HTTPError as e:
        detalhe = ""
        try:
            detalhe = e.response.text[:300]
        except Exception:  # noqa: BLE001
            pass
        return jsonify({"erro": f"Erro da API DeepSeek: {e}. {detalhe}"}), 502
    except Exception as e:  # noqa: BLE001
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
    carregar_rag()
    if CERT_URL:
        print(f"[WEB] URL da certificacao: {CERT_URL}")
    if not DEEPSEEK_API_KEY:
        print("\n[AVISO] DEEPSEEK_API_KEY nao definida. A geracao de perguntas vai falhar.")
        print("        Defina a chave e reinicie:")
        print('        PowerShell:  $env:DEEPSEEK_API_KEY="sk-sua-chave"')
        print("        CMD:         set DEEPSEEK_API_KEY=sk-sua-chave\n")


if __name__ == "__main__":
    inicializar()
    print(f"Simulador {EXAME.upper()} — ")
    print("Abra http://127.0.0.1:5000 no navegador.")
    app.run(host="127.0.0.1", port=5000, debug=False)
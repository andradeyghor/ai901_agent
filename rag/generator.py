#!/usr/bin/env python3
"""
generator.py - Gera respostas a partir do contexto recuperado no ChromaDB.
Suporta:
- DeepSeek (via API compatível com OpenAI)
- OpenAI (GPT-3.5/4)
- Ollama (local, gratuito)

Uso:
    python generator.py --query "Sua pergunta" --model deepseek
    python generator.py --query "Sua pergunta" --model ollama --top-k 3
"""

import os
import sys
import argparse
from typing import Optional

# Verifica e importa dependências
try:
    import openai
except ImportError:
    openai = None

try:
    import requests
except ImportError:
    requests = None


class Generator:
    """
    Classe para gerar respostas usando LLM a partir de um contexto.
    """

    def __init__(
        self,
        model_type: str = "ollama",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        """
        Inicializa o gerador.

        Args:
            model_type: "openai", "deepseek" ou "ollama"
            model_name: nome do modelo (ex: "gpt-3.5-turbo", "deepseek-chat", "llama3")
            api_key: chave da API (se não fornecida, tenta via variável de ambiente)
            base_url: URL base da API (opcional, usado para DeepSeek ou outros compatíveis)
            temperature: temperatura da geração (0.0 a 1.0)
            max_tokens: número máximo de tokens na resposta
        """
        self.model_type = model_type
        self.temperature = temperature
        self.max_tokens = max_tokens

        if model_type == "openai":
            if openai is None:
                raise ImportError(
                    "Biblioteca 'openai' não instalada. Execute: pip install openai"
                )
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY não definida. Passe via argumento ou variável de ambiente."
                )
            self.client = openai.OpenAI(api_key=api_key)
            self.model_name = model_name or "gpt-3.5-turbo"

        elif model_type == "deepseek":
            if openai is None:
                raise ImportError(
                    "Biblioteca 'openai' não instalada. Execute: pip install openai"
                )
            api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError(
                    "DEEPSEEK_API_KEY não definida. Passe via argumento ou variável de ambiente."
                )
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url or "https://api.deepseek.com/v1"
            )
            self.model_name = model_name or "deepseek-chat"

        elif model_type == "ollama":
            if requests is None:
                raise ImportError(
                    "Biblioteca 'requests' não instalada. Execute: pip install requests"
                )
            self.model_name = model_name or "llama3"
            self.api_url = base_url or "http://localhost:11434/api/generate"
            # Testa conectividade com o servidor Ollama
            try:
                resp = requests.get("http://localhost:11434", timeout=2)
                if resp.status_code != 200:
                    print("⚠️  Ollama parece estar rodando, mas retornou status:", resp.status_code)
            except requests.ConnectionError:
                print(
                    "⚠️  Não foi possível conectar ao Ollama. Certifique-se que o servidor está rodando:"
                    "\n   $ ollama serve"
                )
        else:
            raise ValueError(f"model_type inválido: {model_type}. Use 'openai', 'deepseek' ou 'ollama'.")

    def generate(self, query: str, context: str, system_prompt: Optional[str] = None) -> str:
        """
        Gera uma resposta para a query usando o contexto fornecido.

        Args:
            query: pergunta do usuário
            context: contexto recuperado (texto dos chunks)
            system_prompt: prompt de sistema personalizado (opcional)

        Returns:
            resposta gerada pelo LLM
        """
        if not context or context == "Nenhum documento relevante encontrado.":
            return "Desculpe, não encontrei informações relevantes para sua pergunta."

        if not system_prompt:
            system_prompt = (
                "Você é um assistente útil e preciso. Responda à pergunta do usuário "
                "usando APENAS as informações fornecidas no contexto. "
                "Se a resposta não estiver no contexto, diga que não sabe."
            )

        user_prompt = f"""
Contexto:
{context}

Pergunta: {query}

Resposta (baseada APENAS no contexto acima):
"""

        if self.model_type in ("openai", "deepseek"):
            return self._generate_openai(system_prompt, user_prompt)
        elif self.model_type == "ollama":
            return self._generate_ollama(system_prompt, user_prompt)
        else:
            raise RuntimeError(f"Tipo de modelo não suportado: {self.model_type}")

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Chama a API OpenAI/DeepSeek (formato compatível)."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erro ao chamar API: {str(e)}"

    def _generate_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Chama a API do Ollama (geração sem streaming)."""
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "Erro: resposta vazia do Ollama.")
        except requests.exceptions.RequestException as e:
            return f"Erro ao conectar com Ollama: {str(e)}"


def rag_pipeline(
    query: str,
    top_k: int = 5,
    filter_meta: Optional[dict] = None,
    model_type: str = "ollama",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Função que orquestra o pipeline RAG completo: recuperação + geração.

    Args:
        query: pergunta do usuário
        top_k: número de chunks a recuperar
        filter_meta: filtro por metadados (ex: {"topico": "machine-learning"})
        model_type: "openai", "deepseek" ou "ollama"
        model_name: nome do modelo (opcional)
        api_key: chave da API (opcional)

    Returns:
        resposta gerada
    """
    # Importação local para evitar dependência circular
    try:
        from retriever import Retriever
    except ImportError:
        raise ImportError(
            "Arquivo retriever.py não encontrado. Certifique-se que está no mesmo diretório."
        )

    retriever = Retriever()
    context = retriever.retrieve_formatted(query, top_k, filter_meta)

    generator = Generator(
        model_type=model_type,
        model_name=model_name,
        api_key=api_key
    )
    return generator.generate(query, context)


def main():
    parser = argparse.ArgumentParser(
        description="Gera respostas RAG usando contexto do ChromaDB."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Pergunta do usuário"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Número de chunks a recuperar (padrão: 5)"
    )
    parser.add_argument(
        "--model",
        choices=["openai", "deepseek", "ollama"],
        default="ollama",
        help="Tipo de modelo (padrão: ollama)"
    )
    parser.add_argument(
        "--model-name",
        help="Nome específico do modelo (ex: 'gpt-4', 'deepseek-chat', 'llama3')"
    )
    parser.add_argument(
        "--api-key",
        help="Chave da API (se não fornecida, usa variável de ambiente)"
    )
    parser.add_argument(
        "--filter",
        nargs=2,
        action="append",
        metavar=("KEY", "VALUE"),
        help="Filtro por metadado (ex: --filter topico machine-learning). Pode repetir."
    )
    args = parser.parse_args()

    # Converte filtros para dicionário
    filter_meta = None
    if args.filter:
        filter_meta = {k: v for k, v in args.filter}

    # Executa o pipeline
    resposta = rag_pipeline(
        query=args.query,
        top_k=args.top_k,
        filter_meta=filter_meta,
        model_type=args.model,
        model_name=args.model_name,
        api_key=args.api_key
    )

    print("\n" + "=" * 60)
    print(f"🔍 Pergunta: {args.query}")
    print("=" * 60)
    print("\n📝 Resposta:\n")
    print(resposta)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
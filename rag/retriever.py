"""
retriever.py - Módulo para buscar chunks relevantes no ChromaDB.
Uso: python retriever.py --query "Qual é a sua pergunta?" --top-k 5
"""

import argparse
import chromadb
from chromadb.utils import embedding_functions

import os
from dotenv import load_dotenv

load_dotenv()  # Carrega as variáveis do .env

class Retriever:
    def __init__(self, persist_dir="./chroma_db", collection_name="documentos_ai901"):
        # Inicializa o cliente
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Cria a MESMA função de embedding usada no ingest
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # =============================================
        # OPÇÃO OPENAI (paga, via API) - Descomente para usar
        # =============================================
        # from dotenv import load_dotenv
        # import os
        # load_dotenv()
        # api_key = os.getenv("OPENAI_API_KEY")
        # if not api_key:
        #     raise ValueError("OPENAI_API_KEY não encontrada no .env")
        # embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        #     api_key=api_key,
        #     model_name="text-embedding-ada-002"
        # )


        # Passa a função explicitamente para garantir que as queries sejam vetorizadas corretamente
        self.collection = self.client.get_collection(
            collection_name,
            embedding_function=embedding_fn
        )
        
    def retrieve(self, query, top_k=5, filter_meta=None):
        """
        Busca os top_k chunks mais relevantes para a query.
        filter_meta: dicionário opcional para filtrar por metadados (ex: {"topico": "machine-learning"})
        """
        resultados = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_meta  # Filtro opcional
        )
        
        # Estrutura a saída em uma lista de dicionários
        retrieved_docs = []
        if resultados["documents"]:
            for i, doc in enumerate(resultados["documents"][0]):
                retrieved_docs.append({
                    "texto": doc,
                    "metadata": resultados["metadatas"][0][i],
                    "id": resultados["ids"][0][i],
                    "distancia": resultados["distances"][0][i] if "distances" in resultados else None
                })
        return retrieved_docs

    def retrieve_formatted(self, query, top_k=5, filter_meta=None):
        """Retorna o texto puro dos documentos concatenados, para colocar no prompt."""
        docs = self.retrieve(query, top_k, filter_meta)
        if not docs:
            return "Nenhum documento relevante encontrado."
        
        contextos = []
        for i, doc in enumerate(docs):
            contextos.append(f"[Documento {i+1}] {doc['texto']}")
        
        return "\n\n".join(contextos)


def main():
    parser = argparse.ArgumentParser(description="Testar o recuperador")
    parser.add_argument("--query", required=True, help="Pergunta do usuário")
    parser.add_argument("--top-k", type=int, default=5, help="Número de chunks a recuperar")
    parser.add_argument("--persist-dir", default="./chroma_db")
    parser.add_argument("--collection", default="documentos_ai901")
    args = parser.parse_args()

    retriever = Retriever(args.persist_dir, args.collection)
    docs = retriever.retrieve(args.query, args.top_k)

    print(f"\n🔍 Top {len(docs)} resultados para: '{args.query}'\n")
    for i, doc in enumerate(docs):
        print(f"--- Resultado {i+1} (ID: {doc['id']}, Distância: {doc['distancia']:.4f}) ---")
        print(f"Arquivo: {doc['metadata'].get('arquivo', 'N/A')}")
        print(f"Seção: {doc['metadata'].get('secao', 'N/A')}")
        print(f"Texto: {doc['texto'][:200]}...\n")

if __name__ == "__main__":
    main()
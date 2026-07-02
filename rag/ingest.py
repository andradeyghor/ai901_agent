"""
Script para ingerir chunks.json no ChromaDB.
Uso: python ingest.py --chunks chunks.json --collection meu_docs
"""

import json
import argparse
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

def main():
    parser = argparse.ArgumentParser(description="Ingerir chunks no ChromaDB")
    parser.add_argument("--chunks", default="chunks/chunks.json", help="Arquivo JSON gerado pelo chunker")
    parser.add_argument("--collection", default="documentos_ai901", help="Nome da coleção no ChromaDB")
    parser.add_argument("--persist-dir", default="./chroma_db", help="Diretório de persistência")
    parser.add_argument("--batch-size", type=int, default=100, help="Tamanho do lote para inserção")
    args = parser.parse_args()

    # 1. Carregar os chunks do JSON
    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise SystemExit(f"Arquivo não encontrado: {chunks_path}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise SystemExit("Nenhum chunk encontrado no JSON.")

    print(f"Carregados {len(chunks)} chunks do arquivo.")

    # 2. Configurar a função de embedding (local, gratuita)
    # se for usar modelo local
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # Pode trocar por outro modelo, ex: "all-mpnet-base-v2"
    )

        # # se for usar tiktoken (OpenAI)
        # embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        #     api_key="sua-chave",
        #     model_name="text-embedding-ada-002"
        # )

    # 3. Inicializar cliente persistente
    client = chromadb.PersistentClient(path=args.persist_dir)

    # 4. Obter ou criar a coleção
    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=embedding_fn
    )

    # 5. Preparar os dados para inserção
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        # O ID do chunk já é único e descritivo
        ids.append(chunk["id"])
        documents.append(chunk["texto"])

        # Metadados: campos úteis para filtragem
        meta = {
            "topico": chunk.get("topico", ""),
            "arquivo": chunk.get("arquivo", ""),
            "secao": chunk.get("secao", ""),
            "nivel_secao": chunk.get("nivel_secao", 0),
            "num_tokens": chunk.get("num_tokens", 0)
        }
        metadatas.append(meta)

    # 6. Inserir em lotes
    total = len(ids)
    for i in range(0, total, args.batch_size):
        batch_end = min(i + args.batch_size, total)
        collection.add(
            ids=ids[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )
        print(f"Inseridos {batch_end} de {total} chunks...")

    print(f"\n✅ Todos os {total} chunks foram inseridos na coleção '{args.collection}'.")
    print(f"Persistência em: {args.persist_dir}")

    # (Opcional) Exibir estatísticas
    count = collection.count()
    print(f"Total de documentos na coleção: {count}")

if __name__ == "__main__":
    main()
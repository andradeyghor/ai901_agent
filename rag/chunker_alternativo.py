# from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter

# # Inicializa o divisor com o modelo desejado e o limite de tokens
# splitter = SentenceTransformersTokenTextSplitter(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     tokens_per_chunk=256,
#     chunk_overlap=50 # Sobreposição para manter o contexto entre os chunks
# )

# texto = "Seu texto longo vai aqui..."
# chunks = splitter.split_text(text=texto)

# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i+1}: {chunk}\n")


pip install langchain langchain-experimental langchain-community sentence-transformers

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Modelo local (mesmo que você usa no ingest)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Cria o divisor semântico
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # ou "standard_deviation", "interquartile"
    min_chunk_size=100,   # tamanho mínimo em caracteres
    max_chunk_size=2000   # tamanho máximo (opcional)
)

# Exemplo de uso
texto = """# Meu documento
## Introdução
Aqui vai o texto de introdução...
## Machine Learning
Conceitos de ML...
"""

chunks = splitter.split_text(texto)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk[:100]}...")
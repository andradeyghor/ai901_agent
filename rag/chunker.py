"""
Script de chunking para preparar os arquivos .md para RAG.

Estratégia:
- Cada arquivo .md é dividido em seções pelos cabeçalhos ## e ###.
- Texto antes do primeiro cabeçalho vira uma seção própria chamada "Introdução".
- Cada seção é sub-dividida em chunks de ~TARGET_TOKENS tokens, agrupando
  parágrafos inteiros (nunca corta um parágrafo ou item de lista ao meio).
- Se uma seção sozinha já for menor que TARGET_TOKENS, ela vira um chunk único.
- Aplica overlap entre chunks consecutivos DENTRO da mesma seção (repete o
  último parágrafo do chunk anterior no início do próximo).
- Contagem de tokens feita com tiktoken (encoding cl100k_base).

Saída: um único arquivo JSON (chunks.json) com uma lista de objetos:
    {
        "id": "topico__arquivo__0001",
        "topico": "nome-da-subpasta",
        "arquivo": "nome_do_arquivo.md",
        "secao": "Texto do cabeçalho (ou 'Introdução')",
        "nivel_secao": 2 | 3 | 0,   # 0 = introdução sem cabeçalho
        "texto": "conteúdo do chunk",
        "num_tokens": 342
    }

Uso:
    python chunker.py --docs-dir docs --output chunks.json
"""

import argparse
import json
import re
from pathlib import Path

import tiktoken

TARGET_TOKENS = 450        # tamanho-alvo por chunk (dentro da faixa 300-500)
MAX_TOKENS = 600           # limite antes de forçar corte mesmo no meio de um parágrafo grande
OVERLAP_PARAGRAPHS = 1     # quantos parágrafos do fim do chunk anterior repetir no início do próximo

ENCODING = tiktoken.get_encoding("cl100k_base")

# Detecta linhas de cabeçalho ## ou ### (não #### em diante, não # de nível 1)
HEADER_RE = re.compile(r"^(#{2,3})\s+(.*)$", re.MULTILINE)


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def split_into_sections(md_text: str):
    """
    Divide o markdown em seções usando ## e ### como separadores.
    Retorna uma lista de dicts: {"titulo": str, "nivel": int, "corpo": str}
    O texto antes do primeiro cabeçalho (se houver) vira a seção "Introdução" (nivel 0).
    """
    matches = list(HEADER_RE.finditer(md_text))
    sections = []

    # Texto antes do primeiro cabeçalho
    first_start = matches[0].start() if matches else len(md_text)
    intro_text = md_text[:first_start].strip()
    if intro_text:
        sections.append({"titulo": "Introdução", "nivel": 0, "corpo": intro_text})

    for i, m in enumerate(matches):
        nivel = len(m.group(1))  # 2 ou 3
        titulo = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        corpo = md_text[body_start:body_end].strip()
        sections.append({"titulo": titulo, "nivel": nivel, "corpo": corpo})

    return sections


def split_into_blocks(corpo: str):
    """
    Divide o corpo de uma seção em blocos por linha em branco (parágrafos).
    Blocos de lista (linhas consecutivas começando com '-', '*' ou dígito+'.')
    são agrupados como um único bloco, para nunca serem cortados ao meio.
    """
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", corpo) if p.strip()]

    blocks = []
    buffer_list = []

    def flush_list():
        if buffer_list:
            blocks.append("\n".join(buffer_list))
            buffer_list.clear()

    for p in raw_paragraphs:
        lines = p.split("\n")
        is_list_paragraph = all(
            re.match(r"^\s*([-*]|\d+\.)\s+", line) for line in lines if line.strip()
        )
        if is_list_paragraph:
            buffer_list.append(p)
        else:
            flush_list()
            blocks.append(p)
    flush_list()

    return blocks


def group_blocks_into_chunks(blocks):
    """
    Agrupa blocos (parágrafos/listas) sequenciais até atingir ~TARGET_TOKENS,
    aplicando overlap de OVERLAP_PARAGRAPHS blocos entre chunks consecutivos.
    Retorna lista de strings (um texto por chunk).
    """
    chunks = []
    current_blocks = []
    current_tokens = 0

    def flush_chunk():
        if current_blocks:
            chunks.append("\n\n".join(current_blocks))

    for block in blocks:
        block_tokens = count_tokens(block)

        # Bloco sozinho já excede o limite máximo: vira chunk próprio (sem tentar dividir).
        if block_tokens >= MAX_TOKENS:
            flush_chunk()
            chunks.append(block)
            current_blocks = []
            current_tokens = 0
            continue

        if current_tokens + block_tokens > TARGET_TOKENS and current_blocks:
            flush_chunk()
            # overlap: mantém os últimos N blocos do chunk anterior
            overlap_blocks = current_blocks[-OVERLAP_PARAGRAPHS:] if OVERLAP_PARAGRAPHS > 0 else []
            current_blocks = overlap_blocks.copy()
            current_tokens = sum(count_tokens(b) for b in current_blocks)

        current_blocks.append(block)
        current_tokens += block_tokens

    flush_chunk()
    return chunks


def process_file(filepath: Path, topico: str):
    md_text = filepath.read_text(encoding="utf-8")
    sections = split_into_sections(md_text)

    chunk_records = []
    seq = 0
    for section in sections:
        blocks = split_into_blocks(section["corpo"])
        if not blocks:
            continue
        section_chunks = group_blocks_into_chunks(blocks)

        for texto_chunk in section_chunks:
            seq += 1
            chunk_records.append({
                "id": f"{topico}__{filepath.stem}__{seq:04d}",
                "topico": topico,
                "arquivo": filepath.name,
                "secao": section["titulo"],
                "nivel_secao": section["nivel"],
                "texto": texto_chunk,
                "num_tokens": count_tokens(texto_chunk),
            })

    return chunk_records


def main():
    parser = argparse.ArgumentParser(description="Chunking de arquivos .md para RAG")
    parser.add_argument("--docs-dir", default="docs", help="Pasta raiz contendo subpastas por tópico")
    parser.add_argument("--output", default="chunks.json", help="Arquivo JSON de saída")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        raise SystemExit(f"Pasta não encontrada: {docs_dir}")

    all_chunks = []
    md_files = sorted(docs_dir.rglob("*.md"))

    if not md_files:
        raise SystemExit(f"Nenhum arquivo .md encontrado em {docs_dir}")

    for filepath in md_files:
        # tópico = nome da subpasta imediatamente dentro de docs/
        try:
            topico = filepath.relative_to(docs_dir).parts[0]
        except IndexError:
            topico = "sem_topico"

        chunks = process_file(filepath, topico)
        all_chunks.extend(chunks)
        print(f"  {filepath.relative_to(docs_dir)}: {len(chunks)} chunks")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    total_tokens = sum(c["num_tokens"] for c in all_chunks)
    print(f"\nTotal: {len(all_chunks)} chunks de {len(md_files)} arquivos")
    print(f"Total de tokens: {total_tokens}")
    print(f"Média de tokens por chunk: {total_tokens / len(all_chunks):.0f}")
    print(f"Salvo em: {args.output}")


if __name__ == "__main__":
    main()
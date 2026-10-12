"""RAG sobre um PDF com LangChain 1.x, Chroma e LLM gratuito (Gemini por padrão).

Uso:
    python app.py documento.pdf                      # modo interativo
    python app.py documento.pdf -q "Sua pergunta"    # pergunta única
    python app.py documento.pdf --reindex            # recria o índice do zero

Provedores de LLM (variável LLM_PROVIDER no .env):
    gemini      -> Google AI Studio (padrão, precisa de GOOGLE_API_KEY)
    openrouter  -> modelos ":free" (precisa de OPENROUTER_API_KEY)
    ollama      -> modelo local, sem chave (precisa do Ollama instalado)
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------------------------------------------------------------
# Configurações (podem ser sobrescritas por variáveis de ambiente / .env)
# ---------------------------------------------------------------------------
DEFAULT_PDF = "1758887557527-attachment.pdf"
PERSIST_DIR = "text_index"

# Embeddings rodam localmente (modelo público, sem token). Multilíngue, pois o
# bge-small-en é só inglês. Para mais qualidade e mais peso: "BAAI/bge-m3".
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # "cuda" se tiver GPU Nvidia

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
# Confira os nomes vigentes nos painéis de cada serviço, pois eles mudam.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

SYSTEM_PROMPT = (
    "Você é um assistente que responde perguntas sobre um documento. "
    "Use apenas os trechos de contexto abaixo para responder, sempre em português. "
    "Se a resposta não estiver no contexto, diga que não sabe e que não pode "
    "criar uma resposta sem contexto.\n\n"
    "Contexto:\n{context}"
)


# ---------------------------------------------------------------------------
# Construção dos componentes
# ---------------------------------------------------------------------------
def require_env(name: str, how_to_get: str) -> str:
    """Lê uma chave do ambiente/.env. Nunca deixe chaves escritas no código."""
    value = os.getenv(name)
    if not value:
        sys.exit(f"Erro: defina {name} no arquivo .env ou no ambiente.\n{how_to_get}")
    return value


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
    )


def build_llm():
    """Cria o chat model conforme LLM_PROVIDER. Imports sob demanda."""
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = require_env(
            "GOOGLE_API_KEY",
            "Gere a chave em aistudio.google.com (opção 'Get API key').",
        )
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=key,
            temperature=0.1,
            max_output_tokens=1024,
        )

    if LLM_PROVIDER == "openrouter":
        from langchain_openai import ChatOpenAI

        key = require_env(
            "OPENROUTER_API_KEY",
            "Gere a chave em openrouter.ai (Keys). Use modelos com sufixo ':free'.",
        )
        return ChatOpenAI(
            model=OPENROUTER_MODEL,
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1,
            max_tokens=1024,
        )

    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )

    sys.exit(
        f"Erro: LLM_PROVIDER '{LLM_PROVIDER}' inválido. "
        "Use 'gemini', 'openrouter' ou 'ollama'."
    )


def collection_name_for(pdf_path: Path) -> str:
    """Uma coleção por PDF, para não misturar documentos diferentes no índice."""
    digest = hashlib.md5(pdf_path.name.encode("utf-8")).hexdigest()[:16]
    return f"pdf_{digest}"


def load_and_split(pdf_path: Path) -> list:
    print("Processando e fatiando o documento PDF...")
    pages = PyPDFLoader(str(pdf_path)).load()

    if not any(page.page_content.strip() for page in pages):
        sys.exit(
            "Erro: nenhum texto foi extraído do PDF. Ele pode ser escaneado "
            "(imagens), o que exigiria OCR."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(pages)


def get_vector_store(pdf_path: Path, embeddings: HuggingFaceEmbeddings, reindex: bool) -> Chroma:
    """Reaproveita o índice existente; só indexa se estiver vazio ou com --reindex."""
    collection = collection_name_for(pdf_path)
    db = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    if reindex:
        print("Recriando o índice do zero...")
        db.delete_collection()
        db = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )

    if db.get(limit=1)["ids"]:
        print("Índice existente encontrado, reaproveitando (use --reindex para recriar).")
        return db

    chunks = load_and_split(pdf_path)
    print(f"Indexando {len(chunks)} fragmentos no Chroma...")
    db.add_documents(chunks)
    return db


def build_rag_chain(db: Chroma, llm):
    retriever = db.as_retriever(
        search_type="mmr",  # reduz trechos redundantes entre os resultados
        search_kwargs={"k": TOP_K, "fetch_k": 20},
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def ask(rag_chain, question: str) -> None:
    print(f"\nPergunta: {question}")
    try:
        result = rag_chain.invoke({"input": question})
    except Exception as e:
        print(f"\nErro ao invocar o LLM ({LLM_PROVIDER}): {e}")
        print(
            "Verifique: se a chave está correta, se o nome do modelo ainda existe "
            "no provedor e se você não estourou a cota gratuita (erro 429 = "
            "aguarde um pouco e tente de novo)."
        )
        return

    print("\nResposta:")
    print(result["answer"])

    pages = sorted({doc.metadata.get("page", -1) + 1 for doc in result["context"]})
    print(f"\nTrechos usados (páginas): {', '.join(map(str, pages))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perguntas e respostas sobre um PDF (RAG).")
    parser.add_argument("pdf", nargs="?", default=DEFAULT_PDF, help="caminho do PDF")
    parser.add_argument("-q", "--question", help="faz uma única pergunta e encerra")
    parser.add_argument("--reindex", action="store_true", help="recria o índice vetorial")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        sys.exit(f"Erro: o arquivo '{pdf_path}' não foi encontrado.")

    print(f"Carregando modelos de IA (LLM: {LLM_PROVIDER})...")
    llm = build_llm()  # valida a chave antes de baixar o modelo de embeddings
    embeddings = build_embeddings()

    db = get_vector_store(pdf_path, embeddings, args.reindex)
    rag_chain = build_rag_chain(db, llm)

    if args.question:
        ask(rag_chain, args.question)
        return

    print("\nPronto. Digite sua pergunta (ou 'sair' para encerrar).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"sair", "exit", "quit"}:
            break
        if question:
            ask(rag_chain, question)


if __name__ == "__main__":
    main()

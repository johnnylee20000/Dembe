"""Trinidad and Tobago legal RAG agent package."""

from .chunker import LegalChunk, load_markdown_chunks, split_markdown_into_segments
from .rag_pipeline import RAGAnswer, TTRAGPipeline
from .tools import GazetteResult, summarize_document, web_search_gazette
from .vector_store import RetrievedChunk, TTVectorStore

__all__ = [
    "LegalChunk",
    "load_markdown_chunks",
    "split_markdown_into_segments",
    "RetrievedChunk",
    "TTVectorStore",
    "RAGAnswer",
    "TTRAGPipeline",
    "GazetteResult",
    "summarize_document",
    "web_search_gazette",
]


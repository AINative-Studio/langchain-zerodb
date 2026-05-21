"""ZeroDB vector store for LangChain."""
from __future__ import annotations

import uuid
from typing import Any, Iterable, List, Optional, Tuple, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class ZeroDBVectorStore(VectorStore):
    """LangChain VectorStore backed by ZeroDB.

    Usage::

        from langchain_zerodb import ZeroDBVectorStore
        from langchain_openai import OpenAIEmbeddings

        store = ZeroDBVectorStore(
            api_key="your-zerodb-api-key",
            namespace="my-namespace",
            embedding=OpenAIEmbeddings(),
        )
        store.add_texts(["hello world", "goodbye world"])
        docs = store.similarity_search("hello", k=2)
    """

    def __init__(
        self,
        api_key: str,
        namespace: str,
        embedding: Embeddings,
        base_url: str = "https://api.ainative.studio",
    ) -> None:
        self._api_key = api_key
        self._namespace = namespace
        self._embedding = embedding
        self._base_url = base_url.rstrip("/")

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        import requests

        texts = list(texts)
        vectors = self._embedding.embed_documents(texts)
        doc_ids = ids or [str(uuid.uuid4()) for _ in texts]
        metas = metadatas or [{} for _ in texts]

        for doc_id, text, vector, meta in zip(doc_ids, texts, vectors, metas):
            requests.post(
                f"{self._base_url}/api/v1/public/vectors/upsert",
                headers=self._headers(),
                json={
                    "id": doc_id,
                    "namespace": self._namespace,
                    "content": text,
                    "vector": vector,
                    "metadata": meta,
                },
            ).raise_for_status()

        return doc_ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        import requests

        vector = self._embedding.embed_query(query)
        resp = requests.post(
            f"{self._base_url}/api/v1/public/vectors/search",
            headers=self._headers(),
            json={"namespace": self._namespace, "vector": vector, "top_k": k},
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])

        docs_and_scores = []
        for r in results:
            doc = Document(
                page_content=r.get("content", ""),
                metadata={**r.get("metadata", {}), "id": r.get("id")},
            )
            docs_and_scores.append((doc, r.get("score", 0.0)))
        return docs_and_scores

    @classmethod
    def from_texts(
        cls: Type["ZeroDBVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        api_key: str = "",
        namespace: str = "default",
        **kwargs: Any,
    ) -> "ZeroDBVectorStore":
        store = cls(api_key=api_key, namespace=namespace, embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas)
        return store

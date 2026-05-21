# langchain-zerodb

LangChain vector store integration for [ZeroDB](https://ainative.studio/products/zerodb) — free embeddings, sub-millisecond semantic search.

## Install

```bash
pip install langchain-zerodb
```

## Quick Start

```python
from langchain_zerodb import ZeroDBVectorStore
from langchain_openai import OpenAIEmbeddings

store = ZeroDBVectorStore(
    api_key="your-zerodb-api-key",
    namespace="my-docs",
    embedding=OpenAIEmbeddings(),
)

store.add_texts(["AINative builds AI-native infrastructure"])
docs = store.similarity_search("AI infrastructure", k=3)
```

## License

MIT — [AINative Studio](https://ainative.studio)

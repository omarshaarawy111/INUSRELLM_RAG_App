# 🏢 Insurellm Expert Assistant - RAG Application

A production-ready Retrieval-Augmented Generation (RAG) application built from scratch without LangChain, using Chroma vector database, OpenAI embeddings, and LiteLLM for LLM abstraction.

## 📋 Features

- **Query Rewriting**: Intelligently rewrites user queries for better retrieval
- **Vector Retrieval**: Uses OpenAI embeddings to retrieve relevant documents
- **Chunk Merging**: Combines results from original and rewritten queries
- **Reranking**: LLM-based reranking to improve relevance ordering
- **Gradio UI**: User-friendly chat interface
- **Error Handling**: Exponential backoff retry logic with tenacity decorator
- **Parallel Processing**: Multiprocessing support for faster chunking

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- Gradio
- Chroma DB
- LiteLLM

### Installation

1. **Clone/Setup the project**
```bash
cd RAG_App_From_Scratch
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install openai gradio chromadb litellm tenacity python-dotenv pydantic tqdm
```

4. **Configure environment variables**
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
```

5. **Ingest documents** (one-time setup)
```bash
python ingest.py
```
This will process all markdown files from the `knowledge-base/` directory and create vector embeddings.

6. **Run the application**
```bash
python app.py
```
The application will open in your browser at `http://localhost:7860`

## 📁 Project Structure

```
RAG_App_From_Scratch/
├── app.py              # Gradio UI application
├── answer.py           # Core RAG logic and LLM interface
├── ingest.py           # Document ingestion and embedding pipeline
├── evaluator.py        # Evaluation metrics
├── readme.md           # This file
├── knowledge-base/     # Source documents (markdown files)
│   ├── company/        # Company information
│   ├── contracts/      # Contract documents
│   ├── employees/      # Employee information
│   └── products/       # Product information
├── preprocessed_db/    # Chroma vector database (auto-created)
├── evaluation/         # Evaluation scripts and test data
│   ├── eval.py
│   ├── test.py
│   └── tests.jsonl
```

## 🔄 RAG Pipeline Flow

```
User Query
    ↓
Query Rewriting (LLM rewrites for better retrieval)
    ↓
Query Embedding (Convert to vector using OpenAI)
    ↓
Retrieval (Fetch top-K relevant chunks from Chroma DB)
    ↓
Chunk Merging (Combine results from original + rewritten queries)
    ↓
Reranking (LLM reranks chunks by relevance)
    ↓
Prompt Construction (Format retrieved context)
    ↓
LLM Response (Generate answer using GPT)
    ↓
Return Answer + Context to User
```

## 🔧 Configuration

Edit these constants in `answer.py` or `ingest.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_NAME` | `preprocessed_db` | Path to Chroma database |
| `collection_name` | `docs` | Chroma collection name |
| `embedding_model` | `text-embedding-3-large` | OpenAI embedding model |
| `GENERATOR_MODEL` | `gpt-4.1-nano` | OpenAI generation model |
| `RETRIEVAL_K` | `10` | Number of chunks to retrieve |
| `FINAL_K` | `10` | Final number of chunks after reranking |
| `WORKERS` | `3` | Number of parallel processes for ingestion |
| `AVERAGE_CHUNK_SIZE` | `500` | Target characters per chunk |

## 📚 Key Classes and Functions

### answer.py

- **`Result`**: Holds retrieved document chunks with metadata
- **`RankOrder`**: Validation schema for LLM-based reranking
- **`rewrite_query(question, history)`**: Rewrites user query for better retrieval
- **`fetch_context_unranked(question)`**: Retrieves top-K chunks from vector DB
- **`merge_chunks(chunks1, chunks2)`**: Combines chunks from multiple queries
- **`rerank(question, chunks)`**: Reranks chunks by relevance using LLM
- **`fetch_context(original_question)`**: Full context retrieval pipeline
- **`answer_question(question, history)`**: Main RAG function

### app.py

- **`format_context(context)`**: Formats retrieved context for UI display
- **`chat(history)`**: Main chat handler
- **`put_message_in_chatbot(message, history)`**: Adds user message to chat

### ingest.py

- **`fetch_documents()`**: Loads all markdown files from knowledge-base
- **`split_into_chunks(document, chunk_size)`**: Splits documents into chunks
- **`create_chunks_with_llm(raw_chunk)`**: Uses LLM to create structured chunks

## 🎯 How It Works

### Document Ingestion
1. Scans `knowledge-base/` for markdown files
2. Splits documents into chunks (~500 characters each)
3. Uses LLM to generate headlines and summaries for each chunk
4. Stores chunks in Chroma with OpenAI embeddings

### Query Processing
1. **Rewriting**: LLM rewrites the query to be more specific
2. **Embedding**: Both original and rewritten queries are converted to vectors
3. **Retrieval**: Chroma searches for top-10 most similar chunks
4. **Merging**: Combines unique chunks from both query versions
5. **Reranking**: LLM reorders chunks by relevance to the original question
6. **Response**: Top-10 chunks are used to generate the answer

### Error Handling
- Uses `@retry` decorator with exponential backoff (10-240 seconds)
- Handles OpenAI rate limiting gracefully
- Validates LLM outputs with Pydantic schemas

## 📊 Evaluation

Run evaluation tests:
```bash
python evaluation/eval.py
```

This evaluates:
- Answer accuracy
- Context relevance
- Completeness of responses

## 🐛 Troubleshooting

### Interface stuck after sending message
**Issue**: `collection` not initialized  
**Solution**: Make sure `ingest.py` has been run first to create the vector database

### "OPENAI_API_KEY not found"
**Solution**: Create `.env` file with your OpenAI API key

### Slow response times
**Possible causes**:
- Large knowledge base - increase `RETRIEVAL_K` carefully
- Network latency to OpenAI API
- Try increasing `WORKERS` in ingest.py for parallel processing

### OutOfMemory errors during ingestion
**Solution**: Reduce `WORKERS` in `ingest.py` or increase `AVERAGE_CHUNK_SIZE`

## 📝 Notes

- This implementation avoids LangChain for more control and transparency
- Pydantic is used for schema validation and structured outputs
- LiteLLM provides abstraction over multiple LLM providers
- All API calls use retry logic with exponential backoff

## 🔐 Security

- Never commit `.env` files with API keys
- Use environment variables for sensitive data
- Consider rate limiting in production

## 📄 License

MIT

---

**Built with**: OpenAI, Chroma, Gradio, LiteLLM, Pydantic

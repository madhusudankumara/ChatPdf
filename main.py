import os
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import logging
import google.generativeai as genai

from custom_types import RAGChunkAndSrc, RAGSearchResult
from data_loader import load_and_chunk_pdf, embed_text, EMBED_DIM
from vector_db import QdrantStorage

# Setup logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="chat_pdf",
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        """Load and chunk PDF file"""
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        logger.info(f"Loaded {len(chunks)} chunks from {pdf_path}")
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunk_and_src: RAGChunkAndSrc) -> dict:
        """Embed and upsert chunks to Qdrant"""
        chunks = chunk_and_src.chunks
        source_id = chunk_and_src.source_id
        
        logger.info(f"Starting embedding for {len(chunks)} chunks")
        vecs = embed_text(chunks)
        logger.info(f"Got {len(vecs)} embeddings")
        
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        
        logger.info(f"Upserting {len(ids)} points with {len(vecs)} embeddings")
        
        if len(ids) > 0 and len(vecs) > 0 and len(payloads) > 0:
            # Initialize Qdrant with correct embedding dimension
            qdrant = QdrantStorage(dim=EMBED_DIM)
            qdrant.upsert(ids, vecs, payloads)
            logger.info(f"Successfully upserted {len(chunks)} chunks")
            return {"ingested": len(chunks)}
        else:
            logger.error(f"Empty data: ids={len(ids)}, vecs={len(vecs)}, payloads={len(payloads)}")
            return {"ingested": 0}

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    result = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=dict)
    return result

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        """Search for relevant chunks"""
        query_vec = embed_text([question])[0]
        storage = QdrantStorage(dim=EMBED_DIM)
        found = storage.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely based on the context above. If the answer is not contained within the context, say you don't know."
    )

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # Try to find an available generative model
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        # Fallback: get available models
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if not available_models:
            raise RuntimeError("No available generative models found")
        model_name = available_models[0].split('/')[-1]
        logger.info(f"Using available model: {model_name}")
        model = genai.GenerativeModel(model_name)

    message = model.generate_content(
        [
            "You answer questions using only the provided context.",
            user_content
        ],
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=1024
        )
    )

    answer = message.text.strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

app = FastAPI()

# Search endpoint to query stored chunks and sources
@app.post("/search")
async def search_chunks(query: str, top_k: int = 5):
    """
    Search for relevant chunks based on query text.
    Returns matching chunks and their source IDs.

    Args:
        query: Search query text
        top_k: Number of top results to return

    Returns:
        Dictionary with contexts (chunks) and sources
    """
    try:
        logger.info(f"Searching for: {query}")

        # Embed the query
        query_embedding = embed_text([query])[0]

        # Search in Qdrant
        qdrant = QdrantStorage(dim=EMBED_DIM)
        results = qdrant.search(query_embedding, top_k=top_k)

        logger.info(f"Found {len(results.get('contexts', []))} relevant chunks")
        return results
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return {"error": str(e), "contexts": [], "sources": []}

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf])

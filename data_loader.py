import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import logging
import pypdf
from pdf2image import convert_from_path
import pytesseract

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize Sentence Transformers embedding model (local, no API key needed)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, lightweight embedding model
EMBED_DIM = 384  # Dimension of all-MiniLM-L6-v2 model
logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
embedding_model = SentenceTransformer(EMBED_MODEL_NAME)

# Initialize sentence splitter for chunking PDFs
# Splits text into chunks of 1000 characters with 200 character overlap
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str) -> list[str]:
    """
    Load a PDF file and split its content into chunks.
    Tries multiple extraction methods:
    1. LlamaIndex PDFReader (for text-based PDFs)
    2. pypdf (for text-based PDFs as fallback)
    3. OCR with Tesseract (for image-based PDFs)

    Args:
        path (str): File path to the PDF

    Returns:
        list[str]: List of text chunks from the PDF
    """
    try:
        logger.info(f"Loading PDF from: {path}")

        # Check if file exists
        file_path = Path(path)
        if not file_path.exists():
            logger.error(f"File does not exist: {path}")
            return []

        logger.info(f"File exists. Size: {file_path.stat().st_size} bytes")

        # Try LlamaIndex PDFReader first
        logger.info("Trying LlamaIndex PDFReader...")
        docs = PDFReader().load_data(file=file_path)
        logger.info(f"Loaded {len(docs)} documents from PDF")

        # Extract text from documents
        texts = []
        for i, doc in enumerate(docs):
            text = getattr(doc, "text", None)
            logger.debug(f"Document {i}: has_text={text is not None}, text_length={len(text) if text else 0}")
            if text:
                texts.append(text)

        logger.info(f"Extracted {len(texts)} documents with text from LlamaIndex")

        # If LlamaIndex didn't extract text, try pypdf as fallback
        if not texts:
            logger.warning("LlamaIndex extraction found no text. Trying pypdf fallback...")
            try:
                with open(file_path, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    logger.info(f"pypdf: Found {len(reader.pages)} pages")

                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text and text.strip():
                            texts.append(text)
                            logger.debug(f"Page {page_num}: extracted {len(text)} chars")

                    logger.info(f"pypdf extracted {len(texts)} pages with text")
            except Exception as e:
                logger.error(f"pypdf fallback failed: {str(e)}")

        # If still no text, try OCR for image-based PDFs
        if not texts:
            logger.warning("Text extraction found no content. Attempting OCR on image-based PDF...")
            try:
                logger.info("Converting PDF pages to images...")
                images = convert_from_path(file_path)
                logger.info(f"Converted to {len(images)} images")

                for page_num, image in enumerate(images):
                    logger.debug(f"Running OCR on page {page_num + 1}/{len(images)}...")
                    text = pytesseract.image_to_string(image)
                    if text and text.strip():
                        texts.append(text)
                        logger.debug(f"Page {page_num}: OCR extracted {len(text)} chars")

                logger.info(f"OCR extracted {len(texts)} pages with text")
            except Exception as e:
                logger.error(f"OCR extraction failed: {str(e)}", exc_info=True)
                return []

        if not texts:
            logger.error("No text could be extracted from PDF using any method")
            return []

        # Split texts into smaller chunks for better processing
        chunks = []
        for i, text in enumerate(texts):
            text_chunks = splitter.split_text(text)
            logger.debug(f"Text {i} split into {len(text_chunks)} chunks")
            chunks.extend(text_chunks)

        logger.info(f"Successfully created {len(chunks)} chunks from {path}")
        return chunks
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}", exc_info=True)
        raise

def embed_text(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using Sentence Transformers.
    This is a local embedding model, no API key required.

    Args:
        texts (list[str]): List of text strings to embed

    Returns:
        list[list[float]]: List of embedding vectors for each input text
    """
    if not texts:
        logger.warning("No texts provided for embedding")
        return []

    try:
        logger.info(f"Embedding {len(texts)} texts using {EMBED_MODEL_NAME}")

        # Encode all texts at once (more efficient than one-by-one)
        embeddings_array = embedding_model.encode(texts, convert_to_tensor=False)

        # Convert to list of lists
        embeddings = [embedding.tolist() for embedding in embeddings_array]

        logger.info(f"Successfully embedded {len(embeddings)} texts")
        if embeddings:
            logger.debug(f"Embedding dimension: {len(embeddings[0])}")

        return embeddings
    except Exception as e:
        logger.error(f"Error embedding text: {str(e)}", exc_info=True)
        raise

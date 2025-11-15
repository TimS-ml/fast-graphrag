"""Example usage of GraphRAG with Google Gemini API from AI Studio.

This module demonstrates how to use fast-graphrag with Google's Gemini models accessed through
the Gemini API (AI Studio). It provides a complete example of:
- Using Google Gemini models for language understanding and entity extraction
- Using Gemini embeddings for semantic search
- Processing text documents with multi-encoding support
- Building and querying a knowledge graph with streaming responses
- Interactive query interface for real-time knowledge exploration

Key Features:
    - Async/await pattern for non-blocking API calls
    - Intelligent text file handling with multiple encoding fallbacks (UTF-8, ASCII, ISO-8859-1, CP1252)
    - Custom entity types optimized for story/document analysis
    - High token limits configured for Gemini's generous context window
    - Rate limiting to stay within API quota constraints
    - Interactive streaming query loop for conversational exploration

Setup Instructions:
    1. Install required dependencies:
       pip install fast-graphrag google-generativeai

    2. Obtain a Gemini API key:
       - Visit https://aistudio.google.com
       - Create a new API key in the "API keys" section
       - Add billing to your Google Cloud project for higher rate limits

    3. Set environment variable:
       export GEMINI_API_KEY="your-api-key-here"

    4. Prepare your input file:
       - Place a text file named "book.txt" in the same directory
       - The file should contain the text you want to analyze

    5. Configure domain and entity types (optional):
       - Modify DOMAIN to describe what you're analyzing
       - Adjust ENTITY_TYPES to match your analysis needs

    6. Run the script:
       python gemini_example.py

Rate Limiting Notes:
    - Gemini API has generous rate limits (2000 RPM for 2.0 Flash as of Feb 2025)
    - The example uses conservative rate limits to avoid hitting quota issues
    - For higher throughput, adjust max_requests_per_minute and max_requests_per_second
    - Consider using VoyageAI embeddings for even higher rate limits

Performance Tips:
    - Gemini-2.0-flash provides the best balance of speed and quality
    - The embedding service can handle up to 100 concurrent requests
    - Token limits are configured high (250k+ entities/relations) for comprehensive analysis
"""

import os
import asyncio
from fast_graphrag import GraphRAG, QueryParam
from fast_graphrag._utils import logger
from fast_graphrag._llm import GeminiLLMService, GeminiEmbeddingService

# === Environment and Directory Setup ===
# Create working directory if it doesn't exist - this stores the knowledge graph index and data
WORKING_DIR = "./book_example"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Load Gemini API key from environment variables (set via: export GEMINI_API_KEY="...")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# === Configuration Parameters ===
# DOMAIN: Instructs the LLM on what to focus on when analyzing documents
DOMAIN = "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships."

# EXAMPLE_QUERIES: Sample queries used to optimize entity extraction and knowledge graph structure
# These help the system understand what kinds of questions you'll be asking
EXAMPLE_QUERIES = [
    "What is the significance of Christmas Eve in A Christmas Carol?",
    "How does the setting of Victorian London contribute to the story's themes?",
    "Describe the chain of events that leads to Scrooge's transformation.",
    "How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
    "Why does Dickens choose to divide the story into \"staves\" rather than chapters?"
]

# ENTITY_TYPES: Specifies what types of entities to extract from documents
# Customize these based on your domain (e.g., for medical texts: ["Disease", "Medication", "Symptom"])
ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event"]

# === Optional Alternative Services ===
# Uncomment to use VoyageAI embeddings instead of Gemini (provides higher rate limits)
# from fast_graphrag._llm import VoyageAIEmbeddingService
# VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

# === Optional PDF Processing Support ===
# Uncomment if you want to process PDF files instead of plain text
# Requires: pip install langchain-community pymupdf
#from langchain_community.document_loaders import PyMuPDFLoader
#async def process_pdf(file_path: str) -> str:
#    """Process PDFs with error handling.
#
#    Args:
#        file_path: Path to the PDF file to process
#
#    Returns:
#        str: Extracted text from the PDF
#
#    Raises:
#        FileNotFoundError: If the file doesn't exist
#    """
#    if not os.path.exists(file_path):
#        raise FileNotFoundError(f"PDF file not found: {file_path}")
#
#    try:
#        loader = PyMuPDFLoader(file_path)
#        pages = ""
#        for page in loader.lazy_load():
#            pages += page.page_content
#        return pages
#
#    except Exception as e:
#        raise


async def process_text(file_path: str) -> str:
    """Process text file with robust encoding handling.

    This function handles text files that may be encoded in different character encodings.
    It tries multiple encodings in order of likelihood, falling back if one fails.

    Args:
        file_path: Path to the text file to read

    Returns:
        str: Cleaned and decoded text content

    Raises:
        FileNotFoundError: If the file doesn't exist
        UnicodeError: If no encoding can successfully decode the file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")

    try:
        # Try multiple encodings in order of likelihood
        # UTF-8 is most common, but we fall back to ASCII and other common encodings
        encodings = ['utf-8', 'ascii', 'iso-8859-1', 'cp1252']
        text = None

        # Attempt to decode with each encoding until one succeeds
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                # This encoding failed, try the next one
                continue

        if text is None:
            raise UnicodeError("Failed to decode file with any supported encoding")

        # Clean and normalize text: remove non-ASCII characters and strip whitespace
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text.strip()

    except Exception as e:
        # Log the error with full traceback for debugging
        logger.exception("An error occurred:", exc_info=True)
        raise


async def streaming_query_loop(rag: GraphRAG):
    """Interactive query interface for exploring the knowledge graph.

    Provides an interactive REPL (Read-Eval-Print Loop) where users can ask questions
    about the analyzed documents. Results are streamed in real-time.

    Args:
        rag: Initialized GraphRAG instance with loaded documents
    """
    print("\nStreaming Query Interface (type 'exit' to quit)")
    print("="*50)

    while True:
        try:
            # Get user input
            query = input("\nYou: ").strip()
            if query.lower() == 'exit':
                print("\nExiting chat...")
                break

            print("Assistant: ", end='', flush=True)

            try:
                # Query the knowledge graph with high token limits
                # Gemini has a large context window, so we can use generous limits
                # entities_max_tokens: Maximum tokens for entity context (250k)
                # relations_max_tokens: Maximum tokens for relationship context (200k)
                # chunks_max_tokens: Maximum tokens for document chunks (500k)
                response = await rag.async_query(
                    query,
                    params=QueryParam(
                        with_references=False,  # Don't include reference citations
                        only_context=False,  # Include generated summaries, not just raw context
                        entities_max_tokens=250000,  # High limit for comprehensive entity context
                        relations_max_tokens=200000,  # High limit for relationship context
                        chunks_max_tokens=500000  # High limit for source document chunks
                    )
                )
                # Print the response from the knowledge graph
                print(response.response)
            except Exception as e:
                # Print detailed error information for debugging
                import traceback
                traceback.print_exc()

        except KeyboardInterrupt:
            # Handle user pressing Ctrl+C
            print("\nInterrupted by user")
            break
        except Exception as e:
            # Log unexpected errors but continue the loop
            logger.exception("An error occurred:", exc_info=True)
            continue


async def main():
    """Main function: Initialize GraphRAG and run the interactive query loop.

    This function:
    1. Initializes a GraphRAG instance with Gemini LLM and embedding services
    2. Loads and processes text documents
    3. Builds the knowledge graph from the documents
    4. Starts an interactive query session
    """
    try:
        # Initialize GraphRAG with Gemini services
        grag = GraphRAG(
            working_dir=WORKING_DIR,
            domain=DOMAIN,
            example_queries="\n".join(EXAMPLE_QUERIES),
            entity_types=ENTITY_TYPES,
            config=GraphRAG.Config(
                # LLM Service Configuration
                # Note: Ensure the Generative API is enabled in Google Cloud Console
                # For Vertex AI, pass project_id and location; for AI Studio, use API keys directly
                # Recommendation: Use API keys from https://aistudio.google.com with billing enabled
                # This gives higher rate limits (2000 RPM for Gemini 2.0 Flash as of Feb 2025)
                llm_service=GeminiLLMService(
                    model="gemini-2.0-flash",  # Fast, capable model with great quality/speed tradeoff
                    api_key=GEMINI_API_KEY,  # API key from environment
                    temperature=0.7,  # Balance between creativity (1.0) and determinism (0.0)
                    rate_limit_per_minute=True,  # Enable per-minute rate limiting
                    rate_limit_per_second=True,  # Enable per-second rate limiting
                    max_requests_per_minute=1950,  # Stay under the 2000 RPM limit with buffer
                    max_requests_per_second=500  # Conservative per-second limit
                ),
                # Embedding Service Configuration
                # Converts text to vectors for semantic similarity and retrieval
                embedding_service=GeminiEmbeddingService(
                    api_key=GEMINI_API_KEY,
                    max_requests_concurrent=100,  # Process up to 100 requests in parallel
                ),
                # === Alternative: VoyageAI Embeddings ===
                # Uncomment to use VoyageAI embeddings instead (higher rate limits)
                #embedding_service=VoyageAIEmbeddingService(
                #    model="voyage-3-large",
                #    api_key=VOYAGE_API_KEY,
                #    embedding_dim=1024,  # Dimensionality of voyage-3-large embeddings
                #),
            ),
        )

        # Load and process the input text file
        # Make sure "./book.txt" exists in the current directory
        file_content = await process_text("./book.txt")

        # Insert the document into the knowledge graph
        # This triggers entity/relationship extraction and graph building
        await grag.async_insert(file_content)

        # Start the interactive query loop for user exploration
        await streaming_query_loop(grag)

    except Exception as e:
        # Log any fatal errors that occur during execution
        logger.exception("An error occurred:", exc_info=True)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())

"""Default document chunking service with overlap support.

This module implements a sophisticated text chunking strategy that balances:
- Token size constraints for LLM processing
- Semantic boundaries (paragraphs, sentences) to maintain context
- Overlapping chunks to prevent information loss at boundaries

The chunking process is a critical first step in the GraphRAG pipeline, as chunk
quality directly impacts downstream entity extraction and retrieval accuracy.

Key features:
- Multi-level separators (paragraphs, sentences, punctuation)
- Configurable chunk size and overlap
- Unicode handling and text sanitization
- Content-based hashing for deduplication
- Metadata preservation from source documents
"""

import re
from dataclasses import dataclass, field
from itertools import chain
from typing import Iterable, List, Set, Tuple

import xxhash

from fast_graphrag._types import TChunk, TDocument, THash
from fast_graphrag._utils import TOKEN_TO_CHAR_RATIO

from ._base import BaseChunkingService

# Default separator hierarchy for splitting text at natural boundaries
# Ordered from strongest (paragraph breaks) to weakest (sentence endings)
DEFAULT_SEPARATORS = [
    # Paragraph and page separators
    "\n\n\n",     # Triple newline (page/section break)
    "\n\n",       # Double newline (paragraph break)
    "\r\n\r\n",   # Windows-style paragraph break
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
]


@dataclass
class DefaultChunkingServiceConfig:
    """Configuration for the default chunking strategy.

    Attributes:
        separators: Hierarchical list of text separators to use for splitting.
            The chunker will try to split on earlier separators first (e.g., paragraphs)
            before falling back to later ones (e.g., sentence endings).
        chunk_token_size: Target size for each chunk in tokens. This is approximate
            since we convert to characters using TOKEN_TO_CHAR_RATIO. Default 800 tokens
            fits comfortably within most LLM context windows while preserving context.
        chunk_token_overlap: Number of overlapping tokens between consecutive chunks.
            This ensures entities/relationships spanning chunk boundaries aren't lost.
            Default 100 tokens provides ~12-15% overlap for 800-token chunks.
    """
    separators: List[str] = field(default_factory=lambda: DEFAULT_SEPARATORS)
    chunk_token_size: int = field(default=800)
    chunk_token_overlap: int = field(default=100)


@dataclass
class DefaultChunkingService(BaseChunkingService[TChunk]):
    """Production-ready text chunking service with overlap and deduplication.

    This implementation provides a robust chunking strategy that:
    1. Respects semantic boundaries (paragraphs, sentences) when possible
    2. Maintains configurable chunk size and overlap
    3. Handles Unicode and special characters gracefully
    4. Deduplicates chunks within and across documents using content hashing
    5. Preserves document metadata in each chunk

    The chunking algorithm works as follows:
    - Split text on hierarchical separators (paragraphs first, then sentences)
    - Merge splits into chunks up to the target size
    - Create overlapping chunks by including content from previous chunks
    - Generate stable hashes for deduplication

    Attributes:
        config: Configuration object controlling chunk size, overlap, and separators.
    """

    config: DefaultChunkingServiceConfig = field(default_factory=DefaultChunkingServiceConfig)

    def __post_init__(self):
        """Initialize the chunker by compiling regex and computing size limits.

        This method:
        - Compiles a regex pattern that matches any configured separator
        - Converts token-based sizes to character counts using TOKEN_TO_CHAR_RATIO
        """
        # Compile regex that matches any separator (escaped for regex safety)
        self._split_re = re.compile(f"({'|'.join(re.escape(s) for s in self.config.separators or [])})")

        # Convert token sizes to approximate character counts
        self._chunk_size = self.config.chunk_token_size * TOKEN_TO_CHAR_RATIO
        self._chunk_overlap = self.config.chunk_token_overlap * TOKEN_TO_CHAR_RATIO

    async def extract(self, data: Iterable[TDocument]) -> Iterable[Iterable[TChunk]]:
        """Extract and deduplicate chunks from documents.

        This method processes each document independently, chunking its content and
        removing duplicates within that document. Duplicates across documents are
        handled later by the state manager's filter_new_chunks method.

        Args:
            data: Iterable of documents to chunk.

        Returns:
            Nested list structure where each inner list contains unique chunks
            from a single document.
        """
        chunks_per_data: List[List[TChunk]] = []

        for d in data:
            # Track chunk IDs within this document to remove duplicates
            unique_chunk_ids: Set[THash] = set()
            extracted_chunks = await self._extract_chunks(d)
            chunks: List[TChunk] = []

            # Deduplicate chunks within this document
            for chunk in extracted_chunks:
                if chunk.id not in unique_chunk_ids:
                    unique_chunk_ids.add(chunk.id)
                    chunks.append(chunk)
            chunks_per_data.append(chunks)

        return chunks_per_data

    async def _extract_chunks(self, data: TDocument) -> List[TChunk]:
        """Extract chunks from a single document with sanitization and hashing.

        This method:
        1. Sanitizes the document text (handles Unicode errors, removes control characters)
        2. Splits the text into chunks if it exceeds the chunk size
        3. Creates TChunk objects with stable content-based hashes

        Args:
            data: The document to extract chunks from.

        Returns:
            List of chunks, each with a unique hash, content, and inherited metadata.
        """
        # Sanitize input data to handle Unicode encoding issues
        try:
            # Re-encode with replacement for invalid characters
            data.data = data.data.encode(errors="replace").decode()
        except UnicodeDecodeError:
            # If that fails, manually replace control characters with spaces
            data.data = re.sub(r"[\x00-\x09\x11-\x12\x14-\x1f]", " ", data.data)

        # If document fits in one chunk, no need to split
        if len(data.data) <= self._chunk_size:
            chunks = [data.data]
        else:
            chunks = self._split_text(data.data)

        # Create TChunk objects with stable hashes for deduplication
        return [
            TChunk(
                # Use xxhash for fast, stable hashing; divide by 2 to get positive integers
                id=THash(xxhash.xxh3_64_intdigest(chunk) // 2),
                content=chunk,
                metadata=data.metadata,  # Preserve metadata from source document
            )
            for chunk in chunks
        ]

    def _split_text(self, text: str) -> List[str]:
        """Split text into segments using configured separators.

        Args:
            text: The text to split.

        Returns:
            List of text segments split on separators and merged to chunk size.
        """
        return self._merge_splits(self._split_re.split(text))

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge split text segments into chunks of appropriate size.

        The regex split creates an alternating list of [content, separator, content, separator, ...].
        This method intelligently merges these segments into chunks that:
        1. Respect the target chunk size
        2. Keep separators with their preceding content (maintaining semantic boundaries)
        3. Reserve space for overlap when building subsequent chunks

        Args:
            splits: Alternating list of content and separator strings from regex split.

        Returns:
            List of merged chunk strings, optionally with overlap.
        """
        if not splits:
            return []

        # Add empty string to ensure last chunk has a separator at the end
        splits.append("")

        merged_splits: List[List[Tuple[str, int]]] = []
        current_chunk: List[Tuple[str, int]] = []
        current_chunk_length: int = 0

        for i, split in enumerate(splits):
            split_length: int = len(split)

            # Odd indices are separators; always include them with their preceding content
            # For content (even indices), check if it fits within the chunk size limit
            # Note: Reserve overlap space for all chunks except the first (i > 0)
            if (i % 2 == 1) or (
                current_chunk_length + split_length <= self._chunk_size - (self._chunk_overlap if i > 0 else 0)
            ):
                # Add to current chunk
                current_chunk.append((split, split_length))
                current_chunk_length += split_length
            else:
                # Current chunk is full, start a new one
                merged_splits.append(current_chunk)
                current_chunk = [(split, split_length)]
                current_chunk_length = split_length

        # Don't forget the last chunk
        merged_splits.append(current_chunk)

        # Apply overlap if configured
        if self._chunk_overlap > 0:
            return self._enforce_overlap(merged_splits)
        else:
            # No overlap: just concatenate the text segments
            r = ["".join((c[0] for c in chunk)) for chunk in merged_splits]

        return r

    def _enforce_overlap(self, chunks: List[List[Tuple[str, int]]]) -> List[str]:
        """Add overlapping content from previous chunks to ensure continuity.

        This method prepends text from the end of chunk N-1 to the beginning of chunk N,
        ensuring that entities or relationships spanning chunk boundaries are captured
        in at least one complete chunk.

        Algorithm:
        - First chunk: return as-is (no previous chunk to overlap with)
        - Subsequent chunks: prepend up to chunk_overlap characters from the end
          of the previous chunk

        Args:
            chunks: List of chunks, where each chunk is a list of (text, length) tuples.

        Returns:
            List of chunk strings with overlap applied.
        """
        result: List[str] = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk has no predecessor, just join it
                result.append("".join((c[0] for c in chunk)))
            else:
                # For subsequent chunks, prepend overlap from previous chunk
                overlap_length: int = 0
                overlap: List[str] = []

                # Walk backwards through previous chunk until we have enough overlap
                for text, length in reversed(chunks[i - 1]):
                    if overlap_length + length > self._chunk_overlap:
                        break
                    overlap_length += length
                    overlap.append(text)

                # Concatenate: reversed overlap (now in correct order) + current chunk
                result.append("".join(chain(reversed(overlap), (c[0] for c in chunk))))
        return result

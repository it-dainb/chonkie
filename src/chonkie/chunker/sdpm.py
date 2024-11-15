from typing import Any, List, Union, Optional
import warnings

from .semantic import SemanticChunk, SemanticChunker


class SDPMChunker(SemanticChunker):
    def __init__(
        self,
        tokenizer: Union[str, Any] = "gpt2",
        embedding_model: Union[str, Any] = "sentence-transformers/all-MiniLM-L6-v2",
        appending_threshold: float = None,
        merging_threshold: float = None,
        appending_percentile: float = None,
        chunk_size: int = 512,
        min_sentences_per_chunk: int = 1,
        sentence_mode: str = "heuristic",
        spacy_model: str = "en_core_web_sm",
        skip_window: int = 1,  # How many chunks to skip when looking for similarities
        word_segmentation: bool = False,
        sentence_transformers_kargs: dict = {},
    ):
        """Initialize the SDPMChunker.

        Args:
            Same as SemanticChunker, plus:
            merging_threshold: Threshold for merging similar groups
            skip_window: Number of chunks to skip when looking for similarities
        """
        super().__init__(
            tokenizer=tokenizer,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            appending_threshold=appending_threshold,
            appending_percentile=appending_percentile,
            min_sentences_per_chunk=min_sentences_per_chunk,
            sentence_mode=sentence_mode,
            spacy_model=spacy_model,
            word_segmentation=word_segmentation,
            sentence_transformers_kargs=sentence_transformers_kargs,
        )

        if skip_window < 0:
            raise ValueError("Skip window must be >= 0")
        
        self.skip_window = skip_window
        self.merging_threshold = merging_threshold

        if self.merging_threshold is None:
            self.merging_threshold = self.similarity_threshold
            warnings.warn(
                "Merging threshold not specified, using appending threshold instead"
            )

    def _merge_chunks(self, chunks: List[SemanticChunk]) -> SemanticChunk:
        """Merge the groups together"""
        merged_sents = []
        for chunk in chunks:
            merged_sents.extend(chunk.sentences)
            
        return self._create_chunk(merged_sents)

    def _skip_and_merge(
        self, chunks: List[SemanticChunk], id: str
    ) -> List[SemanticChunk]:
        """Merge similar groups considering skip window."""
        if len(chunks) <= 1:
            return chunks

        merged_chunks = []
        embeddings = [self._compute_chunk_embedding(chunk) for chunk in chunks]

        while chunks:
            if len(chunks) == 1:
                merged_chunks.append(
                    self._create_chunk(chunks[0].sentences, len(merged_chunks), id)
                )
                break

            # Calculate skip index ensuring it's valid
            skip_index = min(self.skip_window + 1, len(chunks) - 1)

            # Compare current group with skipped group
            similarity = self._get_semantic_similarity(
                embeddings[0], embeddings[skip_index]
            )

            if similarity >= self.merging_threshold:
                # Merge groups from 0 to skip_index (inclusive)
                merged = self._merge_chunks(chunks[: skip_index + 1])

                # Remove the merged groups
                for _ in range(skip_index + 1):
                    chunks.pop(0)
                    embeddings.pop(0)

                # Add merged group back at the start
                chunks.insert(0, merged)
                embeddings.insert(0, self._compute_chunk_embedding(merged))
            else:
                # No merge possible, move first group to results
                merged_chunks.append(
                    self._create_chunk(chunks.pop(0).sentences, len(merged_chunks), id)
                )
                embeddings.pop(0)

        return merged_chunks

    def chunk(self, text: str, id: str) -> List[SemanticChunk]:
        """Split text into chunks using the SPDM approach.

        Args:
            text: Input text to be chunked

        Returns:
            List of SemanticChunk objects
        """
        if not text.strip():
            return []

        # Prepare sentences with precomputed information
        sentences = self._prepare_sentences(text)
        if len(sentences) < self.min_sentences_per_chunk:
            return [self._create_chunk(sentences)]

        # First pass: Group sentences by semantic similarity
        initial_chunks = self._chunk_sentences(sentences, id)

        # Second pass: Merge similar groups with skip window
        merged_chunks = self._skip_and_merge(initial_chunks, id)

        # Final pass: Split into size-appropriate chunks
        chunks = self._split_chunks(merged_chunks, id)

        return chunks

    def __repr__(self) -> str:
        threshold_info = (
            f"similarity_threshold={self.similarity_threshold}"
            if self.similarity_threshold is not None
            else f"similarity_percentile={self.similarity_percentile}"
        )
        return (
            f"SPDMChunker(chunk_size={self.chunk_size}, "
            f"lang='{self.lang}', "
            f"{threshold_info}, "
            f"initial_sentences={self.initial_sentences}, "
            f"sentence_mode='{self.sentence_mode}', "
            f"skip_window={self.skip_window})"
        )

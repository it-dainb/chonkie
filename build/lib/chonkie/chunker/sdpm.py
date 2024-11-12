from typing import Any, List, Union
import warnings

from .semantic import SemanticChunk, SemanticChunker, Sentence


class SDPMChunker(SemanticChunker):
    def __init__(
        self,
        tokenizer: Union[str, Any] = "gpt2",
        language: str = "auto",
        embedding_model: Union[str, Any] = "sentence-transformers/all-MiniLM-L6-v2",
        appending_threshold: float = None,
        merging_threshold: float = None,
        appending_percentile: float = None,
        max_chunk_size: int = 512,
        initial_sentences: int = 1,
        sentence_mode: str = "heuristic",
        spacy_model: str = "en_core_web_sm",
        skip_window: int = 1,  # How many chunks to skip when looking for similarities
    ):
        """Initialize the SDPMChunker.

        Args:
            Same as SemanticChunker, plus:
            appending_threshold: Threshold for appending similar groups
            appending_percentile: Percentile threshold for appending similar groups
            merging_threshold: Threshold for merging similar groups
            skip_window: Number of chunks to skip when looking for similarities
        """
        super().__init__(
            tokenizer=tokenizer,
            language=language,
            embedding_model=embedding_model,
            max_chunk_size=max_chunk_size,
            similarity_threshold=appending_threshold,
            similarity_percentile=appending_percentile,
            initial_sentences=initial_sentences,
            sentence_mode=sentence_mode,
            spacy_model=spacy_model,
        )
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
        merged_scores = []
        for chunk in chunks:
            merged_sents.extend(chunk.sentences)
            merged_scores.extend(chunk.scores)
            
        return self._create_chunk(merged_sents, merged_scores)

    def _skip_and_merge(
        self, chunks: List[SemanticChunk]
    ) -> List[SemanticChunk]:
        """Merge similar groups considering skip window."""
        if len(chunks) <= 1:
            return chunks

        merged_chunks = []
        embeddings = [self._compute_chunk_embedding(chunk) for chunk in chunks]

        while chunks:
            if len(chunks) == 1:
                merged_chunks.append(chunks[0])
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
                merged_chunks.append(chunks.pop(0))
                embeddings.pop(0)

        return merged_chunks

    def chunk(self, text: str) -> List[SemanticChunk]:
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
        if len(sentences) < self.initial_sentences:
            return [self._create_chunk(sentences)]

        # First pass: Group sentences by semantic similarity
        initial_chunks = self._chunk_sentences(sentences)

        # Second pass: Merge similar groups with skip window
        merged_chunks = self._skip_and_merge(initial_chunks, self.similarity_threshold)

        # Final pass: Split into size-appropriate chunks
        chunks = self._split_chunks(merged_chunks)

        return chunks

    def __repr__(self) -> str:
        threshold_info = (
            f"similarity_threshold={self.similarity_threshold}"
            if self.similarity_threshold is not None
            else f"similarity_percentile={self.similarity_percentile}"
        )
        return (
            f"SPDMChunker(max_chunk_size={self.max_chunk_size}, "
            f"lang='{self.lang}', "
            f"{threshold_info}, "
            f"initial_sentences={self.initial_sentences}, "
            f"sentence_mode='{self.sentence_mode}', "
            f"skip_window={self.skip_window})"
        )

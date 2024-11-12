import importlib.util
import re
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np

from .base import BaseChunker
from .sentence import Sentence, SentenceChunk, SentenceChunker

@dataclass
class SemanticSentence(Sentence):
    embedding: Optional[np.ndarray] = None

@dataclass
class SemanticChunk(SentenceChunk):
    sentences: List[SemanticSentence] = None
    scores: list[float] = None
    total_score: float = None


class SemanticChunker(SentenceChunker):
    def __init__(
        self,
        tokenizer: Union[str, Any] = "gpt2",
        language: str = "auto",
        chunk_size: int = 512,
        min_sentences_per_chunk: int = 1,
        sentence_mode: str = "heuristic",
        spacy_model: str = "en_core_web_sm",
        embedding_model: Union[str, Any] = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: Optional[float] = None,
        similarity_percentile: Optional[float] = None,
    ):
        """Initialize the SemanticChunker.

        Args:
            tokenizer: Tokenizer for counting tokens
            language: Language of the Chunker
            embedding_model: Name of the sentence-transformers model to load
            chunk_size: Maximum tokens allowed per chunk
            similarity_threshold: Absolute threshold for semantic similarity (0-1)
            similarity_percentile: Percentile threshold for similarity (0-100)
            min_sentences_per_chunk: Number of sentences to start each chunk with
            sentence_mode: "heuristic" or "spacy" for sentence splitting
            spacy_model: Name of spaCy model to use if sentence_mode="spacy"

        Raises:
            ValueError: If parameters are invalid
            ImportError: If required dependencies aren't installed
        """
        super().__init__(
            tokenizer = tokenizer,
            chunk_size = chunk_size,
            chunk_overlap = 1,
            mode = sentence_mode,
            min_sentences_per_chunk = min_sentences_per_chunk,
            spacy_model = spacy_model,
        )

        if similarity_threshold is not None and (
            similarity_threshold < 0 or similarity_threshold > 1
        ):
            raise ValueError("similarity_threshold must be between 0 and 1")
        if similarity_percentile is not None and (
            similarity_percentile < 0 or similarity_percentile > 100
        ):
            raise ValueError("similarity_percentile must be between 0 and 100")
        if similarity_threshold is not None and similarity_percentile is not None:
            raise ValueError(
                "Cannot specify both similarity_threshold and similarity_percentile"
            )
        if similarity_threshold is None and similarity_percentile is None:
            raise ValueError(
                "Must specify either similarity_threshold or similarity_percentile"
            )

        self.language = language
        self.similarity_threshold = similarity_threshold
        self.similarity_percentile = similarity_percentile

        # Load sentence-transformers model
        self._import_sentence_transformers()
        if isinstance(embedding_model, str):
            self.embedding_model = self._load_sentence_transformer_model(
                embedding_model
            )
        else:
            self.embedding_model = embedding_model

        # Fix chunk_size to fit embedding model
        if self.embedding_model.max_seq_length < self.chunk_size:
            self.chunk_size = self.embedding_model.max_seq_length
            warnings.warn(
                f"Model '{embedding_model}' has a max_seq_length of {self.embedding_model.max_seq_length}, "
                f"which is smaller than chunk_size of {self.chunk_size}. "
                f"Setting chunk_size to {self.embedding_model.max_seq_length}."
            )

    def _import_sentence_transformers(self) -> Any:
        """Import sentence-transformers library. Imports mentioned inside the class,
        because it takes too long to import the whole library at the beginning of the file.
        """
        # Check if sentence-transformers is available
        SENTENCE_TRANSFORMERS_AVAILABLE = (
            importlib.util.find_spec("sentence_transformers") is not None
        )
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                global SentenceTransformer
                from sentence_transformers import SentenceTransformer
            except ImportError:
                SENTENCE_TRANSFORMERS_AVAILABLE = False
                warnings.warn(
                    "Failed to import sentence-transformers despite it being installed. SemanticChunker will not work."
                )
        else:
            warnings.warn(
                "sentence-transformers is not installed. SemanticChunker will not work."
            )

    def _load_sentence_transformer_model(self, model_name: str) -> Any:
        """Load a sentence-transformers model by name."""
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            raise ImportError(
                f"Failed to load sentence-transformers model '{model_name}'. "
                f"Make sure it is installed and available."
            ) from e
        return model

    def _compute_similarity_threshold(self, all_similarities: List[float]) -> float:
        """Compute similarity threshold based on percentile if specified."""
        if self.similarity_threshold is not None:
            return self.similarity_threshold
        else:
            return float(np.percentile(all_similarities, self.similarity_percentile))

    def _prepare_sentences(self, text: str) -> List[SemanticSentence]:
        """Prepare sentences with precomputed information.

        Args:
            text: Input text to be processed

        Returns:
            List of Sentence objects with precomputed token counts and embeddings
        """
        if not text.strip():
            return []

        # Split text into sentences
        raw_sentences = self._split_into_sentences(text)

        embeddings = self.embedding_model.encode(
            [sent.text for sent in raw_sentences], 
            convert_to_numpy=True
        )

        # Create Sentence objects with all precomputed information
        sentences = []
        for sent, embedding in zip(raw_sentences, embeddings):
            sentences.append(SemanticSentence(
                text=sent.text,
                start_index=sent.start_index,
                end_index=sent.end_index,
                token_count=sent.token_count,
                embedding=embedding,
            ))

        return sentences

    def _get_semantic_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        similarity = self.embedding_model.similarity(embedding1, embedding2)
        return similarity

    def _compute_chunk_embedding(self, chunk: SemanticChunk) -> np.ndarray:
        """Compute mean embedding for a group of sentences."""
        return np.divide(
            np.sum([(sent.embedding * sent.token_count) for sent in chunk.sentences], axis=0),
            np.sum([sent.token_count for sent in chunk.sentences]),
            dtype=np.float32,
        )

    def _chunk_sentences(self, sentences: List[SemanticSentence]) -> List[SemanticChunk]:
        """Group sentences based on semantic similarity, ignoring token count limits.

        Args:
            sentences: List of Sentence objects with precomputed embeddings

        Returns:
            List of sentence groups, where each group is semantically coherent
        """
        if len(sentences) <= self.min_sentences_per_chunk:
            return [sentences]

        # Get or compute similarity threshold
        if self.similarity_percentile is not None:
            # Compute all pairwise similarities
            all_similarities = [
                self._get_semantic_similarity(
                    sentences[i].embedding, sentences[i + 1].embedding
                )
                for i in range(len(sentences) - 1)
            ]
            similarity_threshold = float(
                np.percentile(all_similarities, self.similarity_percentile)
            )
        else:
            similarity_threshold = self.similarity_threshold

        chunks = []

        initial_sentences = sentences[: self.min_sentences_per_chunk]
        current_chunk = self._create_chunk(initial_sentences, [])
        
        current_embedding = self._compute_chunk_embedding(current_chunk)

        for sentence in sentences[self.min_sentences_per_chunk :]:
            # Compare new sentence against mean embedding of entire current group
            similarity = self._get_semantic_similarity(
                current_embedding, sentence.embedding
            )

            if similarity >= similarity_threshold:
                # Add to current group
                current_chunk.sentences.append(sentence)
                
                # Update score
                current_chunk.scores.append(similarity)
                
                # Update mean embedding
                current_embedding = self._compute_chunk_embedding(current_chunk)
            else:
                # Update score
                current_chunk.total_score = sum(current_chunk.scores) / len(current_chunk.sentences)
                
                # Start new group
                if current_chunk:
                    chunks.append(current_chunk)
                    
                current_chunk = self._create_chunk([sentence], [])
                current_embedding = sentence.embedding

        # Add final group
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _create_chunk(
        self, sentences: List[SemanticSentence], scores: List[float] = None, normalize_scores: bool = False
    ) -> SemanticChunk:
        """Create a chunk from a list of sentences."""
        if not sentences:
            raise ValueError("Cannot create chunk from empty sentence list")

        # Compute chunk text and token count from sentences
        text = " ".join(sent.text for sent in sentences)
        token_count = sum(sent.token_count for sent in sentences) + (
            len(sentences) - 1
        )  # Add spaces

        total_score = None
        if scores is not None:
            if normalize_scores:
                scores = np.asarray(scores)
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            total_score = scores.mean()

        return SemanticChunk(
            text=text,
            start_index=sentences[0].start_index,
            end_index=sentences[-1].end_index,
            token_count=token_count,
            sentences=sentences,
            scores = scores.tolist() if scores is not None else scores,
            total_score = total_score
        )

    def _split_chunks(
        self, chunks: List[SemanticChunk]
    ) -> List[SemanticChunk]:
        """Split sentence groups into chunks that respect chunk_size.

        Args:
            sentence_groups: List of semantically coherent sentence groups

        Returns:
            List of SemanticChunk objects
        """
        new_chunks = []

        for chunk in chunks:
            current_chunk_sentences = []
            current_tokens = 0
            current_scores = []

            score_idx = -2
            for sentence in chunk.sentences:
                test_tokens = (
                    current_tokens
                    + sentence.token_count
                    + (1 if current_chunk_sentences else 0)
                )

                score_idx += 1

                if test_tokens <= self.chunk_size:
                    # Add to current chunk
                    current_chunk_sentences.append(sentence)
                    current_tokens = test_tokens
                    if score_idx >= 0:
                        current_scores.append(chunk.scores[score_idx])
                else:
                    # Create chunk if we have sentences
                    if current_chunk_sentences:
                        new_chunks.append(self._create_chunk(current_chunk_sentences, current_scores))

                    # Start new chunk with current sentence
                    current_chunk_sentences = [sentence]
                    current_tokens = sentence.token_count
                    current_scores = []

            # Create final chunk for this group
            if current_chunk_sentences:
                new_chunks.append(self._create_chunk(current_chunk_sentences, current_scores))

        return new_chunks

    def chunk(self, text: str) -> List[SemanticChunk]:
        """Split text into semantically coherent chunks using two-pass approach.

        First groups sentences by semantic similarity, then splits groups to respect
        chunk_size while maintaining sentence boundaries.

        Args:
            text: Input text to be chunked

        Returns:
            List of SemanticChunk objects containing the chunked text and metadata
        """
        if not text.strip():
            return []

        # Prepare sentences with precomputed information
        sentences = self._prepare_sentences(text)
        if len(sentences) < self.min_sentences_per_chunk:
            return [self._create_chunk(sentences)]

        # First pass: Group sentences by semantic similarity
        chunks = self._chunk_sentences(sentences)

        # Second pass: Split groups into size-appropriate chunks
        chunks = self._split_chunks(chunks)

        return chunks

    def __repr__(self) -> str:
        threshold_info = (
            f"similarity_threshold={self.similarity_threshold}"
            if self.similarity_threshold is not None
            else f"similarity_percentile={self.similarity_percentile}"
        )
        return (
            f"SemanticChunker(chunk_size={self.chunk_size}, "
            f"lang='{self.lang}', "
            f"{threshold_info}, "
            f"min_sentences_per_chunk={self.min_sentences_per_chunk}, "
            f"sentence_mode='{self.sentence_mode}')"
        )

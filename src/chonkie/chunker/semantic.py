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


class SemanticChunker(SentenceChunker):
    def __init__(
        self,
        tokenizer: Union[str, Any] = "gpt2",
        chunk_size: int = 512,
        min_sentences_per_chunk: int = 1,
        sentence_mode: str = "heuristic",
        spacy_model: str = "en_core_web_sm",
        embedding_model: Union[str, Any] = "sentence-transformers/all-MiniLM-L6-v2",
        appending_threshold: Optional[float] = None,
        appending_percentile: Optional[float] = None,
        word_segmentation: bool = False,
        sentence_transformers_kargs: dict = {},
    ):
        """Initialize the SemanticChunker.

        Args:
            tokenizer: Tokenizer for counting tokens
            embedding_model: Name of the sentence-transformers model to load
            chunk_size: Maximum tokens allowed per chunk
            appending_threshold: Absolute threshold for semantic similarity (0-1)
            appending_percentile: Percentile threshold for similarity (0-100)
            min_sentences_per_chunk: Number of sentences to start each chunk with
            sentence_mode: "heuristic" or "spacy" for sentence splitting
            spacy_model: Name of spaCy model to use if sentence_mode="spacy"
            word_segmentation: Whether to use word segmentation

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

        if appending_threshold is not None and (
            appending_threshold < 0 or appending_threshold > 1
        ):
            raise ValueError("appending_threshold must be between 0 and 1")
        if appending_percentile is not None and (
            appending_percentile < 0 or appending_percentile > 100
        ):
            raise ValueError("appending_percentile must be between 0 and 100")
        if appending_threshold is not None and appending_percentile is not None:
            raise ValueError(
                "Cannot specify both appending_threshold and appending_percentile"
            )
        if appending_threshold is None and appending_percentile is None:
            raise ValueError(
                "Must specify either appending_threshold or appending_percentile"
            )

        self.word_segmentation = word_segmentation
        self.appending_threshold = appending_threshold
        self.appending_percentile = appending_percentile

        if self.word_segmentation:
            self._import_pyvi()

        # Load sentence-transformers model
        self._import_sentence_transformers()
        if isinstance(embedding_model, str):
            self.embedding_model = self._load_sentence_transformer_model(
                embedding_model,
                **sentence_transformers_kargs
            )
        else:
            self.embedding_model = embedding_model

    def _import_pyvi(self) -> Any:
        """Import pyvi library. Imports mentioned inside the class,
        because it takes too long to import the whole library at the beginning of the file.
        """
        # Check if pyvi is available
        PYVI_AVAILABLE = importlib.util.find_spec("pyvi") is not None
        if PYVI_AVAILABLE:
            try:
                global vi_tokenizer
                from pyvi.ViTokenizer import tokenize as vi_tokenizer
            except ImportError:
                PYVI_AVAILABLE = False
                warnings.warn(
                    "Failed to import pyvi despite it being installed. SemanticChunker will not work."
                )
        else:
            warnings.warn("pyvi is not installed. SemanticChunker will not work.")

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
                global SentenceTransformer, cos_sim
                from sentence_transformers import SentenceTransformer
                from sentence_transformers.util import cos_sim
            except ImportError:
                SENTENCE_TRANSFORMERS_AVAILABLE = False
                warnings.warn(
                    "Failed to import sentence-transformers despite it being installed. SemanticChunker will not work."
                )
        else:
            warnings.warn(
                "sentence-transformers is not installed. SemanticChunker will not work."
            )

    def _load_sentence_transformer_model(self, model_name: str, **sentence_transformers_kargs) -> Any:
        """Load a sentence-transformers model by name."""
        try:
            model = SentenceTransformer(model_name, **sentence_transformers_kargs)
        except Exception as e:
            raise ImportError(
                f"Failed to load sentence-transformers model '{model_name}'. "
                f"Make sure it is installed and available."
            ) from e
        return model

    def _compute_appending_threshold(self, all_similarities: List[float]) -> float:
        """Compute similarity threshold based on percentile if specified."""
        if self.appending_threshold is not None:
            return self.appending_threshold
        else:
            return float(np.percentile(all_similarities, self.appending_percentile))

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
            [
                sent.text if not self.word_segmentation else vi_tokenizer(sent.text)
                for sent in raw_sentences 
            ], 
            convert_to_numpy=True
        )

        # Create Sentence objects with all precomputed information
        sentences = []
        for idx, (sent, embedding) in enumerate(zip(raw_sentences, embeddings)):
            sentences.append(SemanticSentence(
                text=sent.text,
                start_index=sent.start_index,
                end_index=sent.end_index,
                token_count=sent.token_count,
                embedding=embedding,
                idx = idx
            ))

        return sentences

    def _get_semantic_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        similarity = cos_sim(embedding1, embedding2)
        return similarity[0][0]

    def _compute_chunk_embedding(self, chunk: SemanticChunk) -> np.ndarray:
        """Compute mean embedding for a group of sentences."""

        return np.divide(
            np.sum([(sent.embedding * sent.token_count) for sent in chunk.sentences], axis=0),
            np.sum([sent.token_count for sent in chunk.sentences]),
            dtype=np.float32,
        )

    def _chunk_sentences(self, sentences: List[SemanticSentence], id: str) -> List[SemanticChunk]:
        """Group sentences based on semantic similarity, ignoring token count limits.

        Args:
            sentences: List of Sentence objects with precomputed embeddings

        Returns:
            List of sentence groups, where each group is semantically coherent
        """
        
        # Get or compute similarity threshold
        if self.appending_percentile is not None:
            # Compute all pairwise similarities
            all_similarities = [
                self._get_semantic_similarity(
                    sentences[i].embedding, sentences[i + 1].embedding
                )
                for i in range(len(sentences) - 1)
            ]
            appending_threshold = float(
                np.percentile(all_similarities, self.appending_percentile)
            )
        else:
            appending_threshold = self.appending_threshold

        chunks = []

        initial_sentences = sentences[: self.min_sentences_per_chunk]
        current_chunk = self._create_chunk(initial_sentences, len(chunks), id)
        current_embedding = self._compute_chunk_embedding(current_chunk)

        for sentence in sentences[self.min_sentences_per_chunk :]:
            # Compare new sentence against mean embedding of entire current group
            similarity = self._get_semantic_similarity(
                current_embedding, sentence.embedding
            )

            if similarity >= appending_threshold:
                # Add to current group
                current_chunk = self._create_chunk(current_chunk.sentences + [sentence], len(chunks), id)
                
                # Update mean embedding
                current_embedding = self._compute_chunk_embedding(current_chunk)
            else:
                # Start new group
                if current_chunk:
                    chunks.append(current_chunk)
                    
                current_chunk = self._create_chunk(sentence, len(chunks), id)
                current_embedding = sentence.embedding

        # Add final group
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _create_chunk(
        self, sentences: Union[List[SemanticSentence], SemanticSentence], idx: int = None, id: str = None
    ) -> SemanticChunk:
        """Create a chunk from a list of sentences."""
        if isinstance(sentences, SemanticSentence):
            sentences = [sentences]

        if not sentences:
            raise ValueError("Cannot create chunk from empty sentence list")

        # Compute chunk text and token count from sentences
        text = " ".join(sent.text for sent in sentences)
        token_count = sum(sent.token_count for sent in sentences) + (
            len(sentences) - 1
        )  # Add spaces

        return SemanticChunk(
            text=text,
            start_index=sentences[0].start_index,
            end_index=sentences[-1].end_index,
            token_count=token_count,
            sentences=sentences,
            idx=idx,
            id=id
        )

    def _split_chunks(
        self, chunks: List[SemanticChunk], id: str
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

            for sentence in chunk.sentences:
                test_tokens = (
                    current_tokens
                    + sentence.token_count
                    + (1 if current_chunk_sentences else 0)
                )

                if test_tokens <= self.chunk_size:
                    # Add to current chunk
                    current_chunk_sentences.append(sentence)
                    current_tokens = test_tokens

                else:
                    # Create chunk if we have sentences
                    if current_chunk_sentences:
                        new_chunks.append(self._create_chunk(current_chunk_sentences, len(new_chunks), id))

                    # Start new chunk with current sentence
                    current_chunk_sentences = [sentence]
                    current_tokens = sentence.token_count

            # Create final chunk for this group
            if current_chunk_sentences:
                new_chunks.append(self._create_chunk(current_chunk_sentences, len(new_chunks), id))

        return new_chunks

    def chunk(self, text: str, id: str) -> List[SemanticChunk]:
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
            return [self._create_chunk(sentences, id, 0, id)]

        # First pass: Group sentences by semantic similarity
        chunks = self._chunk_sentences(sentences, id)

        # Second pass: Split groups into size-appropriate chunks
        chunks = self._split_chunks(chunks, id)

        return chunks

    def __repr__(self) -> str:
        threshold_info = (
            f"appending_threshold={self.appending_threshold}"
            if self.appending_threshold is not None
            else f"appending_percentile={self.appending_percentile}"
        )
        return (
            f"SemanticChunker(chunk_size={self.chunk_size}, "
            f"lang='{self.lang}', "
            f"{threshold_info}, "
            f"min_sentences_per_chunk={self.min_sentences_per_chunk}, "
            f"sentence_mode='{self.sentence_mode}')"
        )

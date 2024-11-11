import importlib.util
import re
import warnings
from dataclasses import dataclass
from typing import Any, List, Union

from .base import BaseChunker, Chunk


@dataclass
class Sentence:
    text: str
    start_index: int
    end_index: int
    token_count: int


@dataclass
class SentenceChunk(Chunk):
    text: str
    start_index: int
    end_index: int
    token_count: int
    sentences: List[Sentence] = None


class SentenceChunker(BaseChunker):
    def __init__(
        self,
        tokenizer: Union[str, Any] = "gpt2",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        mode: str = "simple",
        min_sentences_per_chunk: int = 1,
        spacy_model: str = "en_core_web_sm",
    ):
        """Initialize the SentenceChunker with configuration parameters.

        Args:
            tokenizer: The tokenizer instance to use for encoding/decoding
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            mode: Sentence detection mode - "heuristic" (rule-based) or "spacy" (ML-based)
            min_sentences_per_chunk: Minimum number of sentences per chunk (defaults to 1)
            spacy_model: Name of spaCy model to use (defaults to "en_core_web_sm")

        Raises:
            ValueError: If parameters are invalid
            Warning: If spacy mode is requested but spacy is not available
        """
        super().__init__(tokenizer)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if mode not in ["simple", "spacy", "underthesea"]:
            raise ValueError("mode must be either 'simple' or 'spacy'")
        if min_sentences_per_chunk < 1:
            raise ValueError("min_sentences_per_chunk must be at least 1")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_sentences_per_chunk = min_sentences_per_chunk

        # Initialize mode and spaCy
        if mode == "spacy":
            self.nlp = None
            self._import_spacy()
            if not self.SPACY_AVAILABLE:
                warnings.warn(
                    "Spacy not found in environment. To use spacy mode, install it using:\n"
                    "pip install spacy\n"
                    "python -m spacy download en_core_web_sm\n"
                    "Falling back to simple mode."
                )
                self.mode = "simple"
            else:
                try:
                    self.nlp = spacy.load(spacy_model)
                    # Optimize for sentence segmentation only
                    self.nlp.select_pipes(enable=["senter"])
                    self.mode = "spacy"
                except OSError:
                    warnings.warn(
                        f"Spacy model '{spacy_model}' not found. To use spacy mode, "
                        f"install it using: python -m spacy download {spacy_model}\n"
                        "Falling back to simple mode."
                    )
                    self.mode = "simple"
        elif mode == "underthesea":
            self._import_underthesea()
            if not self.UNDERTHESA_AVAILABLE:
                warnings.warn(
                    "Underthesea not found in environment. To use underthesea mode, install it using:\n"
                    "pip install underthesea\n"
                    "Falling back to simple mode."
                )
                self.mode = "simple"
            else:
                self.mode = "underthesea"
        else:
            self.mode = "simple"

    def _import_underthesea(self):
        # Check if underthesea is available
        self.UNDERTHESA_AVAILABLE = importlib.util.find_spec("underthesea") is not None
        if self.UNDERTHESA_AVAILABLE:
            try:
                global underthesea
                import underthesea
            except ImportError:
                self.UNDERTHESA_AVAILABLE = False
                warnings.warn(
                    "Failed to import underthesea despite it being installed. Using heuristic mode only."
                )

    def _import_spacy(self):
        # Check if spacy is available
        self.SPACY_AVAILABLE = importlib.util.find_spec("spacy") is not None
        if self.SPACY_AVAILABLE:
            try:
                global spacy
                import spacy
            except ImportError:
                self.SPACY_AVAILABLE = False
                warnings.warn(
                    "Failed to import spacy despite it being installed. Using heuristic mode only."
                )

    def _craft_sentences(self, text: str, sentences: List[str]) -> List[Sentence]:
        token_counts = self._get_token_counts(sentences)
        current_pos = 0
        for sent_text, token_count in zip(sentences, token_counts):
            if not sent_text:
                continue
            start_idx = text.find(sent_text, current_pos)
            end_idx = start_idx + len(sent_text)
            current_pos = end_idx

            # Get the token count for the sentence
            sentences.append(
                Sentence(
                    text=sent_text,
                    start_index=start_idx,
                    end_index=end_idx,
                    token_count=token_count,
                )
            )
        return sentences

    def _split_into_sentences_via_spacy(self, text: str) -> List[str]:
        """Split text into sentences via spaCy.

        Args:
            text: Input text to be split into sentences

        Returns:
            List of sentences
        """
        # Use spaCy's sentence segmentation
        doc = self.nlp(text)
        sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        return text, sents

    def _split_into_sentences_underthesea(self, text: str) -> List[str]:
        """Split text into sentences via underthesea.

        Args:
            text: Input text to be split into sentences

        Returns:
            List of sentences
        """
        # Use underthesea's sentence segmentation
        sents = underthesea.sent_tokenize(text)

        return text, sents

    def _split_into_sentences_simple(self, text: str) -> List[str]:
        # Fallback to heuristic mode
        # Simple rule-based sentence splitting with common abbreviations
        text = re.sub(
            r'([.!?])([^"])', r"\1\n\2", text
        )  # Add newlines after sentence endings
        text = re.sub(r'([.!?]")(\s*[A-Z])', r"\1\n\2", text)  # Handle quotes

        # Handle common abbreviations
        abbrevs = r"(?:Mr|Mrs|Dr|Prof|Sr|Jr|vs|etc|viz|al|Gen|Col|Fig|e\.g|i\.e)\."
        text = re.sub(f"{abbrevs}\n", f"{abbrevs} ", text)

        # Handle initials and acronyms
        text = re.sub(r"([A-Z]\.[A-Z]\.)\n", r"\1 ", text)
        text = re.sub(
            r"([A-Z]\.[A-Z]\.)\n", r"\1 ", text
        )  # Run twice for consecutive initials

        # Handle decimal numbers and ellipsis
        text = re.sub(r"(\d+)\.\n(\d+)", r"\1.\2", text)  # Decimal numbers
        text = re.sub(r"\.{3}\n", "... ", text)  # Ellipsis

        sents = [s.strip() for s in text.split("\n") if s.strip()]

        return text, sents

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences based on the selected mode.

        Args:
            text: Input text to be split into sentences

        Returns:
            List of sentences
        """
        if self.mode == "spacy" and self.nlp is not None:
            split_func = self._split_into_sentences_via_spacy
        elif self.mode == "underthesea":
            split_func = self._split_into_sentences_underthesea
        elif self.mode == "simple":
            split_func = self._split_into_sentences_simple

        text, sents = split_func(text)

        return self._craft_sentences(text, sents)

    def _get_token_counts(self, sentences: List[str]) -> List[int]:
        """Get token counts for a list of sentences in batch.

        Args:
            sentences: List of sentences

        Returns:
            List of token counts for each sentence
        """
        # Batch encode all sentences at once
        encoded_sentences = self._encode_batch(sentences)
        return [len(encoded) for encoded in encoded_sentences]

    def _create_chunk(
        self, sentences: List[Sentence], start_idx: int, token_count: int
    ) -> Chunk:
        """Create a chunk from a list of sentences.

        Args:
            sentences: List of sentences to create chunk from
            start_idx: Starting index in original text
            token_count: Total token count for the chunk

        Returns:
            Chunk object
        """
        chunk_text = " ".join([sentence.text for sentence in sentences])
        return SentenceChunk(
            text=chunk_text,
            start_index=start_idx,
            end_index=start_idx + token_count,
            token_count=token_count,
            sentences=sentences,
        )

    def chunk(self, text: str) -> List[Chunk]:
        """Split text into overlapping chunks based on sentences while respecting token limits.

        Args:
            text: Input text to be chunked

        Returns:
            List of Chunk objects containing the chunked text and metadata
        """
        if not text.strip():
            return []

        sentences = self._split_into_sentences(text)
        token_counts = [sentence.token_count for sentence in sentences]

        chunks = []
        current_sentences = []
        current_tokens = 0
        last_chunk_end = 0

        for i, (sentence, token_count) in enumerate(zip(sentences, token_counts)):
            # Calculate total tokens if we add this sentence
            test_tokens = (
                current_tokens + token_count + (1 if current_sentences else 0)
            )  # Add 1 for space between sentences

            can_add_sentence = test_tokens <= self.chunk_size or (
                len(current_sentences) < self.min_sentences_per_chunk
                and len(current_sentences) + 1 <= self.min_sentences_per_chunk
            )

            if can_add_sentence:
                # Sentence fits within limits, add it
                current_sentences.append(sentence)
                current_tokens = test_tokens
            else:
                # Sentence would exceed limits, create chunk if we have enough sentences
                if len(current_sentences) >= self.min_sentences_per_chunk:
                    chunk = self._create_chunk(
                        current_sentences, last_chunk_end, current_tokens
                    )
                    chunks.append(chunk)

                    # Calculate overlap for next chunk
                    if self.chunk_overlap > 0:
                        # Keep sentences from the end of current chunk until we hit overlap limit
                        overlap_sentences = []
                        overlap_tokens = 0
                        for sent, tokens in zip(
                            reversed(current_sentences),
                            reversed(token_counts[i - len(current_sentences) : i]),
                        ):
                            test_overlap_tokens = (
                                overlap_tokens
                                + tokens
                                + (1 if overlap_sentences else 0)
                            )
                            if test_overlap_tokens <= self.chunk_overlap:
                                overlap_sentences.insert(0, sent)
                                overlap_tokens = test_overlap_tokens
                            else:
                                break

                        current_sentences = overlap_sentences
                        current_tokens = overlap_tokens
                    else:
                        current_sentences = []
                        current_tokens = 0

                    last_chunk_end += current_tokens

                # Add current sentence (either after creating chunk or when forced to meet minimum)
                current_sentences.append(sentence)
                current_tokens = (
                    current_tokens
                    + token_count
                    + (1 if len(current_sentences) > 1 else 0)
                )

        # Handle remaining sentences
        if current_sentences:
            chunk = self._create_chunk(
                current_sentences, last_chunk_end, current_tokens
            )
            chunks.append(chunk)

        return chunks

    def __repr__(self) -> str:
        return (
            f"SentenceChunker(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, mode='{self.mode}', "
            f"min_sentences_per_chunk={self.min_sentences_per_chunk})"
        )

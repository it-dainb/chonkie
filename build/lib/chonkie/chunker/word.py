import re
from typing import List, Tuple

from tokenizers import Tokenizer

from .base import BaseChunker, Chunk


class WordChunker(BaseChunker):
    def __init__(
        self,
        tokenizer: Tokenizer,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        mode: str = "simple",
    ):
        """Initialize the WordChunker with configuration parameters.

        Args:
            tokenizer: The tokenizer instance to use for encoding/decoding
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Maximum number of tokens to overlap between chunks
            mode: Tokenization mode - "heuristic" (space-based) or "advanced" (handles punctuation)

        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap >= chunk_size or invalid mode
        """
        super().__init__(tokenizer)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if mode not in ["simple", "advanced"]:
            raise ValueError("mode must be either 'heuristic' or 'advanced'")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.mode = mode

    def _split_into_words(self, text: str) -> List[str]:
        """Split text into words based on the selected mode.

        Args:
            text: Input text to be split into words

        Returns:
            List of words
        """
        if self.mode == "simple":
            # Simple space-based splitting
            return text.split()
        elif self.mode == "advanced":
            # Advanced tokenization handling various cases

            # Replace common abbreviations with placeholders to preserve periods
            abbreviations = {
                r"(?i)mr\.": "MR_ABBR",
                r"(?i)mrs\.": "MRS_ABBR",
                r"(?i)ms\.": "MS_ABBR",
                r"(?i)dr\.": "DR_ABBR",
                r"(?i)prof\.": "PROF_ABBR",
                r"(?i)sr\.": "SR_ABBR",
                r"(?i)jr\.": "JR_ABBR",
                r"(?i)vs\.": "VS_ABBR",
                r"(?i)etc\.": "ETC_ABBR",
                r"(?i)i\.e\.": "IE_ABBR",
                r"(?i)e\.g\.": "EG_ABBR",
                # Add numbers with decimal points
                r"(\d+)\.(\d+)": r"\1_DECIMAL_\2",
            }

            processed_text = text
            for pattern, replacement in abbreviations.items():
                processed_text = re.sub(pattern, replacement, processed_text)

            # Split on word boundaries while preserving all punctuation
            # This regex handles:
            # - Words (\w+)
            # - Contractions (word's, don't, etc.)
            # - Hyphenated words (self-aware)
            # - Various punctuation marks
            words = re.findall(
                r"\b[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*\b"  # Words with possible hyphens/apostrophes
                r"|[.,!?;:\"'(){}[\]]+"  # Punctuation groups
                r"|\.{3}"  # Ellipsis
                r"|--+"  # Em/En dashes
                r"|'[A-Za-z]+"  # Starting contractions like 'tis
                r"|[A-Za-z]+'"  # Ending contractions like ol'
                r"|[@#$%&*+=/<>~^]",  # Special characters
                processed_text,
            )

            # Process the words
            final_words = []
            skip_next = False

            for i, word in enumerate(words):
                if skip_next:
                    skip_next = False
                    continue

                # Restore abbreviations
                for abbr, placeholder in abbreviations.items():
                    if placeholder in word:
                        word = re.sub(r"_ABBR$", ".", word)
                        word = re.sub(r"_DECIMAL_", ".", word)

                # Handle punctuation
                if word in ".,!?;:\"'(){}[]":
                    if final_words:  # Attach punctuation to previous word
                        # Don't attach if previous word already ends with punctuation
                        if final_words[-1][-1] not in ".,!?;:\"'(){}[]":
                            final_words[-1] += word
                        else:
                            final_words.append(word)
                    else:
                        final_words.append(word)

                # Handle ellipsis
                elif word == "...":
                    if final_words:
                        final_words[-1] += word
                    else:
                        final_words.append(word)

                # Handle dashes between words
                elif word.startswith("-") and i > 0:
                    if i < len(words) - 1 and re.match(r"\w+", words[i + 1]):
                        final_words[-1] += word + words[i + 1]
                        skip_next = True
                    else:
                        final_words.append(word)

                # Regular words
                else:
                    final_words.append(word)

            return final_words
        else:
            raise ValueError("mode must be either 'heuristic' or 'advanced'")

    def _get_token_count(self, text: str) -> int:
        """Get the number of tokens in a text string.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self._encode(text))

    def _create_chunk(
        self, words: List[str], start_idx: int, end_idx: int
    ) -> Tuple[Chunk, int]:
        """Create a chunk from a list of words.

        Args:
            words: List of words to create chunk from
            start_idx: Starting index in original text
            end_idx: Ending index in original text

        Returns:
            Tuple of (Chunk object, number of tokens in chunk)
        """
        chunk_text = " ".join(words)
        token_count = self._get_token_count(chunk_text)
        return Chunk(
            text=chunk_text,
            start_index=start_idx,
            end_index=end_idx,
            token_count=token_count,
        )

    def _get_word_list_token_counts(self, words: List[str]) -> List[int]:
        """Get the number of tokens for each word in a list.

        Args:
            words: List of words

        Returns:
            List of token counts for each word
        """
        words = [
            " " + word.strip() for word in words
        ]  # Add space in the beginning because tokenizers usually split that differently
        encodings = self._encode_batch(words)
        return [len(encoding) for encoding in encodings]

    def chunk(self, text: str) -> List[Chunk]:
        """Split text into overlapping chunks based on words while respecting token limits.

        Args:
            text: Input text to be chunked

        Returns:
            List of Chunk objects containing the chunked text and metadata
        """
        if not text.strip():
            return []

        words = self._split_into_words(text)
        lengths = self._get_word_list_token_counts(words)
        chunks = []

        # Saving the current chunk
        current_chunk = []
        current_chunk_length = 0

        for i, (word, length) in enumerate(zip(words, lengths)):
            if current_chunk_length + length <= self.chunk_size:
                current_chunk.append(word)
                current_chunk_length += length
            else:
                chunk = self._create_chunk(current_chunk, i - len(current_chunk), i - 1)
                chunks.append(chunk)

                # update the current_chunk and previous chunk
                previous_chunk_length = current_chunk_length

                current_chunk = []
                current_chunk_length = 0

                overlap = []
                overlap_length = 0
                # calculate the overlap from the current chunk in reverse
                for j in range(0, previous_chunk_length):
                    cwi = i - 1 - j
                    oword = words[cwi]
                    olength = lengths[cwi]
                    if overlap_length + olength <= self.chunk_overlap:
                        overlap.append(oword)
                        overlap_length += olength
                    else:
                        break

                current_chunk = [w for w in reversed(overlap)]
                current_chunk_length = overlap_length

                current_chunk.append(word)
                current_chunk_length += length

        # Add the final chunk if it has any words
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk, len(words) - len(current_chunk), len(words) - 1
            )
            chunks.append(chunk)
        return chunks

    def __repr__(self) -> str:
        return (
            f"WordChunker(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, mode='{self.mode}')"
        )

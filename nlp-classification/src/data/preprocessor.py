"""
Text preprocessing utilities for NLP classification.
"""
import re
import string
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from collections import Counter
import logging
import jieba
import jieba.posseg as pseg

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class TextPreprocessor:
    """
    Comprehensive text preprocessing for NLP classification tasks.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = True,
        stemming: bool = False,
        lemmatization: bool = True,
        min_word_length: int = 2,
        max_word_length: int = 50,
        custom_stopwords: Optional[List[str]] = None,
        language: str = 'english'
    ):
        """
        Initialize TextPreprocessor.

        Args:
            lowercase (bool): Convert text to lowercase
            remove_punctuation (bool): Remove punctuation
            remove_numbers (bool): Remove numbers
            remove_stopwords (bool): Remove stopwords
            stemming (bool): Apply stemming
            lemmatization (bool): Apply lemmatization (if both stemming and lemmatization are True, lemmatization takes priority)
            min_word_length (int): Minimum word length to keep
            max_word_length (int): Maximum word length to keep
            custom_stopwords (List[str], optional): Additional stopwords to remove
            language (str): Language for stopwords and other processing
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.lemmatization = lemmatization
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.language = language

        # Initialize NLTK components
        if self.remove_stopwords:
            if language == 'chinese':
                # Common Chinese stopwords
                chinese_stopwords = [
                    '的', '了', '在', '是', '我', '有', '和', '就',
                    '不', '人', '都', '一', '一个', '上', '也', '很',
                    '到', '说', '要', '去', '你', '会', '着', '没有',
                    '看', '好', '自己', '这'
                ]
                self.stop_words = set(chinese_stopwords)
                if custom_stopwords:
                    self.stop_words.update(custom_stopwords)
            else:
                self.stop_words = set(stopwords.words(language))
                if custom_stopwords:
                    self.stop_words.update(custom_stopwords)
        else:
            self.stop_words = set()

        if self.stemming:
            self.stemmer = PorterStemmer()

        if self.lemmatization:
            self.lemmatizer = WordNetLemmatizer()

        # Regex patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'[\+]?[1-9]?[0-9]{7,15}')
        self.number_pattern = re.compile(r'\d+')
        self.whitespace_pattern = re.compile(r'\s+')

        logger.info("TextPreprocessor initialized with the following settings:")
        logger.info(f"  lowercase: {self.lowercase}")
        logger.info(f"  remove_punctuation: {self.remove_punctuation}")
        logger.info(f"  remove_numbers: {self.remove_numbers}")
        logger.info(f"  remove_stopwords: {self.remove_stopwords}")
        logger.info(f"  stemming: {self.stemming}")
        logger.info(f"  lemmatization: {self.lemmatization}")

    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning operations.

        Args:
            text (str): Input text

        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Remove URLs
        text = self.url_pattern.sub(' ', text)

        # Remove email addresses
        text = self.email_pattern.sub(' ', text)

        # Remove phone numbers
        text = self.phone_pattern.sub(' ', text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove numbers
        if self.remove_numbers:
            text = self.number_pattern.sub(' ', text)

        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text).strip()

        return text

    def _is_chinese_text(self, text: str) -> bool:
        """
        Check if text contains Chinese characters.

        Args:
            text (str): Input text

        Returns:
            bool: True if text contains Chinese characters
        """
        chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')
        return bool(chinese_char_pattern.search(text))

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text (str): Input text

        Returns:
            List[str]: List of tokens
        """
        # Check if text contains Chinese characters or if language is set to Chinese
        if self.language == 'chinese' or self._is_chinese_text(text):
            try:
                # Use jieba for Chinese text segmentation
                tokens = list(jieba.cut(text, cut_all=False))
                # Filter out empty tokens and whitespace
                tokens = [token.strip() for token in tokens if token.strip()]
            except Exception as e:
                logger.warning(f"Jieba tokenization failed, falling back to character split: {e}")
                # Fallback to character-level tokenization for Chinese
                tokens = list(text)
        else:
            try:
                tokens = word_tokenize(text)
            except Exception as e:
                logger.warning(f"NLTK tokenization failed, falling back to simple split: {e}")
                tokens = text.split()

        return tokens

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens based on length and stopwords.

        Args:
            tokens (List[str]): List of tokens

        Returns:
            List[str]: Filtered tokens
        """
        filtered_tokens = []

        for token in tokens:
            # Check word length
            if len(token) < self.min_word_length or len(token) > self.max_word_length:
                continue

            # Check stopwords
            if self.remove_stopwords and token.lower() in self.stop_words:
                continue

            # Check if token is not just whitespace
            if token.strip():
                filtered_tokens.append(token)

        return filtered_tokens

    def apply_morphological_processing(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming or lemmatization.

        Args:
            tokens (List[str]): List of tokens

        Returns:
            List[str]: Processed tokens
        """
        if not tokens:
            return tokens

        processed_tokens = []

        # Skip morphological processing for Chinese as it's not applicable
        if self.language == 'chinese':
            # For Chinese, we can optionally use jieba's part-of-speech tagging
            # but typically don't need stemming/lemmatization
            processed_tokens = tokens
        elif self.lemmatization:
            # Get POS tags for better lemmatization
            try:
                pos_tags = pos_tag(tokens)
                for token, pos in pos_tags:
                    # Convert POS tag to WordNet format
                    wordnet_pos = self._get_wordnet_pos(pos)
                    lemmatized = self.lemmatizer.lemmatize(token, wordnet_pos)
                    processed_tokens.append(lemmatized)
            except Exception as e:
                logger.warning(f"Lemmatization failed, falling back to simple lemmatization: {e}")
                processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        elif self.stemming:
            processed_tokens = [self.stemmer.stem(token) for token in tokens]

        else:
            processed_tokens = tokens

        return processed_tokens

    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """
        Convert TreeBank POS tag to WordNet POS tag.

        Args:
            treebank_tag (str): TreeBank POS tag

        Returns:
            str: WordNet POS tag
        """
        if treebank_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return nltk.corpus.wordnet.NOUN  # Default to noun

    def preprocess_text(self, text: str) -> str:
        """
        Complete text preprocessing pipeline.

        Args:
            text (str): Input text

        Returns:
            str: Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)

        # Tokenize
        tokens = self.tokenize(text)

        # Filter tokens
        tokens = self.filter_tokens(tokens)

        # Apply morphological processing
        tokens = self.apply_morphological_processing(tokens)

        # Join tokens back to text
        return ' '.join(tokens)

    def preprocess_batch(self, texts: List[str], show_progress: bool = True) -> List[str]:
        """
        Preprocess a batch of texts.

        Args:
            texts (List[str]): List of input texts
            show_progress (bool): Show progress bar

        Returns:
            List[str]: List of preprocessed texts
        """
        if show_progress:
            from tqdm import tqdm
            texts_iter = tqdm(texts, desc="Preprocessing texts")
        else:
            texts_iter = texts

        preprocessed_texts = [self.preprocess_text(text) for text in texts_iter]
        return preprocessed_texts

    def get_preprocessing_stats(self, original_texts: List[str], preprocessed_texts: List[str]) -> Dict[str, Any]:
        """
        Get statistics about the preprocessing operation.

        Args:
            original_texts (List[str]): Original texts
            preprocessed_texts (List[str]): Preprocessed texts

        Returns:
            Dict[str, Any]: Preprocessing statistics
        """
        original_lengths = [len(text) for text in original_texts]
        preprocessed_lengths = [len(text) for text in preprocessed_texts]

        original_word_counts = [len(text.split()) for text in original_texts]
        preprocessed_word_counts = [len(text.split()) for text in preprocessed_texts]

        stats = {
            'num_texts': len(original_texts),
            'character_stats': {
                'original': {
                    'total_chars': sum(original_lengths),
                    'avg_chars': np.mean(original_lengths),
                    'max_chars': max(original_lengths),
                    'min_chars': min(original_lengths)
                },
                'preprocessed': {
                    'total_chars': sum(preprocessed_lengths),
                    'avg_chars': np.mean(preprocessed_lengths),
                    'max_chars': max(preprocessed_lengths),
                    'min_chars': min(preprocessed_lengths)
                }
            },
            'word_stats': {
                'original': {
                    'total_words': sum(original_word_counts),
                    'avg_words': np.mean(original_word_counts),
                    'max_words': max(original_word_counts),
                    'min_words': min(original_word_counts)
                },
                'preprocessed': {
                    'total_words': sum(preprocessed_word_counts),
                    'avg_words': np.mean(preprocessed_word_counts),
                    'max_words': max(preprocessed_word_counts),
                    'min_words': min(preprocessed_word_counts)
                }
            },
            'reduction_ratios': {
                'character_reduction': 1 - (sum(preprocessed_lengths) / sum(original_lengths)),
                'word_reduction': 1 - (sum(preprocessed_word_counts) / sum(original_word_counts))
            }
        }

        return stats


class LabelEncoder:
    """
    Encode and decode categorical labels.
    """

    def __init__(self):
        self.label_to_id = {}
        self.id_to_label = {}
        self.num_classes = 0

    def fit(self, labels: List[str]) -> 'LabelEncoder':
        """
        Fit encoder on labels.

        Args:
            labels (List[str]): List of labels

        Returns:
            LabelEncoder: Self
        """
        unique_labels = sorted(list(set(labels)))
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        self.num_classes = len(unique_labels)

        logger.info(f"Label encoder fitted with {self.num_classes} classes:")
        for label, idx in self.label_to_id.items():
            logger.info(f"  {idx}: {label}")

        return self

    def transform(self, labels: List[str]) -> List[int]:
        """
        Transform labels to integers.

        Args:
            labels (List[str]): List of labels

        Returns:
            List[int]: List of encoded labels
        """
        encoded_labels = []
        for label in labels:
            if label not in self.label_to_id:
                raise ValueError(f"Unknown label: {label}")
            encoded_labels.append(self.label_to_id[label])
        return encoded_labels

    def inverse_transform(self, encoded_labels: List[int]) -> List[str]:
        """
        Transform integers back to labels.

        Args:
            encoded_labels (List[int]): List of encoded labels

        Returns:
            List[str]: List of original labels
        """
        decoded_labels = []
        for encoded_label in encoded_labels:
            if encoded_label not in self.id_to_label:
                raise ValueError(f"Unknown encoded label: {encoded_label}")
            decoded_labels.append(self.id_to_label[encoded_label])
        return decoded_labels

    def fit_transform(self, labels: List[str]) -> List[int]:
        """
        Fit and transform labels.

        Args:
            labels (List[str]): List of labels

        Returns:
            List[int]: List of encoded labels
        """
        return self.fit(labels).transform(labels)


if __name__ == "__main__":
    # Test text preprocessor
    import json

    logging.basicConfig(level=logging.INFO)

    print("Text Preprocessor Test")
    print("=" * 50)

    # Sample texts (English and Chinese)
    sample_texts = [
        "Hello World! This is a TEST of the preprocessing system.",
        "Visit https://example.com for more info. Call 123-456-7890 or email test@example.com",
        "The quick brown fox jumps over the lazy dog. 123 times!",
        "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans.",
        "你好世界！这是一个中文文本预处理的测试。",
        "自然语言处理（NLP）是人工智能的一个重要分支，专注于计算机与人类语言之间的交互。",
        "中国是一个历史悠久的国家，有着丰富的文化传统。",
        ""
    ]

    # Test different configurations
    configs = [
        {"name": "Basic", "params": {}},
        {"name": "Aggressive", "params": {"remove_numbers": True, "stemming": True, "lemmatization": False}},
        {"name": "Minimal", "params": {"lowercase": False, "remove_punctuation": False, "remove_stopwords": False}},
        {"name": "Chinese", "params": {"language": "chinese", "lemmatization": False, "stemming": False}}
    ]

    for config in configs:
        print(f"\n{config['name']} Configuration:")
        print("-" * 30)

        preprocessor = TextPreprocessor(**config['params'])

        for i, text in enumerate(sample_texts):
            if text:  # Skip empty text
                processed = preprocessor.preprocess_text(text)
                print(f"Original {i+1}: {text}")
                print(f"Processed {i+1}: {processed}")
                print()

    # Test label encoder
    print("\nLabel Encoder Test:")
    print("-" * 30)

    labels = ['sports', 'politics', 'technology', 'sports', 'entertainment', 'politics']
    encoder = LabelEncoder()

    encoded = encoder.fit_transform(labels)
    decoded = encoder.inverse_transform(encoded)

    print(f"Original labels: {labels}")
    print(f"Encoded labels: {encoded}")
    print(f"Decoded labels: {decoded}")
    print(f"Number of classes: {encoder.num_classes}")

    # Test preprocessing stats
    print("\nPreprocessing Statistics Test:")
    print("-" * 30)

    preprocessor = TextPreprocessor()
    original_texts = sample_texts[:-1]  # Exclude empty text
    preprocessed_texts = preprocessor.preprocess_batch(original_texts, show_progress=False)

    stats = preprocessor.get_preprocessing_stats(original_texts, preprocessed_texts)
    print(json.dumps(stats, indent=2))
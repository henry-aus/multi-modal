"""
Data loading utilities for various file formats.
"""
import pandas as pd
import json
import csv
import os
from typing import Tuple, List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Utility class for loading data from various file formats.

    Supports CSV, TSV, JSON, and TXT formats.
    """

    def __init__(self, text_column: str = "text", label_column: str = "label"):
        """
        Initialize DataLoader.

        Args:
            text_column (str): Name of the text column
            label_column (str): Name of the label column
        """
        self.text_column = text_column
        self.label_column = label_column

    def load_csv(self, file_path: str, delimiter: str = ",", **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            file_path (str): Path to CSV file
            delimiter (str): CSV delimiter
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, **kwargs)
            logger.info(f"Loaded {len(df)} samples from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            raise

    def load_tsv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from TSV file.

        Args:
            file_path (str): Path to TSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            pd.DataFrame: Loaded data
        """
        return self.load_csv(file_path, delimiter="\t", **kwargs)

    def load_json(self, file_path: str) -> pd.DataFrame:
        """
        Load data from JSON file.

        Supports both line-delimited JSON and regular JSON arrays.

        Args:
            file_path (str): Path to JSON file

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # Try to load as regular JSON first
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])

        except json.JSONDecodeError:
            # Try line-delimited JSON
            try:
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
                df = pd.DataFrame(data)
            except Exception as e:
                logger.error(f"Error loading JSON file {file_path}: {e}")
                raise

        logger.info(f"Loaded {len(df)} samples from {file_path}")
        return df

    def load_txt(self, file_path: str, format_type: str = "label_text") -> pd.DataFrame:
        """
        Load data from text file.

        Args:
            file_path (str): Path to text file
            format_type (str): Format of the text file
                - "label_text": Each line contains "label<tab>text"
                - "text_only": Each line contains only text (no labels)
                - "label_prefix": Each line starts with "__label__<label> text"

        Returns:
            pd.DataFrame: Loaded data
        """
        data = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    if format_type == "label_text":
                        parts = line.split('\t', 1)
                        if len(parts) == 2:
                            label, text = parts
                            data.append({self.label_column: label, self.text_column: text})
                        else:
                            logger.warning(f"Line {line_num}: Invalid format, expected label<tab>text")

                    elif format_type == "text_only":
                        data.append({self.text_column: line, self.label_column: None})

                    elif format_type == "label_prefix":
                        if line.startswith("__label__"):
                            parts = line.split(" ", 1)
                            if len(parts) == 2:
                                label = parts[0].replace("__label__", "")
                                text = parts[1]
                                data.append({self.label_column: label, self.text_column: text})
                            else:
                                logger.warning(f"Line {line_num}: Invalid label prefix format")
                        else:
                            data.append({self.text_column: line, self.label_column: None})

                    else:
                        raise ValueError(f"Unsupported format_type: {format_type}")

            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} samples from {file_path}")
            return df

        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            raise

    def load_data(self, file_path: str, file_format: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from file with automatic format detection.

        Args:
            file_path (str): Path to data file
            file_format (str, optional): Force specific format (csv, tsv, json, txt)
            **kwargs: Additional arguments for specific loaders

        Returns:
            pd.DataFrame: Loaded data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Auto-detect format if not specified
        if file_format is None:
            file_extension = os.path.splitext(file_path)[1].lower()
            format_mapping = {
                '.csv': 'csv',
                '.tsv': 'tsv',
                '.json': 'json',
                '.jsonl': 'json',
                '.txt': 'txt'
            }
            file_format = format_mapping.get(file_extension, 'csv')

        # Load data based on format
        if file_format == 'csv':
            return self.load_csv(file_path, **kwargs)
        elif file_format == 'tsv':
            return self.load_tsv(file_path, **kwargs)
        elif file_format == 'json':
            return self.load_json(file_path)
        elif file_format == 'txt':
            return self.load_txt(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate loaded data.

        Args:
            df (pd.DataFrame): Data to validate

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []

        # Check if required columns exist
        if self.text_column not in df.columns:
            issues.append(f"Missing text column: {self.text_column}")

        if self.label_column not in df.columns:
            issues.append(f"Missing label column: {self.label_column}")

        if issues:
            return False, issues

        # Check for empty data
        if len(df) == 0:
            issues.append("Dataset is empty")

        # Check for missing values
        missing_text = df[self.text_column].isnull().sum()
        missing_labels = df[self.label_column].isnull().sum()

        if missing_text > 0:
            issues.append(f"Found {missing_text} missing text values")

        if missing_labels > 0:
            issues.append(f"Found {missing_labels} missing label values")

        # Check for empty strings
        empty_text = (df[self.text_column].str.strip() == "").sum()
        if empty_text > 0:
            issues.append(f"Found {empty_text} empty text values")

        # Basic statistics
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Number of unique labels: {df[self.label_column].nunique()}")
        logger.info(f"Label distribution:\n{df[self.label_column].value_counts().head()}")

        return len(issues) == 0, issues

    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed information about the dataset.

        Args:
            df (pd.DataFrame): Dataset to analyze

        Returns:
            Dict[str, Any]: Dataset information
        """
        info = {
            'num_samples': len(df),
            'num_features': len(df.columns),
            'columns': list(df.columns),
            'text_column': self.text_column,
            'label_column': self.label_column
        }

        if self.text_column in df.columns:
            text_lengths = df[self.text_column].str.len()
            info['text_statistics'] = {
                'min_length': text_lengths.min(),
                'max_length': text_lengths.max(),
                'mean_length': text_lengths.mean(),
                'median_length': text_lengths.median(),
                'std_length': text_lengths.std()
            }

        if self.label_column in df.columns:
            label_counts = df[self.label_column].value_counts()
            info['label_statistics'] = {
                'num_unique_labels': len(label_counts),
                'label_counts': label_counts.to_dict(),
                'most_common_label': label_counts.index[0],
                'least_common_label': label_counts.index[-1],
                'class_balance': {
                    'most_frequent_count': label_counts.iloc[0],
                    'least_frequent_count': label_counts.iloc[-1],
                    'balance_ratio': label_counts.iloc[0] / label_counts.iloc[-1]
                }
            }

        return info


def create_sample_data(output_path: str, num_samples: int = 1000, num_classes: int = 5):
    """
    Create sample topic classification data for testing.

    Args:
        output_path (str): Path to save the sample data
        num_samples (int): Number of samples to generate
        num_classes (int): Number of topic classes
    """
    import random

    # Define sample topics and associated keywords
    topics = {
        'technology': ['computer', 'software', 'programming', 'artificial intelligence', 'machine learning', 'data science', 'python', 'algorithm'],
        'sports': ['football', 'basketball', 'soccer', 'tennis', 'olympics', 'championship', 'athlete', 'competition'],
        'politics': ['government', 'election', 'policy', 'democracy', 'president', 'congress', 'legislation', 'campaign'],
        'science': ['research', 'experiment', 'discovery', 'theory', 'laboratory', 'scientist', 'study', 'analysis'],
        'entertainment': ['movie', 'music', 'celebrity', 'television', 'concert', 'actor', 'film', 'performance']
    }

    # Select topics based on num_classes
    selected_topics = list(topics.keys())[:num_classes]

    data = []
    for _ in range(num_samples):
        topic = random.choice(selected_topics)
        keywords = random.sample(topics[topic], random.randint(2, 4))

        # Generate synthetic text
        text = f"This is about {' and '.join(keywords)}. "
        text += f"The topic of {topic} is very interesting and involves many aspects including "
        text += f"{', '.join(random.sample(topics[topic], random.randint(1, 3)))}."

        data.append({
            'text': text,
            'label': topic
        })

    # Save as CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f"Created sample data with {num_samples} samples and {num_classes} classes at {output_path}")


if __name__ == "__main__":
    # Test data loader
    logging.basicConfig(level=logging.INFO)

    print("Data Loader Test")
    print("=" * 50)

    # Create sample data
    sample_data_path = "sample_data.csv"
    create_sample_data(sample_data_path, num_samples=100, num_classes=3)

    # Test loader
    loader = DataLoader()
    df = loader.load_data(sample_data_path)

    # Validate data
    is_valid, issues = loader.validate_data(df)
    print(f"Data validation: {'PASSED' if is_valid else 'FAILED'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")

    # Get data info
    info = loader.get_data_info(df)
    print(f"\nDataset Info:")
    print(f"  Samples: {info['num_samples']}")
    print(f"  Classes: {info['label_statistics']['num_unique_labels']}")
    print(f"  Text length stats: {info['text_statistics']}")

    # Cleanup
    if os.path.exists(sample_data_path):
        os.remove(sample_data_path)
        print(f"\nCleaned up {sample_data_path}")
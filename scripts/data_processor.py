import pandas as pd
from pathlib import Path
import re
import logging
import os
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Constants
DEFAULT_DATA_DIR = "data/Dataset"
DEFAULT_OUTPUT_PATH = "data/processed_data.xlsx"
CANCER_LABEL = "Cancer"
NON_CANCER_LABEL = "Non-Cancer"

def setup_logging() -> logging.Logger:
    """
    Set up logging configuration.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_dir: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the data processor.
        
        Args:
            data_dir (str): Path to the dataset directory
            logger (Optional[logging.Logger]): Logger instance
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
            
        self.cancer_dir = self.data_dir / "Cancer"
        self.non_cancer_dir = self.data_dir / "Non-Cancer"
        
        if not self.cancer_dir.exists() or not self.non_cancer_dir.exists():
            raise ValueError("Cancer or Non-Cancer directory not found in data directory")
            
        self.logger = logger or setup_logging()
    
    def read_text_file(self, file_path: Path) -> str:
        """
        Read a text file and return its contents.
        
        Args:
            file_path (Path): Path to the text file
            
        Returns:
            str: Contents of the text file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            return ""
    
    def extract_document_id(self, text: str) -> str:
        """
        Extract document ID from the text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Document ID
        """
        id_match = re.search(r'<ID:(\d+)>', text)
        return id_match.group(1) if id_match else ""

    def extract_abstract(self, text: str) -> str:
        """
        Extract only explicitly labeled abstracts from the text.
        Filter out abstracts that contain other section labels like OBJECTIVES, STUDY QUESTION, etc.
        and remove text with special characters.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Abstract text
        """
        abstract_match = re.search(r'Abstract:(.*?)(?=\n\n|\Z)', text, re.DOTALL | re.IGNORECASE)
        if not abstract_match:
            return ""
            
        abstract = abstract_match.group(1).strip()
        abstract = re.sub(r'\bUNLABELLED\b', '', abstract, flags=re.IGNORECASE).strip()
        abstract = re.sub(r'<b>|</b>', '', abstract)
        
        return abstract.lower()
    
    def process_file(self, file_path: Path, label: int) -> Optional[Dict[str, Any]]:
        """
        Process a single file and extract relevant information.
        
        Args:
            file_path (Path): Path to the file
            label (int): Label for the file (1 for cancer, 0 for non-cancer)
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing processed data or None if processing fails
        """
        text = self.read_text_file(file_path)
        if not text:
            return None
            
        doc_id = self.extract_document_id(text)
        abstract = self.extract_abstract(text)
        
        if not abstract:
            return None
            
        return {
            'doc_id': doc_id,
            'text': abstract,
            'label': label,
            'source_file': str(file_path)
        }
   
    def create_dataframe(self) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the text files.
        
        Returns:
            pd.DataFrame: DataFrame containing the processed data
        """
        data: List[Dict[str, Any]] = []
        
        # Process cancer articles
        for file_path in tqdm(self.cancer_dir.glob('*.txt'), desc="Processing cancer articles"):
            if processed_data := self.process_file(file_path, CANCER_LABEL):
                data.append(processed_data)
        
        # Process non-cancer articles
        for file_path in tqdm(self.non_cancer_dir.glob('*.txt'), desc="Processing non-cancer articles"):
            if processed_data := self.process_file(file_path, NON_CANCER_LABEL):
                data.append(processed_data)
        
        if not data:
            raise ValueError("No valid data found in the input directories")
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df['text_length'] = df['text'].str.len()
        df = df.sort_values('doc_id').reset_index(drop=True)
        
        return df

    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save the processed data to an Excel file.
        
        Args:
            df (pd.DataFrame): Processed DataFrame
            output_path (str): Path to save the Excel file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_excel(output_path, index=False, engine='openpyxl')
        self.logger.info(f"Processed data saved to {output_path}")
        
        # Print dataset statistics
        self.logger.info("\nDataset Statistics:")
        self.logger.info(f"Total number of samples: {len(df)}")
        self.logger.info(f"Number of cancer articles: {len(df[df['label'] == CANCER_LABEL])}")
        self.logger.info(f"Number of non-cancer articles: {len(df[df['label'] == NON_CANCER_LABEL])}")
        self.logger.info(f"Average text length: {df['text_length'].mean():.2f} characters")
        self.logger.info(f"Minimum text length: {df['text_length'].min()} characters")
        self.logger.info(f"Maximum text length: {df['text_length'].max()} characters")
        
        self.logger.info("\nSample of processed data:")
        self.logger.info(df.head())

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Process medical text data for classification.')
    parser.add_argument(
        '--data-dir',
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f'Path to the dataset directory (default: {DEFAULT_DATA_DIR})'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f'Path to save the processed data (default: {DEFAULT_OUTPUT_PATH})'
    )
    return parser.parse_args()

def main() -> None:
    """
    Main function to process the data.
    """
    args = parse_args()
    logger = setup_logging()
    
    try:
        # Initialize data processor
        processor = DataProcessor(args.data_dir, logger)
        
        # Create DataFrame
        df = processor.create_dataframe()
        
        # Save processed data
        processor.save_processed_data(df, args.output_path)
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
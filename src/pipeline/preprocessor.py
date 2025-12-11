"""
Text preprocessing and chunking for vector database indexing
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import tiktoken
from config.settings import (
    CHUNK_SIZE, CHUNK_OVERLAP, 
    DATA_PROCESSED_PATH, DATA_METADATA_PATH
)

logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """Preprocess documents for vector database indexing"""
    
    def __init__(self, 
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP,
                 model: str = "gpt-4"):
        """
        Initialize preprocessor
        
        Args:
            chunk_size: Target size of chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
            model: Model for tokenization
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.encoding_for_model(model)
        
    def process_filing(self,
                      ticker: str,
                      fiscal_year: int,
                      pages_data: Dict[int, Dict],
                      sections: Dict[str, any],
                      tables: List[Dict]) -> Dict:
        """
        Process a complete filing into chunks with metadata
        
        Args:
            ticker: Company ticker
            fiscal_year: Fiscal year of filing
            pages_data: Dictionary of page data from parser
            sections: Dictionary of section info (can be page ranges or section objects)
            tables: List of extracted tables
            
        Returns:
            Processing results with chunks and metadata
        """
        logger.info(f"Processing filing for {ticker} - {fiscal_year}")
        
        doc_id = f"{ticker}_{fiscal_year}_10K"
        
        # Process each section
        all_chunks = []
        section_metadata = {}
        
        for section_id, section_info in sections.items():
            # Handle both formats: List[int] (PDF) or Dict (HTML)
            if isinstance(section_info, dict):
                # HTML parser format with 'text' already extracted
                section_text = section_info.get('text', '')
                page_num = section_info.get('page_num', 1)
                page_range = [page_num, page_num]
            else:
                # PDF parser format with page numbers
                page_nums = section_info if isinstance(section_info, list) else [section_info]
                section_text = self._extract_section_text(pages_data, page_nums)
                page_range = [min(page_nums), max(page_nums)]
                page_num = page_range[0]
            
            # Skip empty sections
            if not section_text or not section_text.strip():
                continue
                
            chunks = self._chunk_text(section_text, section_id, page_num)
            
            section_metadata[section_id] = {
                'page_range': page_range,
                'num_chunks': len(chunks),
                'char_length': len(section_text)
            }
            
            all_chunks.extend(chunks)
        
        # Process tables
        table_chunks = self._process_tables(tables)
        all_chunks.extend(table_chunks)
        
        # Create document metadata
        document = {
            'doc_id': doc_id,
            'ticker': ticker,
            'fiscal_year': fiscal_year,
            'filing_type': '10-K',
            'total_chunks': len(all_chunks),
            'sections': section_metadata,
            'num_tables': len(tables),
            'chunks': all_chunks
        }
        
        # Save processed document
        self._save_document(document)
        
        logger.info(f"Created {len(all_chunks)} chunks for {doc_id}")
        return document
    
    def _extract_section_text(self, pages_data: Dict[int, Dict], page_nums: List[int]) -> str:
        """
        Extract text from section pages
        
        Args:
            pages_data: Dictionary of page data
            page_nums: List of page numbers in section
            
        Returns:
            Combined section text
        """
        section_text = []
        
        for page_num in sorted(page_nums):
            if page_num in pages_data:
                section_text.append(pages_data[page_num]['text'])
        
        return "\n\n".join(section_text)
    
    def _chunk_text(self, 
                   text: str, 
                   section_id: str, 
                   start_page: int) -> List[Dict]:
        """
        Chunk text with overlap
        
        Args:
            text: Text to chunk
            section_id: Section identifier
            start_page: Starting page number
            
        Returns:
            List of chunk dictionaries
        """
        # Tokenize
        tokens = self.encoding.encode(text)
        
        chunks = []
        start_idx = 0
        chunk_num = 0
        
        while start_idx < len(tokens):
            # Get chunk tokens
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create chunk metadata
            chunk = {
                'chunk_id': f"{section_id}_chunk_{chunk_num}",
                'section_id': section_id,
                'text': chunk_text,
                'token_count': len(chunk_tokens),
                'char_count': len(chunk_text),
                'start_page': start_page,
                'chunk_index': chunk_num
            }
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start_idx = end_idx - self.chunk_overlap
            chunk_num += 1
        
        return chunks
    
    def _process_tables(self, tables: List[Dict]) -> List[Dict]:
        """
        Process tables into searchable chunks
        
        Args:
            tables: List of extracted tables
            
        Returns:
            List of table chunk dictionaries
        """
        table_chunks = []
        
        for table in tables:
            # Convert table to text representation
            df = table['dataframe']
            
            # Create table text
            table_text = f"Table on page {table['page']}:\n"
            table_text += df.to_string(index=False)
            
            # Create chunk
            chunk = {
                'chunk_id': f"table_{table['table_id']}",
                'section_id': 'TABLE',
                'text': table_text,
                'token_count': len(self.encoding.encode(table_text)),
                'char_count': len(table_text),
                'start_page': table['page'],
                'table_metadata': {
                    'shape': table['shape'],
                    'method': table['method'],
                    'accuracy': table.get('accuracy')
                }
            }
            
            table_chunks.append(chunk)
        
        return table_chunks
    
    def _save_document(self, document: Dict):
        """
        Save processed document to disk
        
        Args:
            document: Document dictionary
        """
        # Save full document as JSON
        doc_path = DATA_PROCESSED_PATH / f"{document['doc_id']}.json"
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(doc_path, 'w') as f:
            json.dump(document, f, indent=2)
        
        # Save metadata separately
        metadata = {
            'doc_id': document['doc_id'],
            'ticker': document['ticker'],
            'fiscal_year': document['fiscal_year'],
            'filing_type': document['filing_type'],
            'total_chunks': document['total_chunks'],
            'sections': document['sections'],
            'num_tables': document['num_tables']
        }
        
        metadata_path = DATA_METADATA_PATH / f"{document['doc_id']}_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save chunks as JSONL for easy loading
        chunks_path = DATA_PROCESSED_PATH / f"{document['doc_id']}_chunks.jsonl"
        
        with open(chunks_path, 'w') as f:
            for chunk in document['chunks']:
                f.write(json.dumps(chunk) + '\n')
        
        logger.info(f"Saved document to {doc_path}")
    
    def load_document(self, doc_id: str) -> Optional[Dict]:
        """
        Load processed document from disk
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document dictionary or None if not found
        """
        doc_path = DATA_PROCESSED_PATH / f"{doc_id}.json"
        
        if not doc_path.exists():
            logger.warning(f"Document not found: {doc_id}")
            return None
        
        with open(doc_path, 'r') as f:
            return json.load(f)
    
    def load_chunks(self, doc_id: str) -> List[Dict]:
        """
        Load document chunks from JSONL file
        
        Args:
            doc_id: Document identifier
            
        Returns:
            List of chunk dictionaries
        """
        chunks_path = DATA_PROCESSED_PATH / f"{doc_id}_chunks.jsonl"
        
        if not chunks_path.exists():
            logger.warning(f"Chunks file not found: {doc_id}")
            return []
        
        chunks = []
        with open(chunks_path, 'r') as f:
            for line in f:
                chunks.append(json.loads(line))
        
        return chunks


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    preprocessor = DocumentPreprocessor()
    
    # Example: Process a filing
    # pages_data = {...}  # From parser
    # sections = {...}  # From parser
    # tables = [...]  # From table extractor
    
    # document = preprocessor.process_filing(
    #     ticker="AAPL",
    #     fiscal_year=2023,
    #     pages_data=pages_data,
    #     sections=sections,
    #     tables=tables
    # )
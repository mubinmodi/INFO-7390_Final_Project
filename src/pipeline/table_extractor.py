"""
Table extraction from PDFs using Camelot and pdfplumber
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import camelot
import pdfplumber

logger = logging.getLogger(__name__)


class TableExtractor:
    """Extract tables from PDF files"""
    
    def __init__(self):
        self.extraction_methods = ['camelot_lattice', 'camelot_stream', 'pdfplumber']
        
    def extract_tables(self, 
                      pdf_path: Path, 
                      pages: Optional[str] = 'all',
                      method: str = 'auto') -> List[Dict]:
        """
        Extract tables from PDF using multiple methods
        
        Args:
            pdf_path: Path to PDF file
            pages: Page numbers (e.g., '1-5', 'all')
            method: Extraction method ('auto', 'camelot_lattice', 'camelot_stream', 'pdfplumber')
            
        Returns:
            List of table dictionaries with metadata
        """
        logger.info(f"Extracting tables from {pdf_path.name} using method: {method}")
        
        if method == 'auto':
            # Try multiple methods and combine results
            tables = self._extract_auto(pdf_path, pages)
        elif method == 'camelot_lattice':
            tables = self._extract_camelot_lattice(pdf_path, pages)
        elif method == 'camelot_stream':
            tables = self._extract_camelot_stream(pdf_path, pages)
        elif method == 'pdfplumber':
            tables = self._extract_pdfplumber(pdf_path, pages)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Extracted {len(tables)} tables")
        return tables
    
    def _extract_auto(self, pdf_path: Path, pages: str) -> List[Dict]:
        """
        Automatically select best extraction method
        
        Args:
            pdf_path: Path to PDF file
            pages: Page numbers
            
        Returns:
            List of extracted tables
        """
        all_tables = []
        
        # Try Camelot lattice first (best for bordered tables)
        try:
            lattice_tables = self._extract_camelot_lattice(pdf_path, pages)
            all_tables.extend(lattice_tables)
        except Exception as e:
            logger.warning(f"Camelot lattice failed: {e}")
        
        # Try Camelot stream for borderless tables
        try:
            stream_tables = self._extract_camelot_stream(pdf_path, pages)
            # Filter out duplicates based on location
            stream_tables = self._filter_duplicates(all_tables, stream_tables)
            all_tables.extend(stream_tables)
        except Exception as e:
            logger.warning(f"Camelot stream failed: {e}")
        
        # Try pdfplumber as fallback
        if len(all_tables) == 0:
            try:
                pdfplumber_tables = self._extract_pdfplumber(pdf_path, pages)
                all_tables.extend(pdfplumber_tables)
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}")
        
        return all_tables
    
    def _extract_camelot_lattice(self, pdf_path: Path, pages: str) -> List[Dict]:
        """
        Extract tables using Camelot lattice method (for bordered tables)
        
        Args:
            pdf_path: Path to PDF file
            pages: Page numbers
            
        Returns:
            List of extracted tables
        """
        tables = []
        
        try:
            camelot_tables = camelot.read_pdf(
                str(pdf_path),
                pages=pages,
                flavor='lattice',
                line_scale=40
            )
            
            for idx, table in enumerate(camelot_tables):
                tables.append({
                    'table_id': f"lattice_{idx}",
                    'page': table.page,
                    'method': 'camelot_lattice',
                    'accuracy': table.accuracy,
                    'bbox': table._bbox,
                    'dataframe': table.df,
                    'data': table.df.to_dict('records'),
                    'shape': table.df.shape
                })
        except Exception as e:
            logger.error(f"Camelot lattice extraction error: {e}")
            raise
        
        return tables
    
    def _extract_camelot_stream(self, pdf_path: Path, pages: str) -> List[Dict]:
        """
        Extract tables using Camelot stream method (for borderless tables)
        
        Args:
            pdf_path: Path to PDF file
            pages: Page numbers
            
        Returns:
            List of extracted tables
        """
        tables = []
        
        try:
            camelot_tables = camelot.read_pdf(
                str(pdf_path),
                pages=pages,
                flavor='stream',
                edge_tol=50
            )
            
            for idx, table in enumerate(camelot_tables):
                tables.append({
                    'table_id': f"stream_{idx}",
                    'page': table.page,
                    'method': 'camelot_stream',
                    'accuracy': table.accuracy,
                    'bbox': table._bbox,
                    'dataframe': table.df,
                    'data': table.df.to_dict('records'),
                    'shape': table.df.shape
                })
        except Exception as e:
            logger.error(f"Camelot stream extraction error: {e}")
            raise
        
        return tables
    
    def _extract_pdfplumber(self, pdf_path: Path, pages: str) -> List[Dict]:
        """
        Extract tables using pdfplumber
        
        Args:
            pdf_path: Path to PDF file
            pages: Page numbers
            
        Returns:
            List of extracted tables
        """
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Parse page range
                if pages == 'all':
                    page_nums = range(len(pdf.pages))
                else:
                    # Parse page range like '1-5' or '1,3,5'
                    page_nums = self._parse_page_range(pages)
                
                for page_num in page_nums:
                    if page_num >= len(pdf.pages):
                        continue
                    
                    page = pdf.pages[page_num]
                    page_tables = page.extract_tables()
                    
                    for idx, table_data in enumerate(page_tables):
                        # Convert to DataFrame
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        
                        tables.append({
                            'table_id': f"pdfplumber_{page_num}_{idx}",
                            'page': page_num + 1,
                            'method': 'pdfplumber',
                            'accuracy': None,
                            'bbox': None,
                            'dataframe': df,
                            'data': df.to_dict('records'),
                            'shape': df.shape
                        })
        except Exception as e:
            logger.error(f"pdfplumber extraction error: {e}")
            raise
        
        return tables
    
    def _parse_page_range(self, pages: str) -> List[int]:
        """
        Parse page range string
        
        Args:
            pages: Page range string (e.g., '1-5', '1,3,5')
            
        Returns:
            List of page numbers (0-indexed)
        """
        page_nums = []
        
        if ',' in pages:
            # Handle comma-separated pages
            for page in pages.split(','):
                page = page.strip()
                if '-' in page:
                    start, end = map(int, page.split('-'))
                    page_nums.extend(range(start - 1, end))
                else:
                    page_nums.append(int(page) - 1)
        elif '-' in pages:
            # Handle page range
            start, end = map(int, pages.split('-'))
            page_nums = list(range(start - 1, end))
        else:
            # Single page
            page_nums = [int(pages) - 1]
        
        return page_nums
    
    def _filter_duplicates(self, existing_tables: List[Dict], new_tables: List[Dict]) -> List[Dict]:
        """
        Filter duplicate tables based on location
        
        Args:
            existing_tables: Previously extracted tables
            new_tables: Newly extracted tables
            
        Returns:
            Filtered list of new tables
        """
        filtered = []
        
        for new_table in new_tables:
            is_duplicate = False
            
            for existing_table in existing_tables:
                # Check if on same page and similar location
                if (new_table['page'] == existing_table['page'] and
                    new_table.get('bbox') and existing_table.get('bbox')):
                    
                    # Calculate overlap
                    overlap = self._bbox_overlap(new_table['bbox'], existing_table['bbox'])
                    if overlap > 0.5:  # 50% overlap threshold
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(new_table)
        
        return filtered
    
    def _bbox_overlap(self, bbox1: tuple, bbox2: tuple) -> float:
        """
        Calculate overlap ratio between two bounding boxes
        
        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        return intersection_area / min(bbox1_area, bbox2_area)
    
    def save_tables(self, tables: List[Dict], output_dir: Path):
        """
        Save extracted tables to CSV files
        
        Args:
            tables: List of table dictionaries
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for table in tables:
            filename = f"{table['table_id']}_page{table['page']}.csv"
            filepath = output_dir / filename
            
            table['dataframe'].to_csv(filepath, index=False)
            logger.info(f"Saved table to {filepath}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    extractor = TableExtractor()
    
    # Example: Extract tables from a PDF
    # pdf_path = Path("data/raw/sec-edgar-filings/AAPL/10-K/0000320193-23-000077/filing.pdf")
    # tables = extractor.extract_tables(pdf_path, pages='50-100')
    # extractor.save_tables(tables, Path("data/processed/tables"))
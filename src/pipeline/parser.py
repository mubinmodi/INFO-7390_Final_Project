"""
PDF Parser with pdfplumber and OCR fallback
"""
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pdfplumber
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)


class PDFParser:
    """Parse PDF files with text extraction and OCR fallback"""
    
    def __init__(self):
        self.current_file = None
        
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[int, Dict]:
        """
        Extract text from PDF with word-level bounding boxes
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary mapping page numbers to page content
        """
        self.current_file = pdf_path
        logger.info(f"Parsing PDF: {pdf_path}")
        
        pages_data = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_data = self._extract_page_content(page, page_num)
                    pages_data[page_num] = page_data
                    
            logger.info(f"Successfully parsed {len(pages_data)} pages")
            return pages_data
            
        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {e}")
            raise
    
    def _extract_page_content(self, page, page_num: int) -> Dict:
        """
        Extract content from a single page
        
        Args:
            page: pdfplumber page object
            page_num: Page number
            
        Returns:
            Dictionary with page content
        """
        # Extract text
        text = page.extract_text() or ""
        
        # Extract words with bounding boxes
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False
        )
        
        # If text extraction failed, try OCR
        if not text.strip() and len(words) == 0:
            logger.warning(f"Page {page_num}: Text extraction failed, trying OCR")
            text, words = self._ocr_page(page)
        
        # Extract images
        images = []
        if page.images:
            images = [
                {
                    "bbox": [img["x0"], img["top"], img["x1"], img["bottom"]],
                    "width": img["width"],
                    "height": img["height"]
                }
                for img in page.images
            ]
        
        return {
            "page_num": page_num,
            "text": text,
            "words": words,
            "images": images,
            "width": page.width,
            "height": page.height
        }
    
    def _ocr_page(self, page) -> Tuple[str, List[Dict]]:
        """
        Perform OCR on a page
        
        Args:
            page: pdfplumber page object
            
        Returns:
            Tuple of (text, words with bounding boxes)
        """
        try:
            # Convert page to image
            image = page.to_image(resolution=300)
            pil_image = image.original
            
            # Perform OCR
            text = pytesseract.image_to_string(pil_image)
            
            # Get word-level bounding boxes
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            words = []
            for i, word in enumerate(data['text']):
                if word.strip():
                    words.append({
                        'text': word,
                        'x0': data['left'][i],
                        'top': data['top'][i],
                        'x1': data['left'][i] + data['width'][i],
                        'bottom': data['top'][i] + data['height'][i]
                    })
            
            return text, words
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return "", []
    
    def identify_sections(self, pages_data: Dict[int, Dict]) -> Dict[str, List[int]]:
        """
        Identify SEC 10-K sections based on common patterns
        
        Args:
            pages_data: Dictionary of page data
            
        Returns:
            Dictionary mapping section names to page ranges
        """
        sections = {}
        current_section = None
        
        # Common section patterns for 10-K
        section_patterns = {
            "ITEM_1": [
                r"ITEM\s+1\.?\s*BUSINESS",
                r"Item\s+1\.?\s*Business"
            ],
            "ITEM_1A": [
                r"ITEM\s+1A\.?\s*RISK FACTORS",
                r"Item\s+1A\.?\s*Risk Factors"
            ],
            "ITEM_7": [
                r"ITEM\s+7\.?\s*MANAGEMENT'?S DISCUSSION",
                r"Item\s+7\.?\s*Management'?s Discussion"
            ],
            "ITEM_8": [
                r"ITEM\s+8\.?\s*FINANCIAL STATEMENTS",
                r"Item\s+8\.?\s*Financial Statements"
            ],
            "ITEM_9A": [
                r"ITEM\s+9A\.?\s*CONTROLS AND PROCEDURES",
                r"Item\s+9A\.?\s*Controls and Procedures"
            ]
        }
        
        for page_num, page_data in sorted(pages_data.items()):
            text = page_data['text']
            
            # Check for section headers
            for section_id, patterns in section_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        if section_id not in sections:
                            sections[section_id] = []
                        sections[section_id].append(page_num)
                        current_section = section_id
                        logger.info(f"Found {section_id} on page {page_num}")
                        break
            
            # Assign pages to current section
            if current_section and current_section in sections:
                if page_num not in sections[current_section]:
                    sections[current_section].append(page_num)
        
        return sections
    
    def extract_section_text(self, pages_data: Dict[int, Dict], section_pages: List[int]) -> str:
        """
        Extract text from a specific section
        
        Args:
            pages_data: Dictionary of page data
            section_pages: List of page numbers in the section
            
        Returns:
            Combined text from all pages in the section
        """
        section_text = []
        
        for page_num in sorted(section_pages):
            if page_num in pages_data:
                section_text.append(pages_data[page_num]['text'])
        
        return "\n\n".join(section_text)
    
    def get_word_boxes(self, pages_data: Dict[int, Dict], page_num: int) -> List[Dict]:
        """
        Get word-level bounding boxes for a specific page
        
        Args:
            pages_data: Dictionary of page data
            page_num: Page number
            
        Returns:
            List of word dictionaries with bounding boxes
        """
        if page_num in pages_data:
            return pages_data[page_num].get('words', [])
        return []


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    parser = PDFParser()
    
    # Example: Parse a PDF file
    # pdf_path = Path("data/raw/sec-edgar-filings/AAPL/10-K/0000320193-23-000077/filing.pdf")
    # pages_data = parser.extract_text_from_pdf(pdf_path)
    # sections = parser.identify_sections(pages_data)
    
    # print(f"Found {len(sections)} sections")
    # for section_id, pages in sections.items():
    #     print(f"{section_id}: pages {min(pages)}-{max(pages)}")
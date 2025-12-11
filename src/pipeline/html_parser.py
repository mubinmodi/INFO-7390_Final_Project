"""
HTML Parser for SEC filings
"""
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
import html2text

logger = logging.getLogger(__name__)


class HTMLParser:
    """Parse HTML SEC filings and extract text content"""
    
    def __init__(self):
        self.current_file = None
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = True
        self.h2t.ignore_emphasis = False
        
    def extract_text_from_html(self, html_path: Path) -> Dict[int, Dict]:
        """
        Extract text from HTML filing
        
        Args:
            html_path: Path to HTML file
            
        Returns:
            Dictionary mapping "page numbers" (sections) to content
        """
        self.current_file = html_path
        logger.info(f"Parsing HTML: {html_path}")
        
        try:
            with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # For HTML files, we'll treat the entire document as one "page"
            # but we'll split it into logical sections later
            pages_data = {
                1: {
                    'page_num': 1,
                    'text': text,
                    'word_count': len(text.split()),
                    'char_count': len(text)
                }
            }
            
            logger.info(f"Successfully parsed HTML: {len(text)} characters, {len(text.split())} words")
            return pages_data
            
        except Exception as e:
            logger.error(f"Error parsing HTML {html_path}: {e}")
            raise
    
    def identify_sections(self, pages_data: Dict[int, Dict]) -> Dict[str, Dict]:
        """
        Identify major sections in the document using regex patterns
        
        Args:
            pages_data: Dictionary of page data
            
        Returns:
            Dictionary mapping section IDs to section info
        """
        # Combine all text
        full_text = '\n'.join(page['text'] for page in pages_data.values())
        
        sections = {}
        
        # Section patterns for 10-K filings
        section_patterns = {
            'item_1': r'ITEM\s+1[.\s]+BUSINESS',
            'item_1a': r'ITEM\s+1A[.\s]+RISK\s+FACTORS',
            'item_1b': r'ITEM\s+1B[.\s]+UNRESOLVED\s+STAFF\s+COMMENTS',
            'item_2': r'ITEM\s+2[.\s]+PROPERTIES',
            'item_3': r'ITEM\s+3[.\s]+LEGAL\s+PROCEEDINGS',
            'item_4': r'ITEM\s+4[.\s]+MINE\s+SAFETY',
            'item_5': r'ITEM\s+5[.\s]+MARKET\s+FOR\s+REGISTRANT',
            'item_6': r'ITEM\s+6[.\s]+\[?RESERVED\]?|SELECTED\s+FINANCIAL\s+DATA',
            'item_7': r'ITEM\s+7[.\s]+MANAGEMENT.?S\s+DISCUSSION\s+AND\s+ANALYSIS',
            'item_7a': r'ITEM\s+7A[.\s]+QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES',
            'item_8': r'ITEM\s+8[.\s]+FINANCIAL\s+STATEMENTS\s+AND\s+SUPPLEMENTARY\s+DATA',
            'item_9': r'ITEM\s+9[.\s]+CHANGES\s+IN\s+AND\s+DISAGREEMENTS',
            'item_9a': r'ITEM\s+9A[.\s]+CONTROLS\s+AND\s+PROCEDURES',
            'item_9b': r'ITEM\s+9B[.\s]+OTHER\s+INFORMATION',
            'item_10': r'ITEM\s+10[.\s]+DIRECTORS.?\s+EXECUTIVE\s+OFFICERS',
            'item_11': r'ITEM\s+11[.\s]+EXECUTIVE\s+COMPENSATION',
            'item_12': r'ITEM\s+12[.\s]+SECURITY\s+OWNERSHIP',
            'item_13': r'ITEM\s+13[.\s]+CERTAIN\s+RELATIONSHIPS',
            'item_14': r'ITEM\s+14[.\s]+PRINCIPAL\s+ACCOUNTANT',
            'item_15': r'ITEM\s+15[.\s]+EXHIBITS.?\s+FINANCIAL\s+STATEMENT\s+SCHEDULES',
        }
        
        for section_id, pattern in section_patterns.items():
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                start_pos = match.start()
                sections[section_id] = {
                    'section_id': section_id,
                    'title': match.group(),
                    'start_pos': start_pos,
                    'page_num': 1  # All in same "page" for HTML
                }
                logger.debug(f"Found {section_id}: {match.group()}")
        
        # Calculate end positions
        section_list = sorted(sections.items(), key=lambda x: x[1]['start_pos'])
        for i, (section_id, section_info) in enumerate(section_list):
            if i < len(section_list) - 1:
                next_start = section_list[i + 1][1]['start_pos']
                section_info['end_pos'] = next_start
                section_info['text'] = full_text[section_info['start_pos']:next_start]
            else:
                section_info['end_pos'] = len(full_text)
                section_info['text'] = full_text[section_info['start_pos']:]
            
            section_info['char_count'] = len(section_info['text'])
            section_info['word_count'] = len(section_info['text'].split())
        
        logger.info(f"Identified {len(sections)} sections")
        return sections
    
    def extract_section_text(self, 
                           pages_data: Dict[int, Dict],
                           section_info: Dict) -> str:
        """
        Extract text for a specific section
        
        Args:
            pages_data: Dictionary of page data
            section_info: Section information from identify_sections
            
        Returns:
            Section text as string
        """
        return section_info.get('text', '')
    
    def get_section_summary(self, sections: Dict[str, Dict]) -> str:
        """
        Generate a summary of identified sections
        
        Args:
            sections: Dictionary of section information
            
        Returns:
            Summary string
        """
        if not sections:
            return "No sections identified"
        
        summary_lines = ["Identified Sections:"]
        for section_id, info in sorted(sections.items(), key=lambda x: x[1]['start_pos']):
            word_count = info.get('word_count', 0)
            summary_lines.append(f"  {section_id}: {info['title']} ({word_count:,} words)")
        
        return '\n'.join(summary_lines)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    parser = HTMLParser()
    
    # Test with a sample HTML file
    # html_path = Path("data/raw/sec-edgar-filings/AAPL/10-K/0000320193-23-000106/primary-document.html")
    # pages_data = parser.extract_text_from_html(html_path)
    # sections = parser.identify_sections(pages_data)
    # print(parser.get_section_summary(sections))
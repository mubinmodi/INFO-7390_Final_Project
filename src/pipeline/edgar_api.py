"""
SEC EDGAR API Client - Fetch filings directly via API
"""
import requests
import logging
import re
from typing import Dict, Optional, List
from datetime import datetime
from bs4 import BeautifulSoup
import time

logger = logging.getLogger(__name__)


class EdgarAPIClient:
    """Fetch SEC filings directly via EDGAR API"""
    
    BASE_URL = "https://data.sec.gov"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions"
    
    def __init__(self, user_agent: str):
        """
        Initialize EDGAR API client
        
        Args:
            user_agent: User agent string (required by SEC, format: "Name email@domain.com")
        """
        self.headers = {
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def get_company_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK number for a ticker symbol
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            CIK string (10 digits, zero-padded) or None
        """
        ticker = ticker.upper().strip()
        
        try:
            # Get ticker to CIK mapping
            url = "https://www.sec.gov/files/company_tickers.json"
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            
            tickers_data = response.json()
            
            # Search for ticker
            for item in tickers_data.values():
                if item['ticker'] == ticker:
                    cik = str(item['cik_str']).zfill(10)
                    logger.info(f"Found CIK {cik} for ticker {ticker}")
                    return cik
            
            logger.error(f"Ticker {ticker} not found")
            return None
            
        except Exception as e:
            logger.error(f"Error getting CIK for {ticker}: {e}")
            return None
    
    def get_recent_filings(self, cik: str, filing_type: str = "10-K", count: int = 1) -> List[Dict]:
        """
        Get recent filings for a company
        
        Args:
            cik: Company CIK number
            filing_type: Type of filing (10-K, 10-Q, etc.)
            count: Number of filings to retrieve
            
        Returns:
            List of filing metadata dictionaries
        """
        try:
            # Format CIK (remove leading zeros for URL, but keep for reference)
            cik_formatted = cik.lstrip('0') or '0'
            
            url = f"{self.SUBMISSIONS_URL}/CIK{cik}.json"
            logger.info(f"Fetching filings from {url}")
            
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract recent filings
            filings = data.get('filings', {}).get('recent', {})
            
            # Find matching filings
            results = []
            for i in range(len(filings.get('form', []))):
                form = filings['form'][i]
                if form == filing_type:
                    filing_info = {
                        'accession_number': filings['accessionNumber'][i],
                        'filing_date': filings['filingDate'][i],
                        'report_date': filings['reportDate'][i],
                        'form': form,
                        'primary_document': filings['primaryDocument'][i],
                        'cik': cik
                    }
                    results.append(filing_info)
                    
                    if len(results) >= count:
                        break
            
            logger.info(f"Found {len(results)} {filing_type} filing(s)")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching filings for CIK {cik}: {e}")
            return []
    
    def get_filing_text(self, filing_info: Dict) -> Optional[str]:
        """
        Fetch the full text of a filing
        
        Args:
            filing_info: Filing metadata from get_recent_filings
            
        Returns:
            Filing text content or None
        """
        try:
            # Build URL
            accession = filing_info['accession_number'].replace('-', '')
            primary_doc = filing_info['primary_document']
            cik = filing_info['cik'].lstrip('0') or '0'
            
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"
            
            logger.info(f"Fetching filing from {url}")
            
            # Rate limiting - SEC allows 10 requests per second
            time.sleep(0.11)
            
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Parse HTML and extract text
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style tags
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            logger.info(f"Successfully fetched filing: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Error fetching filing text: {e}")
            return None
    
    def identify_sections(self, text: str) -> Dict[str, Dict]:
        """
        Identify major sections in a 10-K filing
        
        Args:
            text: Full text of filing
            
        Returns:
            Dictionary mapping section IDs to section info
        """
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
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_pos = match.start()
                sections[section_id] = {
                    'section_id': section_id,
                    'title': match.group(),
                    'start_pos': start_pos,
                    'page_num': 1
                }
        
        # Calculate end positions and extract text
        section_list = sorted(sections.items(), key=lambda x: x[1]['start_pos'])
        for i, (section_id, section_info) in enumerate(section_list):
            if i < len(section_list) - 1:
                next_start = section_list[i + 1][1]['start_pos']
                section_info['end_pos'] = next_start
                section_info['text'] = text[section_info['start_pos']:next_start]
            else:
                section_info['end_pos'] = len(text)
                section_info['text'] = text[section_info['start_pos']:]
            
            section_info['char_count'] = len(section_info['text'])
            section_info['word_count'] = len(section_info['text'].split())
        
        logger.info(f"Identified {len(sections)} sections")
        return sections
    
    def get_10k_for_ticker(self, ticker: str, year: Optional[int] = None) -> Optional[Dict]:
        """
        Get 10-K filing for a ticker (convenience method)
        
        Args:
            ticker: Company ticker symbol
            year: Optional fiscal year (if None, gets most recent)
            
        Returns:
            Dictionary with filing text and sections
        """
        # Get CIK
        cik = self.get_company_cik(ticker)
        if not cik:
            return None
        
        # Get recent 10-K filings
        filings = self.get_recent_filings(cik, "10-K", count=5)
        if not filings:
            return None
        
        # Filter by year if specified
        if year:
            for filing in filings:
                filing_year = int(filing['report_date'].split('-')[0])
                if filing_year == year:
                    selected_filing = filing
                    break
            else:
                logger.error(f"No 10-K found for {ticker} in {year}")
                return None
        else:
            selected_filing = filings[0]
        
        # Fetch text
        text = self.get_filing_text(selected_filing)
        if not text:
            return None
        
        # Identify sections
        sections = self.identify_sections(text)
        
        return {
            'ticker': ticker,
            'cik': cik,
            'filing_info': selected_filing,
            'text': text,
            'sections': sections,
            'filing_date': selected_filing['filing_date'],
            'report_date': selected_filing['report_date']
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    client = EdgarAPIClient("MyCompany contact@example.com")
    
    # Get Apple's latest 10-K
    result = client.get_10k_for_ticker("AAPL", year=2023)
    
    if result:
        print(f"Filing Date: {result['filing_date']}")
        print(f"Report Date: {result['report_date']}")
        print(f"Sections found: {len(result['sections'])}")
        print(f"Total text length: {len(result['text'])} characters")
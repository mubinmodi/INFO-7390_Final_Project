"""
SEC Filing Downloader using sec-edgar-downloader
"""
import os
import logging
from pathlib import Path
from typing import List, Optional
from sec_edgar_downloader import Downloader
from config.settings import DATA_RAW_PATH, SEC_USER_AGENT

logger = logging.getLogger(__name__)


class SECDownloader:
    """Download SEC filings for specified companies"""
    
    def __init__(self, download_path: Optional[Path] = None):
        """
        Initialize SEC downloader
        
        Args:
            download_path: Path to save downloaded filings
        """
        self.download_path = download_path or DATA_RAW_PATH
        self.download_path.mkdir(parents=True, exist_ok=True)
        
        # Extract company name and email from user agent
        # Format: "CompanyName email@domain.com"
        parts = SEC_USER_AGENT.split()
        company_name = parts[0] if len(parts) > 0 else "Research"
        email = parts[1] if len(parts) > 1 else "contact@example.com"
        
        self.downloader = Downloader(
            company_name=company_name,
            email_address=email,
            download_folder=str(self.download_path)
        )
        
    def download_10k(self, 
                     ticker: str, 
                     num_filings: int = 1,
                     after_date: Optional[str] = None,
                     before_date: Optional[str] = None) -> Path:
        """
        Download 10-K filings for a ticker
        
        Args:
            ticker: Company ticker symbol
            num_filings: Number of filings to download
            after_date: Download filings after this date (YYYY-MM-DD)
            before_date: Download filings before this date (YYYY-MM-DD)
            
        Returns:
            Path to downloaded files
        """
        logger.info(f"Downloading {num_filings} 10-K filing(s) for {ticker}")
        
        try:
            self.downloader.get(
                "10-K",
                ticker,
                limit=num_filings,
                after=after_date,
                before=before_date,
                download_details=True  # Include XBRL and exhibits
            )
            
            ticker_path = self.download_path / "sec-edgar-filings" / ticker / "10-K"
            logger.info(f"Successfully downloaded to {ticker_path}")
            return ticker_path
            
        except Exception as e:
            logger.error(f"Error downloading 10-K for {ticker}: {e}")
            raise
    
    def download_10q(self,
                     ticker: str,
                     num_filings: int = 1,
                     after_date: Optional[str] = None,
                     before_date: Optional[str] = None) -> Path:
        """
        Download 10-Q filings for a ticker
        
        Args:
            ticker: Company ticker symbol
            num_filings: Number of filings to download
            after_date: Download filings after this date (YYYY-MM-DD)
            before_date: Download filings before this date (YYYY-MM-DD)
            
        Returns:
            Path to downloaded files
        """
        logger.info(f"Downloading {num_filings} 10-Q filing(s) for {ticker}")
        
        try:
            self.downloader.get(
                "10-Q",
                ticker,
                limit=num_filings,
                after=after_date,
                before=before_date,
                download_details=True
            )
            
            ticker_path = self.download_path / "sec-edgar-filings" / ticker / "10-Q"
            logger.info(f"Successfully downloaded to {ticker_path}")
            return ticker_path
            
        except Exception as e:
            logger.error(f"Error downloading 10-Q for {ticker}: {e}")
            raise
    
    def list_downloaded_filings(self, ticker: str) -> List[Path]:
        """
        List all downloaded filings for a ticker
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            List of filing directories
        """
        ticker_path = self.download_path / "sec-edgar-filings" / ticker
        
        if not ticker_path.exists():
            return []
        
        filings = []
        for filing_type in ticker_path.iterdir():
            if filing_type.is_dir():
                for filing in filing_type.iterdir():
                    if filing.is_dir():
                        filings.append(filing)
        
        return sorted(filings, reverse=True)  # Most recent first
    
    def get_filing_path(self, ticker: str, filing_type: str = "10-K", index: int = 0) -> Optional[Path]:
        """
        Get path to a specific filing
        
        Args:
            ticker: Company ticker symbol
            filing_type: Type of filing (10-K, 10-Q, etc.)
            index: Index of filing (0 = most recent)
            
        Returns:
            Path to filing directory or None if not found
        """
        filing_path = self.download_path / "sec-edgar-filings" / ticker / filing_type
        
        if not filing_path.exists():
            return None
        
        filings = sorted([f for f in filing_path.iterdir() if f.is_dir()], reverse=True)
        
        if index >= len(filings):
            return None
        
        return filings[index]


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    downloader = SECDownloader()
    
    # Download latest 10-K for Apple
    path = downloader.download_10k("AAPL", num_filings=1)
    print(f"Downloaded to: {path}")
    
    # List all downloaded filings
    filings = downloader.list_downloaded_filings("AAPL")
    print(f"Found {len(filings)} filings")
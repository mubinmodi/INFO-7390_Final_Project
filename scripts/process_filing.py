"""
Script to process a single SEC filing through the complete pipeline
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.downloader import SECDownloader
from src.pipeline.parser import PDFParser
from src.pipeline.html_parser import HTMLParser
from src.pipeline.table_extractor import TableExtractor
from src.pipeline.preprocessor import DocumentPreprocessor
from src.vectordb.embeddings import EmbeddingGenerator
from src.vectordb.milvus_client import MilvusClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_filing(ticker: str, year: int, filing_type: str = "10-K"):
    """
    Process a SEC filing through the complete pipeline
    
    Args:
        ticker: Company ticker symbol
        year: Fiscal year
        filing_type: Type of filing (10-K or 10-Q)
    """
    logger.info(f"Processing {filing_type} for {ticker} - {year}")
    
    # Step 1: Download filing
    logger.info("Step 1/5: Downloading filing...")
    downloader = SECDownloader()
    
    if filing_type == "10-K":
        filing_path = downloader.download_10k(ticker, num_filings=1)
    else:
        filing_path = downloader.download_10q(ticker, num_filings=1)
    
    logger.info(f"Downloaded to: {filing_path}")
    
    # Step 2: Parse filing (HTML or PDF)
    logger.info("Step 2/5: Parsing filing...")
    
    # Find HTML or PDF files
    html_files = list(filing_path.rglob("*.html")) + list(filing_path.rglob("*.htm"))
    pdf_files = list(filing_path.rglob("*.pdf"))
    
    # Prefer primary-document.html or full-submission.txt for 10-K/10-Q
    primary_html = None
    for html_file in html_files:
        if 'primary-document' in html_file.name.lower() or 'full-submission' in html_file.name.lower():
            primary_html = html_file
            break
    
    if primary_html:
        # Parse HTML filing
        logger.info(f"Parsing HTML: {primary_html.name}")
        html_parser = HTMLParser()
        pages_data = html_parser.extract_text_from_html(primary_html)
        sections = html_parser.identify_sections(pages_data)
        tables = []  # HTML parsing - tables embedded in text
        
    elif pdf_files:
        # Parse PDF filing
        pdf_path = pdf_files[0]
        logger.info(f"Parsing PDF: {pdf_path.name}")
        
        pdf_parser = PDFParser()
        pages_data = pdf_parser.extract_text_from_pdf(pdf_path)
        sections = pdf_parser.identify_sections(pages_data)
        
        # Extract tables from PDF
        logger.info("Step 2b/5: Extracting tables from PDF...")
        extractor = TableExtractor()
        total_pages = len(pages_data)
        start_page = total_pages // 2
        pages_str = f"{start_page}-{total_pages}"
        
        try:
            tables = extractor.extract_tables(pdf_path, pages=pages_str, method='auto')
            logger.info(f"Extracted {len(tables)} tables")
        except Exception as e:
            logger.warning(f"Table extraction had issues: {e}")
            tables = []
    else:
        logger.error("No HTML or PDF files found in filing")
        return False
    
    logger.info(f"Parsed {len(pages_data)} pages/sections, found {len(sections)} identified sections")
    
    # Step 3: Preprocess and chunk
    logger.info("Step 3/5: Preprocessing document...")
    preprocessor = DocumentPreprocessor()
    
    document = preprocessor.process_filing(
        ticker=ticker,
        fiscal_year=year,
        pages_data=pages_data,
        sections=sections,
        tables=tables
    )
    
    logger.info(f"Created {document['total_chunks']} chunks")
    
    # Step 4: Generate embeddings and index
    logger.info("Step 4/5: Generating embeddings and indexing...")
    embedding_gen = EmbeddingGenerator()
    embeddings = embedding_gen.embed_chunks(document['chunks'])
    
    # Add metadata to chunks
    for chunk in document['chunks']:
        chunk['ticker'] = ticker
        chunk['fiscal_year'] = year
        chunk['doc_id'] = document['doc_id']
    
    # Index in Milvus
    milvus_client = MilvusClient()
    milvus_client.insert_chunks(document['chunks'], embeddings)
    
    logger.info(f"✅ Successfully processed and indexed {ticker} - {year}")
    logger.info(f"Document ID: {document['doc_id']}")
    
    milvus_client.close()
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Process a SEC filing')
    parser.add_argument('--ticker', required=True, help='Company ticker symbol')
    parser.add_argument('--year', type=int, required=True, help='Fiscal year')
    parser.add_argument('--filing-type', default='10-K', choices=['10-K', '10-Q'], 
                       help='Type of filing')
    
    args = parser.parse_args()
    
    try:
        success = process_filing(
            ticker=args.ticker.upper(),
            year=args.year,
            filing_type=args.filing_type
        )
        
        if success:
            logger.info("✅ Processing completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Processing failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
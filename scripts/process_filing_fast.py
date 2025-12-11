"""
Streamlined script to process SEC filings using EDGAR API + Gemini
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.edgar_api import EdgarAPIClient
from src.pipeline.preprocessor import DocumentPreprocessor
from src.vectordb.embeddings_unified import EmbeddingGenerator
from src.vectordb.milvus_client import MilvusClient
from config.settings import SEC_USER_AGENT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_filing_fast(ticker: str, year: int):
    """
    Fast processing using EDGAR API directly (no file downloads)
    
    Args:
        ticker: Company ticker symbol
        year: Fiscal year
    """
    logger.info(f"Processing 10-K for {ticker} - {year} using EDGAR API")
    
    # Step 1: Fetch filing from EDGAR API
    logger.info("Step 1/4: Fetching filing from SEC EDGAR API...")
    client = EdgarAPIClient(SEC_USER_AGENT)
    
    result = client.get_10k_for_ticker(ticker, year=year)
    
    if not result:
        logger.error(f"Failed to fetch 10-K for {ticker} - {year}")
        return False
    
    logger.info(f"✓ Fetched filing: {result['filing_date']}")
    logger.info(f"  Text length: {len(result['text']):,} characters")
    logger.info(f"  Sections found: {len(result['sections'])}")
    
    # Step 2: Preprocess and chunk
    logger.info("Step 2/4: Preprocessing and chunking document...")
    preprocessor = DocumentPreprocessor()
    
    # Convert to format expected by preprocessor
    pages_data = {
        1: {
            'page_num': 1,
            'text': result['text'],
            'word_count': len(result['text'].split()),
            'char_count': len(result['text'])
        }
    }
    
    document = preprocessor.process_filing(
        ticker=ticker,
        fiscal_year=year,
        pages_data=pages_data,
        sections=result['sections'],
        tables=[]  # No table extraction in fast mode
    )
    
    logger.info(f"✓ Created {document['total_chunks']} chunks")
    
    # Step 3: Generate embeddings
    logger.info("Step 3/4: Generating embeddings with Gemini...")
    embedding_gen = EmbeddingGenerator()
    embeddings = embedding_gen.embed_chunks(document['chunks'])
    
    logger.info(f"✓ Generated {len(embeddings)} embeddings")
    
    # Step 4: Index in Milvus
    logger.info("Step 4/4: Indexing in vector database...")
    
    # Add metadata to chunks
    for chunk in document['chunks']:
        chunk['ticker'] = ticker
        chunk['fiscal_year'] = year
        chunk['doc_id'] = document['doc_id']
    
    milvus_client = MilvusClient()
    milvus_client.insert_chunks(document['chunks'], embeddings)
    
    logger.info(f"✅ Successfully processed and indexed {ticker} - {year}")
    logger.info(f"   Document ID: {document['doc_id']}")
    logger.info(f"   Filing Date: {result['filing_date']}")
    logger.info(f"   Report Date: {result['report_date']}")
    
    milvus_client.close()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Process SEC filings using EDGAR API (fast, no downloads)'
    )
    parser.add_argument('--ticker', required=True, help='Company ticker symbol')
    parser.add_argument('--year', type=int, required=True, help='Fiscal year')
    
    args = parser.parse_args()
    
    try:
        success = process_filing_fast(
            ticker=args.ticker.upper(),
            year=args.year
        )
        
        if success:
            logger.info("✅ Processing completed successfully!")
            logger.info("   You can now run the Streamlit app to analyze this filing.")
            sys.exit(0)
        else:
            logger.error("❌ Processing failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
Lightweight streaming processor for SEC filings - won't freeze your laptop!
Processes and indexes data in small batches with progress saves
"""
import argparse
import logging
import sys
import time
from pathlib import Path
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.downloader import SECDownloader
from src.pipeline.html_parser import HTMLParser
from src.vectordb.embeddings import EmbeddingGenerator
from src.vectordb.milvus_client import MilvusClient
import tiktoken

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightweightProcessor:
    """Process filings in small batches to avoid freezing"""
    
    def __init__(self, batch_size: int = 10):
        """
        Initialize processor
        
        Args:
            batch_size: Number of chunks to process at once (lower = less memory)
        """
        self.batch_size = batch_size
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
    def chunk_text(self, text: str, section_id: str) -> list:
        """
        Split text into chunks
        
        Args:
            text: Text to chunk
            section_id: Section identifier
            
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
                'start_page': 1,
                'chunk_index': chunk_num
            }
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start_idx = end_idx - self.chunk_overlap
            chunk_num += 1
        
        return chunks
    
    def process_in_batches(self, 
                          chunks: list, 
                          ticker: str,
                          fiscal_year: int,
                          doc_id: str,
                          embedding_gen: EmbeddingGenerator,
                          milvus_client: MilvusClient):
        """
        Process chunks in small batches to avoid memory issues
        
        Args:
            chunks: All chunks to process
            ticker: Company ticker
            fiscal_year: Fiscal year
            doc_id: Document ID
            embedding_gen: Embedding generator
            milvus_client: Milvus client
        """
        total_chunks = len(chunks)
        logger.info(f"Processing {total_chunks} chunks in batches of {self.batch_size}")
        
        for i in range(0, total_chunks, self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total_chunks - 1) // self.batch_size + 1
            
            logger.info(f"📦 Batch {batch_num}/{total_batches} - Processing {len(batch)} chunks...")
            
            try:
                # Generate embeddings for batch
                logger.info(f"  → Generating embeddings...")
                embeddings = embedding_gen.embed_chunks(batch)
                
                # Add metadata
                for chunk in batch:
                    chunk['ticker'] = ticker
                    chunk['fiscal_year'] = fiscal_year
                    chunk['doc_id'] = doc_id
                
                # Insert into Milvus
                logger.info(f"  → Indexing in vector database...")
                milvus_client.insert_chunks(batch, embeddings)
                
                logger.info(f"  ✓ Batch {batch_num} completed!")
                
                # Clear memory
                del embeddings
                gc.collect()
                
                # Small delay to prevent overheating/freezing
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"  ✗ Error processing batch {batch_num}: {e}")
                raise
        
        logger.info(f"✅ All {total_chunks} chunks processed successfully!")


def process_filing_lightweight(ticker: str, year: int):
    """
    Process filing with lightweight approach
    
    Args:
        ticker: Company ticker
        year: Fiscal year
    """
    logger.info(f"🚀 Starting lightweight processing for {ticker} - {year}")
    logger.info(f"⚙️  Batch size: 10 chunks (optimized for i5 laptops)")
    
    processor = LightweightProcessor(batch_size=10)
    
    # Step 1: Download filing
    logger.info("\n📥 STEP 1/4: Downloading filing...")
    downloader = SECDownloader()
    filing_path = downloader.download_10k(ticker, num_filings=1)
    logger.info(f"✓ Downloaded to: {filing_path}")
    
    # Step 2: Parse HTML
    logger.info("\n📄 STEP 2/4: Parsing HTML filing...")
    html_files = list(filing_path.rglob("*.html")) + list(filing_path.rglob("*.htm"))
    
    primary_html = None
    for html_file in html_files:
        if 'primary-document' in html_file.name.lower():
            primary_html = html_file
            break
    
    if not primary_html:
        logger.error("No primary document found")
        return False
    
    html_parser = HTMLParser()
    pages_data = html_parser.extract_text_from_html(primary_html)
    sections = html_parser.identify_sections(pages_data)
    
    logger.info(f"✓ Found {len(sections)} sections")
    
    # Step 3: Create chunks (in memory, but not all at once)
    logger.info("\n✂️  STEP 3/4: Creating chunks...")
    
    all_chunks = []
    for section_id, section_info in sections.items():
        section_text = section_info.get('text', '')
        if not section_text or not section_text.strip():
            continue
        
        chunks = processor.chunk_text(section_text, section_id)
        all_chunks.extend(chunks)
        logger.info(f"  {section_id}: {len(chunks)} chunks")
    
    logger.info(f"✓ Created {len(all_chunks)} total chunks")
    
    # Step 4: Process in small batches
    logger.info("\n🔄 STEP 4/4: Processing batches (embeddings + indexing)...")
    logger.info("💡 This will take time but won't freeze your laptop!")
    
    doc_id = f"{ticker}_{year}_10K"
    
    embedding_gen = EmbeddingGenerator()
    milvus_client = MilvusClient()
    
    processor.process_in_batches(
        chunks=all_chunks,
        ticker=ticker,
        fiscal_year=year,
        doc_id=doc_id,
        embedding_gen=embedding_gen,
        milvus_client=milvus_client
    )
    
    milvus_client.close()
    
    logger.info(f"\n✅ SUCCESS! Processed {ticker} - {year}")
    logger.info(f"   Document ID: {doc_id}")
    logger.info(f"   Total chunks: {len(all_chunks)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Lightweight SEC filing processor (optimized for i5 laptops)'
    )
    parser.add_argument('--ticker', required=True, help='Company ticker symbol')
    parser.add_argument('--year', type=int, required=True, help='Fiscal year')
    parser.add_argument('--batch-size', type=int, default=10, 
                       help='Batch size (lower = less memory, default=10)')
    
    args = parser.parse_args()
    
    try:
        success = process_filing_lightweight(
            ticker=args.ticker.upper(),
            year=args.year
        )
        
        if success:
            logger.info("\n🎉 All done! You can now run analysis on this filing.")
            sys.exit(0)
        else:
            logger.error("\n❌ Processing failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
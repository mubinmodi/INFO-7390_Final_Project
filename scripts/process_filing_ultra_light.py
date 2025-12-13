"""
Ultra-lightweight streaming processor - processes section-by-section
Never stores all chunks in memory - perfect for low-memory systems
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


class UltraLightweightProcessor:
    """Process one section at a time - minimal memory usage"""
    
    def __init__(self, chunk_batch_size: int = 5):
        """
        Initialize processor
        
        Args:
            chunk_batch_size: Number of chunks to process at once (default: 5)
        """
        self.chunk_batch_size = chunk_batch_size
        self.chunk_size = 800  # Smaller chunks = less memory
        self.chunk_overlap = 150
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
    def process_section_streaming(self, 
                                  section_text: str,
                                  section_id: str,
                                  ticker: str,
                                  fiscal_year: int,
                                  doc_id: str,
                                  embedding_gen: EmbeddingGenerator,
                                  milvus_client: MilvusClient) -> int:
        """
        Process a single section in streaming fashion
        Creates chunks in small batches and indexes immediately
        
        Returns:
            Number of chunks processed
        """
        if not section_text or not section_text.strip():
            return 0
        
        # Tokenize section
        tokens = self.encoding.encode(section_text)
        total_tokens = len(tokens)
        
        chunk_num = 0
        start_idx = 0
        total_chunks_processed = 0
        
        # Process section in mini-batches
        while start_idx < total_tokens:
            mini_batch = []
            
            # Create a mini-batch of chunks
            for _ in range(self.chunk_batch_size):
                if start_idx >= total_tokens:
                    break
                
                # Get chunk tokens
                end_idx = min(start_idx + self.chunk_size, total_tokens)
                chunk_tokens = tokens[start_idx:end_idx]
                
                # Decode to text
                chunk_text = self.encoding.decode(chunk_tokens)
                
                # Create chunk
                chunk = {
                    'chunk_id': f"{section_id}_chunk_{chunk_num}",
                    'section_id': section_id,
                    'text': chunk_text,
                    'token_count': len(chunk_tokens),
                    'char_count': len(chunk_text),
                    'start_page': 1,
                    'chunk_index': chunk_num,
                    'ticker': ticker,
                    'fiscal_year': fiscal_year,
                    'doc_id': doc_id
                }
                
                mini_batch.append(chunk)
                
                # Move to next chunk
                start_idx = end_idx - self.chunk_overlap
                chunk_num += 1
            
            if not mini_batch:
                break
            
            # Process this mini-batch immediately
            try:
                # Generate embeddings
                embeddings = embedding_gen.embed_chunks(mini_batch)
                
                # Index immediately
                milvus_client.insert_chunks(mini_batch, embeddings)
                
                total_chunks_processed += len(mini_batch)
                
                # Clear memory
                del embeddings
                del mini_batch
                gc.collect()
                
                # Small delay
                time.sleep(0.3)
                
            except Exception as e:
                logger.error(f"Error processing mini-batch: {e}")
                raise
        
        return total_chunks_processed


def process_filing_ultra_light(ticker: str, year: int, chunk_batch_size: int = 5):
    """
    Ultra-lightweight processing - one section at a time
    
    Args:
        ticker: Company ticker
        year: Fiscal year  
        chunk_batch_size: Number of chunks per batch (lower = less memory)
    """
    logger.info(f"🚀 Ultra-lightweight processing for {ticker} - {year}")
    logger.info(f"⚙️  Chunk batch size: {chunk_batch_size} (optimized for low memory)")
    logger.info(f"💡 Processing sections one-by-one to minimize memory usage\n")
    
    processor = UltraLightweightProcessor(chunk_batch_size=chunk_batch_size)
    
    # Step 1: Download filing
    logger.info("📥 STEP 1/4: Downloading filing...")
    downloader = SECDownloader()
    filing_path = downloader.download_10k(ticker, num_filings=1)
    logger.info(f"✓ Downloaded\n")
    
    # Step 2: Parse HTML
    logger.info("📄 STEP 2/4: Parsing HTML filing...")
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
    
    logger.info(f"✓ Found {len(sections)} sections\n")
    
    # Step 3 & 4 Combined: Process each section immediately
    logger.info("🔄 STEP 3-4/4: Processing sections (streaming mode)...")
    logger.info("Each section is chunked → embedded → indexed immediately")
    logger.info("This prevents memory buildup!\n")
    
    doc_id = f"{ticker}_{year}_10K"
    
    embedding_gen = EmbeddingGenerator()
    milvus_client = MilvusClient()
    
    total_chunks = 0
    section_count = 0
    
    for section_id, section_info in sections.items():
        section_count += 1
        section_text = section_info.get('text', '')
        
        if not section_text or not section_text.strip():
            continue
        
        word_count = len(section_text.split())
        logger.info(f"📝 Section {section_count}/{len(sections)}: {section_id} ({word_count:,} words)")
        
        try:
            chunks_processed = processor.process_section_streaming(
                section_text=section_text,
                section_id=section_id,
                ticker=ticker,
                fiscal_year=year,
                doc_id=doc_id,
                embedding_gen=embedding_gen,
                milvus_client=milvus_client
            )
            
            total_chunks += chunks_processed
            logger.info(f"   ✓ Processed {chunks_processed} chunks\n")
            
        except Exception as e:
            logger.error(f"   ✗ Error processing {section_id}: {e}")
            raise
    
    milvus_client.close()
    
    logger.info(f"✅ SUCCESS! Processed {ticker} - {year}")
    logger.info(f"   Document ID: {doc_id}")
    logger.info(f"   Sections: {section_count}")
    logger.info(f"   Total chunks: {total_chunks}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Ultra-lightweight processor for low-memory systems'
    )
    parser.add_argument('--ticker', required=True, help='Company ticker symbol')
    parser.add_argument('--year', type=int, required=True, help='Fiscal year')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Chunks per batch (lower = less memory, default=5)')
    
    args = parser.parse_args()
    
    try:
        success = process_filing_ultra_light(
            ticker=args.ticker.upper(),
            year=args.year,
            chunk_batch_size=args.batch_size
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
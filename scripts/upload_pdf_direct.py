"""
Fast PDF Uploader - Process all chunks at once (no batching)
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pdfplumber
from src.vectordb.embeddings import EmbeddingGenerator
from src.vectordb.milvus_client import MilvusClient
import tiktoken
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FastPDFProcessor:
    """Process PDF fast - all at once"""
    
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract all text from PDF"""
        logger.info(f"📄 Extracting text from PDF...")
        
        full_text = []
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"   Total pages: {total_pages}")
            
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    full_text.append(text)
                
                if i % 50 == 0:
                    logger.info(f"   Extracted {i}/{total_pages} pages...")
        
        combined_text = '\n\n'.join(full_text)
        logger.info(f"✓ Extracted {len(combined_text):,} characters\n")
        
        return combined_text
    
    def identify_sections(self, text: str) -> dict:
        """Identify 10-K sections"""
        logger.info("🔍 Identifying sections...")
        
        sections = {}
        
        section_patterns = {
            'item_1': r'ITEM\s+1[.\s]+BUSINESS',
            'item_1a': r'ITEM\s+1A[.\s]+RISK\s+FACTORS',
            'item_7': r'ITEM\s+7[.\s]+MANAGEMENT.?S\s+DISCUSSION',
            'item_8': r'ITEM\s+8[.\s]+FINANCIAL\s+STATEMENTS',
            'item_9a': r'ITEM\s+9A[.\s]+CONTROLS\s+AND\s+PROCEDURES',
            'item_10': r'ITEM\s+10[.\s]+DIRECTORS',
            'item_11': r'ITEM\s+11[.\s]+EXECUTIVE\s+COMPENSATION',
            'item_15': r'ITEM\s+15[.\s]+EXHIBITS',
        }
        
        for section_id, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_pos = match.start()
                sections[section_id] = {
                    'section_id': section_id,
                    'title': match.group(),
                    'start_pos': start_pos,
                }
        
        # Calculate end positions
        section_list = sorted(sections.items(), key=lambda x: x[1]['start_pos'])
        for i, (section_id, section_info) in enumerate(section_list):
            if i < len(section_list) - 1:
                next_start = section_list[i + 1][1]['start_pos']
                section_info['end_pos'] = next_start
                section_info['text'] = text[section_info['start_pos']:next_start]
            else:
                section_info['end_pos'] = len(text)
                section_info['text'] = text[section_info['start_pos']:]
            
            section_info['word_count'] = len(section_info['text'].split())
        
        logger.info(f"✓ Found {len(sections)} sections\n")
        return sections
    
    def chunk_text(self, text: str, section_id: str) -> list:
        """Create chunks from text"""
        tokens = self.encoding.encode(text)
        chunks = []
        start_idx = 0
        chunk_num = 0
        
        while start_idx < len(tokens):
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk = {
                'chunk_id': f"{section_id}_chunk_{chunk_num}",
                'section_id': section_id,
                'text': chunk_text,
                'token_count': len(chunk_tokens),
                'char_count': len(chunk_text),
                'start_page': 1,
                'chunk_index': chunk_num,
            }
            
            chunks.append(chunk)
            start_idx = end_idx - self.chunk_overlap
            chunk_num += 1
        
        return chunks


def process_pdf_fast(pdf_path: str, ticker: str, year: int):
    """
    Process PDF fast - all at once
    
    Args:
        pdf_path: Path to PDF
        ticker: Company ticker
        year: Fiscal year
    """
    logger.info(f"🚀 Fast processing: {ticker} - {year}")
    logger.info(f"📁 File: {pdf_path}\n")
    
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return False
    
    processor = FastPDFProcessor()
    
    # Step 1: Extract text
    logger.info("STEP 1/4: Extracting text...")
    text = processor.extract_text_from_pdf(pdf_file)
    
    # Step 2: Identify sections
    logger.info("STEP 2/4: Identifying sections...")
    sections = processor.identify_sections(text)
    
    for section_id, info in sections.items():
        logger.info(f"   {section_id}: {info['word_count']:,} words")
    logger.info("")
    
    # Step 3: Create all chunks
    logger.info("STEP 3/4: Creating chunks...")
    all_chunks = []
    
    for section_id, section_info in sections.items():
        section_text = section_info.get('text', '')
        if section_text and section_text.strip():
            chunks = processor.chunk_text(section_text, section_id)
            all_chunks.extend(chunks)
    
    logger.info(f"✓ Created {len(all_chunks)} chunks\n")
    
    # Step 4: Process all at once
    logger.info("STEP 4/4: Generating embeddings and indexing...")
    
    doc_id = f"{ticker}_{year}_10K"
    
    # Add metadata
    for chunk in all_chunks:
        chunk['ticker'] = ticker
        chunk['fiscal_year'] = year
        chunk['doc_id'] = doc_id
    
    # Generate all embeddings
    logger.info("   Generating embeddings...")
    embedding_gen = EmbeddingGenerator()
    embeddings = embedding_gen.embed_chunks(all_chunks)
    
    # Index all
    logger.info("   Indexing in database...")
    milvus_client = MilvusClient()
    milvus_client.insert_chunks(all_chunks, embeddings)
    milvus_client.close()
    
    logger.info(f"\n✅ SUCCESS!")
    logger.info(f"   Document ID: {doc_id}")
    logger.info(f"   Total chunks: {len(all_chunks)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Fast PDF processor - no batching'
    )
    parser.add_argument('--pdf', required=True, help='Path to PDF file')
    parser.add_argument('--ticker', required=True, help='Company ticker')
    parser.add_argument('--year', type=int, required=True, help='Fiscal year')
    
    args = parser.parse_args()
    
    try:
        success = process_pdf_fast(
            pdf_path=args.pdf,
            ticker=args.ticker.upper(),
            year=args.year
        )
        
        if success:
            logger.info("\n🎉 Done!")
            sys.exit(0)
        else:
            logger.error("\n❌ Failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
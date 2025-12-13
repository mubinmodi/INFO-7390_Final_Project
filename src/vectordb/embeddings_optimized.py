"""
Memory-optimized embedding generator with rate limiting for Gemini
"""
import logging
import time
from typing import List
import google.generativeai as genai
from config.settings import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)


class EmbeddingGenerator:
    """Generate embeddings with memory optimization and rate limiting"""
    
    def __init__(self, model: str = GEMINI_EMBEDDING_MODEL):
        """
        Initialize embedding generator
        
        Args:
            model: Gemini embedding model to use
        """
        self.model = model
        self.rate_limit_delay = 0.2  # 200ms between requests (5 per second)
        
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text with rate limiting
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                return result['embedding']
                
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    # Rate limit hit, wait longer
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif attempt == max_retries - 1:
                    logger.error(f"Error generating embedding after {max_retries} attempts: {e}")
                    raise
                else:
                    time.sleep(1)
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings with progress logging
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        total = len(texts)
        
        for i, text in enumerate(texts, 1):
            try:
                embedding = self.generate_embedding(text)
                all_embeddings.append(embedding)
                
                # Progress logging
                if i % 5 == 0 or i == total:
                    logger.info(f"    Generated {i}/{total} embeddings ({i*100//total}%)")
                
            except Exception as e:
                logger.error(f"Failed to generate embedding {i}/{total}: {e}")
                raise
        
        return all_embeddings
    
    def embed_chunks(self, chunks: List[dict]) -> List[List[float]]:
        """
        Generate embeddings for document chunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            
        Returns:
            List of embedding vectors
        """
        texts = [chunk['text'] for chunk in chunks]
        return self.generate_embeddings_batch(texts)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    generator = EmbeddingGenerator()
    
    # Test
    texts = ["Test sentence " + str(i) for i in range(5)]
    embeddings = generator.generate_embeddings_batch(texts)
    print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
"""
Generate embeddings using Google Gemini API
"""
import logging
from typing import List
import google.generativeai as genai
from config.settings import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)


class EmbeddingGenerator:
    """Generate embeddings for text using Google Gemini"""
    
    def __init__(self, model: str = GEMINI_EMBEDDING_MODEL):
        """
        Initialize embedding generator
        
        Args:
            model: Gemini embedding model to use
        """
        self.model = model
        
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            
            return result['embedding']
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        # Process each text (Gemini doesn't have batch API like OpenAI)
        for i, text in enumerate(texts):
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                
                all_embeddings.append(result['embedding'])
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Generated embeddings for {i + 1}/{len(texts)} texts")
                
            except Exception as e:
                logger.error(f"Error generating embedding for text {i}: {e}")
                raise
        
        logger.info(f"Generated {len(all_embeddings)} embeddings")
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
    
    # Test single embedding
    text = "This is a test sentence for embedding generation."
    embedding = generator.generate_embedding(text)
    print(f"Generated embedding of dimension: {len(embedding)}")
    
    # Test batch embeddings
    texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence."
    ]
    embeddings = generator.generate_embeddings_batch(texts)
    print(f"Generated {len(embeddings)} embeddings")
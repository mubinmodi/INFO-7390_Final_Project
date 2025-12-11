"""
Unified Embedding Generator supporting OpenAI and Google Gemini
"""
import logging
from typing import List, Optional
import numpy as np
from config.settings import (
    EMBEDDING_PROVIDER, 
    OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL,
    GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL
)

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using OpenAI or Gemini"""
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize embedding generator
        
        Args:
            provider: "openai" or "gemini" (defaults to EMBEDDING_PROVIDER from settings)
        """
        self.provider = provider or EMBEDDING_PROVIDER
        
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "gemini":
            self._init_gemini()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set")
            
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_EMBEDDING_MODEL
            self.dimension = 3072
            logger.info(f"Initialized OpenAI embeddings: {self.model}")
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI: {e}")
            raise
    
    def _init_gemini(self):
        """Initialize Gemini client"""
        try:
            import google.generativeai as genai
            
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set")
            
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = GEMINI_EMBEDDING_MODEL
            self.dimension = 768
            logger.info(f"Initialized Gemini embeddings: {self.model}")
            
        except Exception as e:
            logger.error(f"Error initializing Gemini: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if self.provider == "openai":
            return self._embed_openai([text])[0]
        else:
            return self._embed_gemini([text])[0]
    
    def embed_chunks(self, chunks: List[dict], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple chunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            batch_size: Number of texts to process in one batch
            
        Returns:
            List of embedding vectors
        """
        texts = [chunk['text'] for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks using {self.provider}")
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            if self.provider == "openai":
                embeddings = self._embed_openai(batch)
            else:
                embeddings = self._embed_gemini(batch)
            
            all_embeddings.extend(embeddings)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [item.embedding for item in response.data]
            
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            raise
    
    def _embed_gemini(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Gemini"""
        try:
            import google.generativeai as genai
            
            embeddings = []
            
            for text in texts:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating Gemini embeddings: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with Gemini
    generator = EmbeddingGenerator(provider="gemini")
    
    text = "Apple Inc. is a technology company that designs and manufactures consumer electronics."
    embedding = generator.embed_text(text)
    
    print(f"Provider: {generator.provider}")
    print(f"Dimension: {len(embedding)}")
    print(f"Sample values: {embedding[:5]}")
"""
Base Agent class for SEC filing analysis
"""
import logging
from typing import Dict, List, Optional
import google.generativeai as genai
from config.settings import GEMINI_API_KEY, GEMINI_MODEL, TEMPERATURE
from src.vectordb.milvus_client import MilvusClient
from src.vectordb.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)


class BaseAgent:
    """Base class for all analysis agents"""
    
    def __init__(self, 
                 milvus_client: MilvusClient,
                 embedding_generator: EmbeddingGenerator,
                 model: str = GEMINI_MODEL,
                 temperature: float = TEMPERATURE):
        """
        Initialize base agent
        
        Args:
            milvus_client: Milvus database client
            embedding_generator: Embedding generator
            model: Gemini model to use
            temperature: Temperature for generation
        """
        self.milvus_client = milvus_client
        self.embedding_generator = embedding_generator
        self.model = model
        self.temperature = temperature
        self.client = genai.GenerativeModel(model)
        
    def retrieve_context(self,
                        queries: List[str],
                        ticker: Optional[str] = None,
                        section_ids: Optional[List[str]] = None,
                        top_k: int = 5) -> str:
        """
        Retrieve relevant context from vector database
        
        Args:
            queries: List of search queries
            ticker: Filter by ticker
            section_ids: Filter by section IDs
            top_k: Number of results per query
            
        Returns:
            Combined context text
        """
        all_results = []
        
        for query in queries:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Search for each section if specified
            if section_ids:
                for section_id in section_ids:
                    results = self.milvus_client.search(
                        query_embedding=query_embedding,
                        ticker=ticker,
                        section_id=section_id,
                        top_k=top_k
                    )
                    all_results.extend(results)
            else:
                results = self.milvus_client.search(
                    query_embedding=query_embedding,
                    ticker=ticker,
                    top_k=top_k
                )
                all_results.extend(results)
        
        # Deduplicate and sort by score
        unique_results = {r['chunk_id']: r for r in all_results}
        sorted_results = sorted(unique_results.values(), 
                               key=lambda x: x['score'], 
                               reverse=True)
        
        # Combine context
        context_parts = []
        for result in sorted_results[:top_k * len(queries)]:
            context_parts.append(
                f"[{result['section_id']} - Page {result['start_page']}]\n{result['text']}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call Gemini LLM
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            
        Returns:
            LLM response
        """
        try:
            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = self.client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": 4000
                }
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise
    
    def analyze(self, **kwargs) -> Dict:
        """
        Perform analysis - to be implemented by subclasses
        
        Returns:
            Analysis results
        """
        raise NotImplementedError("Subclasses must implement analyze method")
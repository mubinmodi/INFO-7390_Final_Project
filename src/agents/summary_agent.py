"""
Summary Agent - Generate executive summary of SEC filings
"""
import logging
from typing import Dict
from src.agents.base_agent import BaseAgent
from config.prompts import SUMMARY_AGENT_PROMPT, RETRIEVAL_QUERY_TEMPLATES
from config.settings import SUMMARY_SECTIONS

logger = logging.getLogger(__name__)


class SummaryAgent(BaseAgent):
    """Agent for generating executive summaries of SEC filings"""
    
    def analyze(self, ticker: str, fiscal_year: int, company_name: str = None) -> Dict:
        """
        Generate executive summary for a company's filing
        
        Args:
            ticker: Company ticker symbol
            fiscal_year: Fiscal year of filing
            company_name: Company name (optional)
            
        Returns:
            Dictionary with summary results
        """
        logger.info(f"Generating summary for {ticker} - {fiscal_year}")
        
        # Use ticker as company name if not provided
        if not company_name:
            company_name = ticker
        
        # Retrieve relevant context
        context = self.retrieve_context(
            queries=RETRIEVAL_QUERY_TEMPLATES['summary'],
            ticker=ticker,
            section_ids=SUMMARY_SECTIONS,
            top_k=3
        )
        
        # Generate summary using LLM
        prompt = SUMMARY_AGENT_PROMPT.format(
            context=context,
            company=company_name,
            fiscal_year=fiscal_year
        )
        
        system_prompt = "You are a financial analyst specializing in SEC filing analysis."
        
        summary = self.call_llm(system_prompt, prompt)
        
        result = {
            'agent': 'summary',
            'ticker': ticker,
            'fiscal_year': fiscal_year,
            'company': company_name,
            'summary': summary,
            'sections_analyzed': SUMMARY_SECTIONS
        }
        
        logger.info(f"Generated summary for {ticker}")
        return result


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from src.vectordb.milvus_client import MilvusClient
    from src.vectordb.embeddings import EmbeddingGenerator
    
    milvus_client = MilvusClient()
    embedding_generator = EmbeddingGenerator()
    
    agent = SummaryAgent(milvus_client, embedding_generator)
    
    # Example analysis
    # result = agent.analyze(ticker="AAPL", fiscal_year=2023)
    # print(result['summary'])
"""
SWOT Agent - Perform rigorous SWOT analysis on SEC filings
"""
import logging
from typing import Dict
from src.agents.base_agent import BaseAgent
from config.prompts import SWOT_AGENT_PROMPT, RETRIEVAL_QUERY_TEMPLATES
from config.settings import SWOT_SECTIONS

logger = logging.getLogger(__name__)


class SWOTAgent(BaseAgent):
    """Agent for performing SWOT analysis on SEC filings"""
    
    def analyze(self, ticker: str, fiscal_year: int, company_name: str = None) -> Dict:
        """
        Perform SWOT analysis for a company's filing
        
        Args:
            ticker: Company ticker symbol
            fiscal_year: Fiscal year of filing
            company_name: Company name (optional)
            
        Returns:
            Dictionary with SWOT analysis results
        """
        logger.info(f"Performing SWOT analysis for {ticker} - {fiscal_year}")
        
        # Use ticker as company name if not provided
        if not company_name:
            company_name = ticker
        
        # Retrieve context for each SWOT component
        strengths_context = self.retrieve_context(
            queries=RETRIEVAL_QUERY_TEMPLATES['strengths'],
            ticker=ticker,
            section_ids=SWOT_SECTIONS,
            top_k=3
        )
        
        weaknesses_context = self.retrieve_context(
            queries=RETRIEVAL_QUERY_TEMPLATES['weaknesses'],
            ticker=ticker,
            section_ids=SWOT_SECTIONS,
            top_k=3
        )
        
        opportunities_context = self.retrieve_context(
            queries=RETRIEVAL_QUERY_TEMPLATES['opportunities'],
            ticker=ticker,
            section_ids=SWOT_SECTIONS,
            top_k=3
        )
        
        threats_context = self.retrieve_context(
            queries=RETRIEVAL_QUERY_TEMPLATES['threats'],
            ticker=ticker,
            section_ids=SWOT_SECTIONS,
            top_k=3
        )
        
        # Combine all context
        combined_context = f"""
STRENGTHS CONTEXT:
{strengths_context}

WEAKNESSES CONTEXT:
{weaknesses_context}

OPPORTUNITIES CONTEXT:
{opportunities_context}

THREATS CONTEXT:
{threats_context}
"""
        
        # Generate SWOT analysis using LLM
        prompt = SWOT_AGENT_PROMPT.format(
            context=combined_context,
            company=company_name,
            fiscal_year=fiscal_year
        )
        
        system_prompt = "You are a buy-side hedge fund analyst performing hostile witness analysis on SEC filings."
        
        swot_analysis = self.call_llm(system_prompt, prompt)
        
        # Parse SWOT analysis into components
        swot_components = self._parse_swot(swot_analysis)
        
        result = {
            'agent': 'swot',
            'ticker': ticker,
            'fiscal_year': fiscal_year,
            'company': company_name,
            'swot_analysis': swot_analysis,
            'swot_components': swot_components,
            'sections_analyzed': SWOT_SECTIONS
        }
        
        logger.info(f"Generated SWOT analysis for {ticker}")
        return result
    
    def _parse_swot(self, swot_text: str) -> Dict:
        """
        Parse SWOT analysis text into components
        
        Args:
            swot_text: Full SWOT analysis text
            
        Returns:
            Dictionary with parsed SWOT components
        """
        components = {
            'strengths': '',
            'weaknesses': '',
            'opportunities': '',
            'threats': ''
        }
        
        # Simple parsing based on section headers
        sections = swot_text.split('**')
        current_section = None
        
        for section in sections:
            section_lower = section.lower().strip()
            
            if section_lower.startswith('strengths'):
                current_section = 'strengths'
            elif section_lower.startswith('weaknesses'):
                current_section = 'weaknesses'
            elif section_lower.startswith('opportunities'):
                current_section = 'opportunities'
            elif section_lower.startswith('threats'):
                current_section = 'threats'
            elif current_section:
                components[current_section] += section + '\n'
        
        # Clean up components
        for key in components:
            components[key] = components[key].strip()
        
        return components


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from src.vectordb.milvus_client import MilvusClient
    from src.vectordb.embeddings import EmbeddingGenerator
    
    milvus_client = MilvusClient()
    embedding_generator = EmbeddingGenerator()
    
    agent = SWOTAgent(milvus_client, embedding_generator)
    
    # Example analysis
    # result = agent.analyze(ticker="AAPL", fiscal_year=2023)
    # print(result['swot_analysis'])
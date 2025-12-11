"""
Metrics Agent - Extract and calculate financial KPIs from SEC filings
"""
import json
import logging
import re
from typing import Dict, Optional
from src.agents.base_agent import BaseAgent
from config.prompts import METRICS_AGENT_PROMPT, RETRIEVAL_QUERY_TEMPLATES
from config.settings import METRICS_SECTIONS, KEY_METRICS, CALCULATED_METRICS

logger = logging.getLogger(__name__)


class MetricsAgent(BaseAgent):
    """Agent for extracting and calculating financial metrics"""
    
    def analyze(self, 
                ticker: str, 
                fiscal_year: int, 
                company_name: str = None,
                prior_year: Optional[int] = None) -> Dict:
        """
        Extract and calculate financial metrics for a company's filing
        
        Args:
            ticker: Company ticker symbol
            fiscal_year: Fiscal year of filing
            company_name: Company name (optional)
            prior_year: Prior fiscal year for comparison (optional)
            
        Returns:
            Dictionary with financial metrics
        """
        logger.info(f"Extracting metrics for {ticker} - {fiscal_year}")
        
        # Use ticker as company name if not provided
        if not company_name:
            company_name = ticker
        
        # Default prior year to fiscal_year - 1
        if not prior_year:
            prior_year = fiscal_year - 1
        
        # Retrieve financial statement context
        context = self.retrieve_context(
            queries=RETRIEVAL_QUERY_TEMPLATES['metrics'],
            ticker=ticker,
            section_ids=METRICS_SECTIONS,
            top_k=5
        )
        
        # Generate metrics extraction using LLM
        prompt = METRICS_AGENT_PROMPT.format(
            context=context,
            company=company_name,
            fiscal_year=fiscal_year,
            prior_year=prior_year
        )
        
        system_prompt = "You are a financial data analyst extracting KPIs from SEC filings."
        
        metrics_response = self.call_llm(system_prompt, prompt)
        
        # Parse JSON response
        metrics_data = self._parse_metrics(metrics_response)
        
        # Calculate additional metrics if base metrics are available
        if metrics_data:
            metrics_data = self._calculate_derived_metrics(metrics_data)
        
        result = {
            'agent': 'metrics',
            'ticker': ticker,
            'fiscal_year': fiscal_year,
            'prior_year': prior_year,
            'company': company_name,
            'metrics': metrics_data,
            'raw_response': metrics_response,
            'sections_analyzed': METRICS_SECTIONS
        }
        
        logger.info(f"Extracted metrics for {ticker}")
        return result
    
    def _parse_metrics(self, metrics_text: str) -> Dict:
        """
        Parse metrics from LLM response
        
        Args:
            metrics_text: Raw LLM response
            
        Returns:
            Parsed metrics dictionary
        """
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', metrics_text, re.DOTALL)
            if json_match:
                metrics_json = json_match.group()
                metrics_data = json.loads(metrics_json)
                return metrics_data
            else:
                logger.warning("Could not find JSON in metrics response")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing metrics JSON: {e}")
            return {}
    
    def _calculate_derived_metrics(self, metrics_data: Dict) -> Dict:
        """
        Calculate derived financial metrics
        
        Args:
            metrics_data: Base metrics dictionary
            
        Returns:
            Metrics dictionary with calculated ratios
        """
        current = metrics_data.get('current_year', {})
        prior = metrics_data.get('prior_year', {})
        
        # Calculate margins
        if 'revenue' in current and 'gross_profit' in current:
            try:
                current['gross_margin'] = (current['gross_profit'] / current['revenue']) * 100
            except (ZeroDivisionError, TypeError):
                pass
        
        if 'revenue' in current and 'operating_income' in current:
            try:
                current['operating_margin'] = (current['operating_income'] / current['revenue']) * 100
            except (ZeroDivisionError, TypeError):
                pass
        
        if 'revenue' in current and 'net_income' in current:
            try:
                current['net_margin'] = (current['net_income'] / current['revenue']) * 100
            except (ZeroDivisionError, TypeError):
                pass
        
        # Calculate returns
        if 'net_income' in current and 'stockholders_equity' in current:
            try:
                current['roe'] = (current['net_income'] / current['stockholders_equity']) * 100
            except (ZeroDivisionError, TypeError):
                pass
        
        if 'net_income' in current and 'total_assets' in current:
            try:
                current['roa'] = (current['net_income'] / current['total_assets']) * 100
            except (ZeroDivisionError, TypeError):
                pass
        
        # Calculate liquidity ratios
        if 'current_assets' in current and 'current_liabilities' in current:
            try:
                current['current_ratio'] = current['current_assets'] / current['current_liabilities']
            except (ZeroDivisionError, TypeError):
                pass
        
        # Calculate leverage
        if 'total_debt' in current and 'stockholders_equity' in current:
            try:
                current['debt_to_equity'] = current['total_debt'] / current['stockholders_equity']
            except (ZeroDivisionError, TypeError):
                pass
        
        # Calculate growth rates
        if prior:
            if 'revenue' in current and 'revenue' in prior:
                try:
                    current['revenue_growth'] = ((current['revenue'] - prior['revenue']) / prior['revenue']) * 100
                except (ZeroDivisionError, TypeError):
                    pass
            
            if 'net_income' in current and 'net_income' in prior:
                try:
                    current['net_income_growth'] = ((current['net_income'] - prior['net_income']) / prior['net_income']) * 100
                except (ZeroDivisionError, TypeError):
                    pass
        
        metrics_data['current_year'] = current
        
        return metrics_data


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from src.vectordb.milvus_client import MilvusClient
    from src.vectordb.embeddings import EmbeddingGenerator
    
    milvus_client = MilvusClient()
    embedding_generator = EmbeddingGenerator()
    
    agent = MetricsAgent(milvus_client, embedding_generator)
    
    # Example analysis
    # result = agent.analyze(ticker="AAPL", fiscal_year=2023)
    # print(json.dumps(result['metrics'], indent=2))
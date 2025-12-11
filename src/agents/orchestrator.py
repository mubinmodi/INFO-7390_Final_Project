"""
Multi-Agent Orchestrator - Coordinates all analysis agents
"""
import logging
from typing import Dict
from src.agents.summary_agent import SummaryAgent
from src.agents.swot_agent import SWOTAgent
from src.agents.metrics_agent import MetricsAgent
from src.agents.decision_agent import DecisionAgent
from src.vectordb.milvus_client import MilvusClient
from src.vectordb.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class AnalysisOrchestrator:
    """Orchestrate multi-agent analysis of SEC filings"""
    
    def __init__(self):
        """Initialize orchestrator with all agents"""
        logger.info("Initializing Analysis Orchestrator")
        
        # Initialize vector database and embeddings
        self.milvus_client = MilvusClient()
        self.embedding_generator = EmbeddingGenerator()
        
        # Initialize all agents
        self.summary_agent = SummaryAgent(self.milvus_client, self.embedding_generator)
        self.swot_agent = SWOTAgent(self.milvus_client, self.embedding_generator)
        self.metrics_agent = MetricsAgent(self.milvus_client, self.embedding_generator)
        self.decision_agent = DecisionAgent(self.milvus_client, self.embedding_generator)
        
        logger.info("All agents initialized successfully")
    
    def analyze_filing(self, 
                      ticker: str, 
                      fiscal_year: int,
                      company_name: str = None) -> Dict:
        """
        Perform complete analysis of a SEC filing
        
        Args:
            ticker: Company ticker symbol
            fiscal_year: Fiscal year of filing
            company_name: Company name (optional)
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info(f"Starting comprehensive analysis for {ticker} - {fiscal_year}")
        
        if not company_name:
            company_name = ticker
        
        results = {
            'ticker': ticker,
            'fiscal_year': fiscal_year,
            'company': company_name,
            'status': 'in_progress'
        }
        
        try:
            # Step 1: Generate Summary
            logger.info("Step 1/4: Generating executive summary...")
            summary_result = self.summary_agent.analyze(
                ticker=ticker,
                fiscal_year=fiscal_year,
                company_name=company_name
            )
            results['summary'] = summary_result
            
            # Step 2: Perform SWOT Analysis
            logger.info("Step 2/4: Performing SWOT analysis...")
            swot_result = self.swot_agent.analyze(
                ticker=ticker,
                fiscal_year=fiscal_year,
                company_name=company_name
            )
            results['swot'] = swot_result
            
            # Step 3: Extract Metrics
            logger.info("Step 3/4: Extracting financial metrics...")
            metrics_result = self.metrics_agent.analyze(
                ticker=ticker,
                fiscal_year=fiscal_year,
                company_name=company_name
            )
            results['metrics'] = metrics_result
            
            # Step 4: Generate Investment Decision
            logger.info("Step 4/4: Generating investment decision...")
            decision_result = self.decision_agent.analyze(
                ticker=ticker,
                fiscal_year=fiscal_year,
                summary_result=summary_result,
                swot_result=swot_result,
                metrics_result=metrics_result,
                company_name=company_name
            )
            results['decision'] = decision_result
            
            results['status'] = 'completed'
            logger.info(f"Comprehensive analysis completed for {ticker}")
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def get_quick_summary(self, ticker: str, fiscal_year: int) -> str:
        """
        Get a quick text summary of the analysis
        
        Args:
            ticker: Company ticker symbol
            fiscal_year: Fiscal year
            
        Returns:
            Quick summary text
        """
        results = self.analyze_filing(ticker, fiscal_year)
        
        if results['status'] != 'completed':
            return f"Analysis failed: {results.get('error', 'Unknown error')}"
        
        summary = f"""
# {results['company']} ({results['ticker']}) - FY {results['fiscal_year']} Analysis

## Investment Recommendation: {results['decision']['recommendation']}
**Confidence:** {results['decision']['confidence']}
**Position Sizing:** {results['decision']['position_sizing']}

## Executive Summary
{results['summary']['summary']}

## Key Metrics
{self._format_key_metrics(results['metrics']['metrics'])}

## SWOT Highlights
{self._format_swot_highlights(results['swot']['swot_components'])}

## Investment Thesis
{results['decision']['investment_thesis']}

## Red Flags
{results['decision']['red_flags']}
"""
        return summary
    
    def _format_key_metrics(self, metrics: Dict) -> str:
        """Format key metrics for summary"""
        current = metrics.get('current_year', {})
        
        parts = []
        if 'revenue' in current:
            parts.append(f"- Revenue: ${current['revenue']:,.0f}")
        if 'net_income' in current:
            parts.append(f"- Net Income: ${current['net_income']:,.0f}")
        if 'revenue_growth' in current:
            parts.append(f"- Revenue Growth: {current['revenue_growth']:.1f}%")
        if 'net_margin' in current:
            parts.append(f"- Net Margin: {current['net_margin']:.1f}%")
        if 'roe' in current:
            parts.append(f"- ROE: {current['roe']:.1f}%")
        
        return "\n".join(parts) if parts else "Metrics not available"
    
    def _format_swot_highlights(self, swot_components: Dict) -> str:
        """Format SWOT highlights for summary"""
        parts = []
        
        for key in ['strengths', 'weaknesses', 'opportunities', 'threats']:
            if key in swot_components and swot_components[key]:
                # Get first few lines of each component
                lines = swot_components[key].split('\n')[:3]
                formatted = '\n  '.join(lines)
                parts.append(f"**{key.title()}:**\n  {formatted}")
        
        return "\n\n".join(parts) if parts else "SWOT analysis not available"
    
    def close(self):
        """Close database connections"""
        self.milvus_client.close()
        logger.info("Orchestrator closed")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    orchestrator = AnalysisOrchestrator()
    
    # Perform complete analysis
    # results = orchestrator.analyze_filing(ticker="AAPL", fiscal_year=2023)
    # print(results['decision']['recommendation'])
    
    orchestrator.close()
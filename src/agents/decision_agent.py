"""
Decision Agent - Synthesize multi-agent analysis into investment recommendation
"""
import logging
from typing import Dict
from src.agents.base_agent import BaseAgent
from config.prompts import DECISION_AGENT_PROMPT, RETRIEVAL_QUERY_TEMPLATES

logger = logging.getLogger(__name__)


class DecisionAgent(BaseAgent):
    """Agent for making investment recommendations based on comprehensive analysis"""
    
    def analyze(self,
                ticker: str,
                fiscal_year: int,
                summary_result: Dict,
                swot_result: Dict,
                metrics_result: Dict,
                company_name: str = None) -> Dict:
        """
        Generate investment recommendation based on all agent analyses
        
        Args:
            ticker: Company ticker symbol
            fiscal_year: Fiscal year of filing
            summary_result: Results from Summary Agent
            swot_result: Results from SWOT Agent
            metrics_result: Results from Metrics Agent
            company_name: Company name (optional)
            
        Returns:
            Dictionary with investment decision and recommendation
        """
        logger.info(f"Generating investment decision for {ticker} - {fiscal_year}")
        
        # Use ticker as company name if not provided
        if not company_name:
            company_name = ticker
        
        # Check for red flags in the filing
        red_flags_context = self.retrieve_context(
            queries=RETRIEVAL_QUERY_TEMPLATES['red_flags'],
            ticker=ticker,
            top_k=3
        )
        
        # Format inputs for decision prompt
        summary_text = summary_result.get('summary', 'Not available')
        swot_text = swot_result.get('swot_analysis', 'Not available')
        
        # Format metrics for readability
        metrics_text = self._format_metrics(metrics_result.get('metrics', {}))
        
        # Generate investment decision using LLM
        prompt = DECISION_AGENT_PROMPT.format(
            summary=summary_text,
            swot=swot_text,
            metrics=metrics_text,
            company=company_name,
            fiscal_year=fiscal_year
        )
        
        system_prompt = """You are a chief investment officer synthesizing multi-agent analysis to make an investment recommendation.
        
Additional context about potential red flags:
{red_flags_context}

Be brutally honest and rigorous in your assessment.""".format(red_flags_context=red_flags_context)
        
        decision_response = self.call_llm(system_prompt, prompt)
        
        # Parse decision components
        decision_components = self._parse_decision(decision_response)
        
        result = {
            'agent': 'decision',
            'ticker': ticker,
            'fiscal_year': fiscal_year,
            'company': company_name,
            'full_decision': decision_response,
            'investment_thesis': decision_components.get('investment_thesis', ''),
            'red_flags': decision_components.get('red_flags', ''),
            'quality_scores': decision_components.get('quality_scores', {}),
            'recommendation': decision_components.get('recommendation', 'HOLD'),
            'confidence': decision_components.get('confidence', 'MEDIUM'),
            'catalysts_and_risks': decision_components.get('catalysts_and_risks', {}),
            'position_sizing': decision_components.get('position_sizing', 'MARKET WEIGHT'),
            'inputs': {
                'summary': summary_result,
                'swot': swot_result,
                'metrics': metrics_result
            }
        }
        
        logger.info(f"Generated investment decision for {ticker}: {decision_components.get('recommendation')}")
        return result
    
    def _format_metrics(self, metrics_data: Dict) -> str:
        """
        Format metrics dictionary into readable text
        
        Args:
            metrics_data: Metrics dictionary
            
        Returns:
            Formatted metrics string
        """
        if not metrics_data:
            return "No metrics available"
        
        formatted_parts = []
        
        current = metrics_data.get('current_year', {})
        prior = metrics_data.get('prior_year', {})
        
        # Income statement
        formatted_parts.append("INCOME STATEMENT:")
        for key in ['revenue', 'gross_profit', 'operating_income', 'net_income', 'eps']:
            if key in current:
                line = f"  {key.replace('_', ' ').title()}: ${current[key]:,.2f}" if isinstance(current[key], (int, float)) else f"  {key.replace('_', ' ').title()}: {current[key]}"
                if key in prior:
                    change = ((current[key] - prior[key]) / prior[key] * 100) if prior[key] != 0 else 0
                    line += f" ({change:+.1f}% YoY)"
                formatted_parts.append(line)
        
        # Balance sheet
        formatted_parts.append("\nBALANCE SHEET:")
        for key in ['total_assets', 'total_liabilities', 'stockholders_equity', 'total_debt']:
            if key in current:
                formatted_parts.append(f"  {key.replace('_', ' ').title()}: ${current[key]:,.2f}" if isinstance(current[key], (int, float)) else f"  {key.replace('_', ' ').title()}: {current[key]}")
        
        # Cash flow
        formatted_parts.append("\nCASH FLOW:")
        for key in ['cash_from_operations', 'free_cash_flow', 'capex']:
            if key in current:
                formatted_parts.append(f"  {key.replace('_', ' ').title()}: ${current[key]:,.2f}" if isinstance(current[key], (int, float)) else f"  {key.replace('_', ' ').title()}: {current[key]}")
        
        # Ratios
        formatted_parts.append("\nKEY RATIOS:")
        for key in ['gross_margin', 'operating_margin', 'net_margin', 'roe', 'roa', 
                    'debt_to_equity', 'current_ratio', 'revenue_growth']:
            if key in current:
                value = current[key]
                if isinstance(value, (int, float)):
                    if 'margin' in key or 'growth' in key or key in ['roe', 'roa']:
                        formatted_parts.append(f"  {key.replace('_', ' ').title()}: {value:.2f}%")
                    else:
                        formatted_parts.append(f"  {key.replace('_', ' ').title()}: {value:.2f}")
                else:
                    formatted_parts.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(formatted_parts)
    
    def _parse_decision(self, decision_text: str) -> Dict:
        """
        Parse decision response into structured components
        
        Args:
            decision_text: Full decision text
            
        Returns:
            Dictionary with parsed decision components
        """
        components = {
            'investment_thesis': '',
            'red_flags': '',
            'quality_scores': {},
            'recommendation': 'HOLD',
            'confidence': 'MEDIUM',
            'catalysts_and_risks': {},
            'position_sizing': 'MARKET WEIGHT'
        }
        
        # Extract recommendation
        if 'STRONG BUY' in decision_text:
            components['recommendation'] = 'STRONG BUY'
        elif 'BUY' in decision_text:
            components['recommendation'] = 'BUY'
        elif 'STRONG SELL' in decision_text:
            components['recommendation'] = 'STRONG SELL'
        elif 'SELL' in decision_text:
            components['recommendation'] = 'SELL'
        else:
            components['recommendation'] = 'HOLD'
        
        # Extract confidence
        if 'HIGH' in decision_text and 'Confidence' in decision_text:
            components['confidence'] = 'HIGH'
        elif 'LOW' in decision_text and 'Confidence' in decision_text:
            components['confidence'] = 'LOW'
        else:
            components['confidence'] = 'MEDIUM'
        
        # Extract position sizing
        if 'OVERWEIGHT' in decision_text:
            components['position_sizing'] = 'OVERWEIGHT'
        elif 'UNDERWEIGHT' in decision_text:
            components['position_sizing'] = 'UNDERWEIGHT'
        elif 'AVOID' in decision_text:
            components['position_sizing'] = 'AVOID'
        else:
            components['position_sizing'] = 'MARKET WEIGHT'
        
        # Simple section extraction
        sections = decision_text.split('\n\n')
        for i, section in enumerate(sections):
            section_lower = section.lower()
            
            if 'investment thesis' in section_lower:
                components['investment_thesis'] = sections[i] if i < len(sections) else ''
            elif 'red flags' in section_lower:
                components['red_flags'] = sections[i] if i < len(sections) else ''
            elif 'catalysts' in section_lower or 'risks' in section_lower:
                components['catalysts_and_risks']['text'] = sections[i] if i < len(sections) else ''
        
        return components


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from src.vectordb.milvus_client import MilvusClient
    from src.vectordb.embeddings import EmbeddingGenerator
    
    milvus_client = MilvusClient()
    embedding_generator = EmbeddingGenerator()
    
    agent = DecisionAgent(milvus_client, embedding_generator)
    
    # Example analysis (requires results from other agents)
    # result = agent.analyze(
    #     ticker="AAPL",
    #     fiscal_year=2023,
    #     summary_result=summary_result,
    #     swot_result=swot_result,
    #     metrics_result=metrics_result
    # )
    # print(result['recommendation'])
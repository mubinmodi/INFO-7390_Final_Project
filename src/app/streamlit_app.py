"""
Main Streamlit Application for SEC Filing Analysis
"""
import streamlit as st
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.orchestrator import AnalysisOrchestrator
from src.pipeline.downloader import SECDownloader
from src.pipeline.parser import PDFParser
from src.pipeline.table_extractor import TableExtractor
from src.pipeline.preprocessor import DocumentPreprocessor
from src.vectordb.embeddings import EmbeddingGenerator
from src.vectordb.milvus_client import MilvusClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="SEC Filing Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        padding: 2rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .buy {
        background-color: #d4edda;
        color: #155724;
    }
    .sell {
        background-color: #f8d7da;
        color: #721c24;
    }
    .hold {
        background-color: #fff3cd;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

def initialize_system():
    """Initialize the analysis system"""
    if st.session_state.orchestrator is None:
        with st.spinner("Initializing system..."):
            try:
                st.session_state.orchestrator = AnalysisOrchestrator()
                return True
            except Exception as e:
                st.error(f"Error initializing system: {e}")
                logger.error(f"Initialization error: {e}")
                return False
    return True

def process_filing(ticker: str, fiscal_year: int):
    """Process a SEC filing through the complete pipeline"""
    
    # Step 1: Download filing
    with st.spinner("📥 Downloading SEC filing..."):
        try:
            downloader = SECDownloader()
            filing_path = downloader.download_10k(ticker, num_filings=1)
            st.success(f"✅ Downloaded filing for {ticker}")
        except Exception as e:
            st.error(f"Error downloading filing: {e}")
            return False
    
    # Step 2: Parse PDF
    with st.spinner("📄 Parsing PDF document..."):
        try:
            parser = PDFParser()
            pdf_files = list(filing_path.rglob("*.pdf"))
            if not pdf_files:
                st.error("No PDF files found in filing")
                return False
            
            pdf_path = pdf_files[0]
            pages_data = parser.extract_text_from_pdf(pdf_path)
            sections = parser.identify_sections(pages_data)
            st.success(f"✅ Parsed {len(pages_data)} pages, found {len(sections)} sections")
        except Exception as e:
            st.error(f"Error parsing PDF: {e}")
            return False
    
    # Step 3: Extract tables
    with st.spinner("📊 Extracting tables..."):
        try:
            extractor = TableExtractor()
            # Focus on financial statement pages (usually second half)
            total_pages = len(pages_data)
            start_page = total_pages // 2
            pages_str = f"{start_page}-{total_pages}"
            tables = extractor.extract_tables(pdf_path, pages=pages_str, method='auto')
            st.success(f"✅ Extracted {len(tables)} tables")
        except Exception as e:
            st.warning(f"Table extraction had issues: {e}")
            tables = []
    
    # Step 4: Preprocess and chunk
    with st.spinner("🔄 Preprocessing document..."):
        try:
            preprocessor = DocumentPreprocessor()
            document = preprocessor.process_filing(
                ticker=ticker,
                fiscal_year=fiscal_year,
                pages_data=pages_data,
                sections=sections,
                tables=tables
            )
            st.success(f"✅ Created {document['total_chunks']} chunks")
        except Exception as e:
            st.error(f"Error preprocessing: {e}")
            return False
    
    # Step 5: Generate embeddings and index
    with st.spinner("🔍 Generating embeddings and indexing..."):
        try:
            embedding_gen = EmbeddingGenerator()
            embeddings = embedding_gen.embed_chunks(document['chunks'])
            
            # Add metadata to chunks
            for chunk in document['chunks']:
                chunk['ticker'] = ticker
                chunk['fiscal_year'] = fiscal_year
                chunk['doc_id'] = document['doc_id']
            
            milvus_client = MilvusClient()
            milvus_client.insert_chunks(document['chunks'], embeddings)
            st.success(f"✅ Indexed {len(embeddings)} chunks in vector database")
        except Exception as e:
            st.error(f"Error indexing: {e}")
            return False
    
    st.session_state.processing_complete = True
    return True

def display_results(results: dict):
    """Display analysis results in the UI"""
    
    if results['status'] != 'completed':
        st.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
        return
    
    # Header
    st.markdown(f"<div class='main-header'>📊 {results['company']} ({results['ticker']})</div>", 
                unsafe_allow_html=True)
    st.markdown(f"### Fiscal Year {results['fiscal_year']} Analysis")
    
    # Investment Recommendation
    decision = results['decision']
    recommendation = decision['recommendation']
    
    rec_class = 'buy' if 'BUY' in recommendation else ('sell' if 'SELL' in recommendation else 'hold')
    
    st.markdown(f"""
    <div class='recommendation-box {rec_class}'>
        🎯 Investment Recommendation: {recommendation}
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence Level", decision['confidence'])
    with col2:
        st.metric("Position Sizing", decision['position_sizing'])
    with col3:
        overall_score = decision['quality_scores'].get('overall', 'N/A')
        st.metric("Overall Quality Score", overall_score)
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🔍 SWOT", "📊 Metrics", "💡 Decision"])
    
    with tab1:
        st.markdown("### Executive Summary")
        st.markdown(results['summary']['summary'])
    
    with tab2:
        st.markdown("### SWOT Analysis")
        swot_components = results['swot']['swot_components']
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 💪 Strengths")
            st.markdown(swot_components.get('strengths', 'Not available'))
            
            st.markdown("#### 🎯 Opportunities")
            st.markdown(swot_components.get('opportunities', 'Not available'))
        
        with col2:
            st.markdown("#### ⚠️ Weaknesses")
            st.markdown(swot_components.get('weaknesses', 'Not available'))
            
            st.markdown("#### 🚨 Threats")
            st.markdown(swot_components.get('threats', 'Not available'))
    
    with tab3:
        st.markdown("### Financial Metrics")
        metrics = results['metrics']['metrics']
        
        if 'current_year' in metrics:
            current = metrics['current_year']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Income Statement")
                if 'revenue' in current:
                    st.metric("Revenue", f"${current['revenue']:,.0f}")
                if 'net_income' in current:
                    st.metric("Net Income", f"${current['net_income']:,.0f}")
                if 'eps' in current:
                    st.metric("EPS", f"${current['eps']:.2f}")
            
            with col2:
                st.markdown("#### Profitability")
                if 'gross_margin' in current:
                    st.metric("Gross Margin", f"{current['gross_margin']:.1f}%")
                if 'operating_margin' in current:
                    st.metric("Operating Margin", f"{current['operating_margin']:.1f}%")
                if 'net_margin' in current:
                    st.metric("Net Margin", f"{current['net_margin']:.1f}%")
            
            with col3:
                st.markdown("#### Returns & Growth")
                if 'roe' in current:
                    st.metric("ROE", f"{current['roe']:.1f}%")
                if 'roa' in current:
                    st.metric("ROA", f"{current['roa']:.1f}%")
                if 'revenue_growth' in current:
                    st.metric("Revenue Growth", f"{current['revenue_growth']:.1f}%")
        
        # Show full metrics table
        with st.expander("📋 View All Metrics"):
            st.json(metrics)
    
    with tab4:
        st.markdown("### Investment Decision Analysis")
        
        st.markdown("#### Investment Thesis")
        st.markdown(decision['investment_thesis'])
        
        st.markdown("#### 🚩 Red Flags Assessment")
        st.markdown(decision['red_flags'])
        
        if 'catalysts_and_risks' in decision:
            st.markdown("#### Key Catalysts & Risks")
            st.markdown(decision['catalysts_and_risks'].get('text', 'Not available'))

def main():
    """Main application"""
    
    # Sidebar
    with st.sidebar:
        st.image("https://www.sec.gov/themes/custom/uswds_sec/images/logo-img.png", width=200)
        st.title("SEC Filing Analyzer")
        st.markdown("---")
        
        st.markdown("### 📊 Analysis Settings")
        
        ticker = st.text_input("Company Ticker", value="AAPL", max_chars=10).upper()
        fiscal_year = st.number_input("Fiscal Year", min_value=2010, max_value=2025, value=2023)
        company_name = st.text_input("Company Name (optional)", value="")
        
        st.markdown("---")
        
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            if not ticker:
                st.error("Please enter a ticker symbol")
            else:
                # Initialize system
                if initialize_system():
                    # Process filing
                    if not st.session_state.processing_complete:
                        success = process_filing(ticker, fiscal_year)
                        if not success:
                            st.error("Failed to process filing")
                            return
                    
                    # Run analysis
                    with st.spinner("🤖 Running multi-agent analysis..."):
                        try:
                            results = st.session_state.orchestrator.analyze_filing(
                                ticker=ticker,
                                fiscal_year=fiscal_year,
                                company_name=company_name if company_name else None
                            )
                            st.session_state.analysis_results = results
                            st.success("✅ Analysis complete!")
                        except Exception as e:
                            st.error(f"Analysis error: {e}")
                            logger.error(f"Analysis error: {e}")
        
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.analysis_results = None
            st.session_state.processing_complete = False
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This system uses multi-agent AI to analyze SEC 10-K filings:
        
        1. **Summary Agent**: Executive overview
        2. **SWOT Agent**: Strategic analysis
        3. **Metrics Agent**: Financial KPIs
        4. **Decision Agent**: Investment recommendation
        """)
    
    # Main content area
    if st.session_state.analysis_results is None:
        st.markdown("<div class='main-header'>📊 SEC Filing Analyzer</div>", unsafe_allow_html=True)
        st.markdown("""
        ### Welcome to the SEC Filing Analyzer
        
        This AI-powered platform performs comprehensive analysis of SEC 10-K filings using a multi-agent system.
        
        #### How it works:
        1. Enter a company ticker and fiscal year in the sidebar
        2. Click "Start Analysis" to begin processing
        3. The system will download, parse, and analyze the filing
        4. Review the investment recommendation and detailed analysis
        
        #### Features:
        - 📝 **Executive Summary**: Key findings and company direction
        - 🔍 **SWOT Analysis**: Rigorous strengths, weaknesses, opportunities, and threats
        - 📊 **Financial Metrics**: Comprehensive KPIs and ratios
        - 💡 **Investment Decision**: AI-powered recommendation with confidence scoring
        
        **Get started by entering a ticker symbol in the sidebar!**
        """)
        
        # Example companies
        st.markdown("### 💡 Try these examples:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button("AAPL", key="example1")
        with col2:
            st.button("MSFT", key="example2")
        with col3:
            st.button("GOOGL", key="example3")
        with col4:
            st.button("TSLA", key="example4")
    
    else:
        display_results(st.session_state.analysis_results)

if __name__ == "__main__":
    main()
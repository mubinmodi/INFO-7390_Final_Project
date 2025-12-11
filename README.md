# SEC Filing Analysis System - Project Structure

```
sec-filing-analyzer/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   ├── __init__.py
│   ├── settings.py           # Configuration management
│   └── prompts.py             # Agent prompt templates
├── data/
│   ├── raw/                   # Raw SEC filings
│   ├── processed/             # Processed JSONs
│   ├── embeddings/            # Vector embeddings
│   └── metadata/              # Document metadata
├── src/
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── downloader.py     # SEC filing downloader
│   │   ├── parser.py         # PDF/XBRL parsing
│   │   ├── table_extractor.py # Table extraction
│   │   └── preprocessor.py   # Text cleaning & chunking
│   ├── vectordb/
│   │   ├── __init__.py
│   │   ├── milvus_client.py  # Milvus connection & operations
│   │   └── embeddings.py     # OpenAI embeddings generation
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py     # Base agent class
│   │   ├── summary_agent.py  # Summary generation
│   │   ├── swot_agent.py     # SWOT analysis
│   │   ├── metrics_agent.py  # KPI extraction
│   │   ├── decision_agent.py # Investment decision
│   │   └── orchestrator.py   # Multi-agent orchestration
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── sec_helpers.py    # SEC API helpers
│   │   ├── financial_calcs.py # Financial calculations
│   │   └── logger.py         # Logging configuration
│   └── app/
│       ├── __init__.py
│       ├── streamlit_app.py  # Main UI
│       └── components/        # UI components
│           ├── __init__.py
│           ├── sidebar.py
│           ├── dashboard.py
│           └── chat.py
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py
│   ├── test_agents.py
│   └── test_vectordb.py
└── scripts/
    ├── setup_milvus.py       # Initialize Milvus collections
    ├── process_filing.py     # Process single filing
    └── batch_process.py      # Batch processing
```

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Setup Milvus**
   ```bash
   # Option 1: Docker (recommended)
   docker-compose up -d
   
   # Option 2: Milvus Lite (embedded)
   # No setup needed, runs in-process
   ```

4. **Initialize Database**
   ```bash
   python scripts/setup_milvus.py
   ```

5. **Process Sample Filing**
   ```bash
   python scripts/process_filing.py --ticker AAPL --year 2023
   ```

6. **Run Application**
   ```bash
   streamlit run src/app/streamlit_app.py
   ```

## Key Technologies

- **Data Pipeline**: sec-edgar-downloader, pdfplumber, Camelot, LayoutParser
- **Vector DB**: Milvus (with OpenAI embeddings)
- **LLM**: OpenAI GPT-4
- **UI**: Streamlit
- **Processing**: pandas, numpy, python-docx

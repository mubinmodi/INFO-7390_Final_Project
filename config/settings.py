"""
Configuration management for SEC Filing Analyzer
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, LOG_DIR, 
                  DATA_DIR / "raw", 
                  DATA_DIR / "processed",
                  DATA_DIR / "embeddings",
                  DATA_DIR / "metadata"]:
    directory.mkdir(parents=True, exist_ok=True)

# Google Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
EMBEDDING_DIMENSION = 768  # Gemini embedding dimension

# Milvus Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_USER = os.getenv("MILVUS_USER", "")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
USE_MILVUS_LITE = os.getenv("USE_MILVUS_LITE", "true").lower() == "true"

# Collection names
COLLECTION_SECTIONS = "sec_filing_sections"
COLLECTION_TABLES = "sec_filing_tables"
COLLECTION_METADATA = "sec_filing_metadata"

# SEC Edgar Configuration
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "Research contact@example.com")

# Data paths
DATA_RAW_PATH = Path(os.getenv("DATA_RAW_PATH", DATA_DIR / "raw"))
DATA_PROCESSED_PATH = Path(os.getenv("DATA_PROCESSED_PATH", DATA_DIR / "processed"))
DATA_EMBEDDINGS_PATH = Path(os.getenv("DATA_EMBEDDINGS_PATH", DATA_DIR / "embeddings"))
DATA_METADATA_PATH = Path(os.getenv("DATA_METADATA_PATH", DATA_DIR / "metadata"))

# Processing Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8000"))

# Agent Configuration
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = Path(os.getenv("LOG_FILE", LOG_DIR / "app.log"))

# SEC Filing Sections (10-K structure)
SEC_10K_SECTIONS = {
    "ITEM_1": "Business",
    "ITEM_1A": "Risk Factors",
    "ITEM_1B": "Unresolved Staff Comments",
    "ITEM_2": "Properties",
    "ITEM_3": "Legal Proceedings",
    "ITEM_4": "Mine Safety Disclosures",
    "ITEM_5": "Market for Registrant's Common Equity",
    "ITEM_6": "Selected Financial Data",
    "ITEM_7": "Management's Discussion and Analysis",
    "ITEM_7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "ITEM_8": "Financial Statements and Supplementary Data",
    "ITEM_9": "Changes in and Disagreements with Accountants",
    "ITEM_9A": "Controls and Procedures",
    "ITEM_9B": "Other Information",
    "ITEM_10": "Directors, Executive Officers and Corporate Governance",
    "ITEM_11": "Executive Compensation",
    "ITEM_12": "Security Ownership of Certain Beneficial Owners",
    "ITEM_13": "Certain Relationships and Related Transactions",
    "ITEM_14": "Principal Accounting Fees and Services",
    "ITEM_15": "Exhibits, Financial Statement Schedules",
}

# Section mappings for agents
SUMMARY_SECTIONS = ["ITEM_1", "ITEM_7"]
SWOT_SECTIONS = ["ITEM_1", "ITEM_1A", "ITEM_7", "ITEM_8"]
METRICS_SECTIONS = ["ITEM_8"]
RISK_SECTIONS = ["ITEM_1A", "ITEM_7A"]

# Financial metrics to extract
KEY_METRICS = [
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "total_assets",
    "total_liabilities",
    "stockholders_equity",
    "cash_from_operations",
    "free_cash_flow",
    "total_debt",
    "current_assets",
    "current_liabilities",
]

# Calculated metrics
CALCULATED_METRICS = [
    "revenue_growth",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "roe",
    "roa",
    "roic",
    "debt_to_equity",
    "current_ratio",
    "quick_ratio",
    "fcf_yield",
]
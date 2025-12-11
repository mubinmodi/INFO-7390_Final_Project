"""
Setup script to initialize Milvus collections
"""
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectordb.milvus_client import MilvusClient
from config.settings import COLLECTION_SECTIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Initialize Milvus collections"""
    logger.info("Starting Milvus setup...")
    
    try:
        # Connect to Milvus
        client = MilvusClient()
        logger.info("Connected to Milvus successfully")
        
        # Create main collection
        logger.info(f"Creating collection: {COLLECTION_SECTIONS}")
        client.create_collection(COLLECTION_SECTIONS)
        
        # Verify creation
        stats = client.get_collection_stats(COLLECTION_SECTIONS)
        logger.info(f"Collection created successfully: {stats}")
        
        logger.info("✅ Milvus setup completed successfully!")
        
        client.close()
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
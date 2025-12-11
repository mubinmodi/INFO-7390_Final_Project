"""
Milvus Vector Database Client
"""
import logging
from typing import Dict, List, Optional
from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType
)
from config.settings import (
    MILVUS_HOST, MILVUS_PORT, MILVUS_USER, MILVUS_PASSWORD,
    USE_MILVUS_LITE, COLLECTION_SECTIONS, EMBEDDING_DIMENSION
)

logger = logging.getLogger(__name__)


class MilvusClient:
    """Client for Milvus vector database operations"""
    
    def __init__(self):
        """Initialize Milvus connection"""
        self.connected = False
        self.collections = {}
        self._connect()
        
    def _connect(self):
        """Establish connection to Milvus"""
        try:
            if USE_MILVUS_LITE:
                # Use embedded Milvus Lite
                connections.connect(
                    alias="default",
                    uri="./data/milvus_lite.db"
                )
                logger.info("Connected to Milvus Lite (embedded)")
            else:
                # Connect to Milvus server
                connections.connect(
                    alias="default",
                    host=MILVUS_HOST,
                    port=MILVUS_PORT,
                    user=MILVUS_USER,
                    password=MILVUS_PASSWORD
                )
                logger.info(f"Connected to Milvus server at {MILVUS_HOST}:{MILVUS_PORT}")
            
            self.connected = True
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def create_collection(self, collection_name: str = COLLECTION_SECTIONS):
        """
        Create collection for SEC filing sections
        
        Args:
            collection_name: Name of the collection
        """
        # Drop existing collection if it exists
        if utility.has_collection(collection_name):
            logger.warning(f"Collection {collection_name} already exists, dropping...")
            utility.drop_collection(collection_name)
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="ticker", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="fiscal_year", dtype=DataType.INT64),
            FieldSchema(name="section_id", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="start_page", dtype=DataType.INT64),
            FieldSchema(name="token_count", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="SEC Filing Sections with Embeddings"
        )
        
        # Create collection
        collection = Collection(
            name=collection_name,
            schema=schema
        )
        
        # Create index on embedding field
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        logger.info(f"Created collection: {collection_name}")
        self.collections[collection_name] = collection
        
        return collection
    
    def insert_chunks(self, 
                     chunks: List[Dict], 
                     embeddings: List[List[float]],
                     collection_name: str = COLLECTION_SECTIONS):
        """
        Insert document chunks with embeddings
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: List of embedding vectors
            collection_name: Name of collection
        """
        if collection_name not in self.collections:
            self.collections[collection_name] = Collection(collection_name)
        
        collection = self.collections[collection_name]
        
        # Prepare data for insertion
        data = [
            [chunk['doc_id'] for chunk in chunks],  # doc_id
            [chunk['chunk_id'] for chunk in chunks],  # chunk_id
            [chunk['ticker'] for chunk in chunks],  # ticker
            [chunk['fiscal_year'] for chunk in chunks],  # fiscal_year
            [chunk['section_id'] for chunk in chunks],  # section_id
            [chunk['text'] for chunk in chunks],  # text
            [chunk['start_page'] for chunk in chunks],  # start_page
            [chunk['token_count'] for chunk in chunks],  # token_count
            embeddings  # embedding
        ]
        
        # Insert data
        collection.insert(data)
        collection.flush()
        
        logger.info(f"Inserted {len(chunks)} chunks into {collection_name}")
    
    def search(self,
              query_embedding: List[float],
              ticker: Optional[str] = None,
              section_id: Optional[str] = None,
              top_k: int = 5,
              collection_name: str = COLLECTION_SECTIONS) -> List[Dict]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query embedding vector
            ticker: Filter by ticker (optional)
            section_id: Filter by section (optional)
            top_k: Number of results to return
            collection_name: Name of collection
            
        Returns:
            List of search results with scores
        """
        if collection_name not in self.collections:
            self.collections[collection_name] = Collection(collection_name)
        
        collection = self.collections[collection_name]
        collection.load()
        
        # Build filter expression
        filter_expr = []
        if ticker:
            filter_expr.append(f'ticker == "{ticker}"')
        if section_id:
            filter_expr.append(f'section_id == "{section_id}"')
        
        expr = " && ".join(filter_expr) if filter_expr else None
        
        # Search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        # Perform search
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["doc_id", "chunk_id", "ticker", "fiscal_year", 
                          "section_id", "text", "start_page", "token_count"]
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    'score': hit.score,
                    'doc_id': hit.entity.get('doc_id'),
                    'chunk_id': hit.entity.get('chunk_id'),
                    'ticker': hit.entity.get('ticker'),
                    'fiscal_year': hit.entity.get('fiscal_year'),
                    'section_id': hit.entity.get('section_id'),
                    'text': hit.entity.get('text'),
                    'start_page': hit.entity.get('start_page'),
                    'token_count': hit.entity.get('token_count')
                })
        
        return formatted_results
    
    def get_by_section(self,
                      ticker: str,
                      fiscal_year: int,
                      section_id: str,
                      collection_name: str = COLLECTION_SECTIONS) -> List[Dict]:
        """
        Retrieve all chunks for a specific section
        
        Args:
            ticker: Company ticker
            fiscal_year: Fiscal year
            section_id: Section identifier
            collection_name: Name of collection
            
        Returns:
            List of chunks
        """
        if collection_name not in self.collections:
            self.collections[collection_name] = Collection(collection_name)
        
        collection = self.collections[collection_name]
        collection.load()
        
        # Query for specific section
        expr = f'ticker == "{ticker}" && fiscal_year == {fiscal_year} && section_id == "{section_id}"'
        
        results = collection.query(
            expr=expr,
            output_fields=["doc_id", "chunk_id", "ticker", "fiscal_year", 
                          "section_id", "text", "start_page", "token_count"]
        )
        
        return results
    
    def delete_document(self, doc_id: str, collection_name: str = COLLECTION_SECTIONS):
        """
        Delete all chunks for a document
        
        Args:
            doc_id: Document identifier
            collection_name: Name of collection
        """
        if collection_name not in self.collections:
            self.collections[collection_name] = Collection(collection_name)
        
        collection = self.collections[collection_name]
        
        # Delete by expression
        expr = f'doc_id == "{doc_id}"'
        collection.delete(expr)
        
        logger.info(f"Deleted document: {doc_id}")
    
    def get_collection_stats(self, collection_name: str = COLLECTION_SECTIONS) -> Dict:
        """
        Get statistics for a collection
        
        Args:
            collection_name: Name of collection
            
        Returns:
            Dictionary with collection statistics
        """
        if collection_name not in self.collections:
            self.collections[collection_name] = Collection(collection_name)
        
        collection = self.collections[collection_name]
        
        stats = {
            'name': collection_name,
            'num_entities': collection.num_entities,
            'description': collection.description
        }
        
        return stats
    
    def close(self):
        """Close Milvus connection"""
        connections.disconnect("default")
        self.connected = False
        logger.info("Disconnected from Milvus")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    client = MilvusClient()
    
    # Create collection
    client.create_collection()
    
    # Get stats
    stats = client.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    client.close()
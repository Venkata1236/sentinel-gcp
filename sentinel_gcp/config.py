"""
Central config — loads environment variables once, used across all nodes
that need API keys or tunable thresholds (per .env.example).
"""
import os
from dotenv import load_dotenv

load_dotenv(encoding="utf-8-sig")  # utf-8-sig strips a BOM if present, behaves
                                     # identically to plain utf-8 if not — safe
                                     # either way. Added after a BOM on .env's
                                     # first line silently broke ANTHROPIC_API_KEY
                                     # loading earlier (dotenv_values showed the
                                     # key as '\ufeffANTHROPIC_API_KEY').


class Settings:
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
    PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.environ.get("PINECONE_INDEX_NAME", "sentinel-gcp-regulations")
    VECTOR_STORE_BACKEND: str = os.environ.get("VECTOR_STORE_BACKEND", "faiss")
    DATABASE_URL: str = os.environ.get("DATABASE_URL", "")
    MAX_EXTRACTION_RETRIES: int = int(os.environ.get("MAX_EXTRACTION_RETRIES", "1"))
    CONFIDENCE_WEIGHT_EXTRACTION: float = float(os.environ.get("CONFIDENCE_WEIGHT_EXTRACTION", "0.3"))
    CONFIDENCE_WEIGHT_RETRIEVAL: float = float(os.environ.get("CONFIDENCE_WEIGHT_RETRIEVAL", "0.3"))
    CONFIDENCE_WEIGHT_LLM_CERTAINTY: float = float(os.environ.get("CONFIDENCE_WEIGHT_LLM_CERTAINTY", "0.4"))


settings = Settings()
class IngestionError(Exception):
    """Raised when PDF ingestion fails."""

class RetrievalError(Exception):
    """Raised when context retrieval fails."""

class ModelError(Exception):
    """Raised when LLM inference fails."""

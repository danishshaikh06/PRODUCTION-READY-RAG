class QdrantConnectionError(Exception):
    """Raised when the Qdrant server cannot be reached."""

    def __init__(self, url: str, original_error: Exception | None = None):
        message = f"Qdrant unreachable at {url}"
        super().__init__(message)
        self.url = url
        self.original_error = original_error


class QdrantCollectionError(Exception):
    """Raised when a Qdrant collection cannot be created or verified."""

    def __init__(self, collection_name: str, original_error: Exception | None = None):
        message = f"Could not create/verify collection: {collection_name}"
        super().__init__(message)
        self.collection_name = collection_name
        self.original_error = original_error


class EmbeddingModelError(Exception):
    """Raised when embedding models cannot be loaded."""

    def __init__(self, message: str = "Could not load embedding models"):
        super().__init__(message)


class QdrantUpdateError(Exception):
    """Raised when updating Qdrant metadata fails."""

    def __init__(self, message: str = "Could not update embedded_at after upsert"):
        super().__init__(message)

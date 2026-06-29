"""
Data pipeline orchestrator — runs ingestion -> cleaning -> metadata -> chunking
in sequence. Order is fixed (each stage depends on the previous one's output).
Stops immediately if any stage raises.
"""

from my_rag_app.logger import get_logger
from my_rag_app.core.ingestion.data_ingestion import IngestionPipeline
from my_rag_app.core.ingestion.data_cleaning import CleaningPipeline
from my_rag_app.core.ingestion.metadata import MetadataPipeline
from my_rag_app.core.ingestion.chunking import ChunkingPipeline

logger = get_logger(__name__)


class DataPipeline:

    def run(self) -> dict:
        reports = {}

        stages = [
            ("ingestion", IngestionPipeline()),
            ("cleaning", CleaningPipeline()),
            ("metadata", MetadataPipeline()),
            ("chunking", ChunkingPipeline()),
        ]

        for name, stage in stages:
            logger.info("=== Starting stage: %s ===", name)
            try:
                reports[name] = stage.run()
            except Exception as e:
                logger.error("Stage '%s' failed | error=%s", name, e)
                raise
            logger.info("=== Finished stage: %s | %s ===", name, reports[name])

        logger.info("Data pipeline complete | %s", reports)
        return reports


if __name__ == "__main__":
    DataPipeline().run()
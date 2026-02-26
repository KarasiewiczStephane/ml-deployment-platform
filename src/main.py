"""Entry point for the ML Deployment Platform.

Starts the FastAPI model serving application with uvicorn.
"""

import uvicorn

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Launch the model serving API."""
    config = load_config()
    serving_config = config.get("serving", {})

    host = serving_config.get("host", "0.0.0.0")
    port = serving_config.get("port", 8000)

    logger.info("Starting ML Deployment Platform API on %s:%d", host, port)

    uvicorn.run(
        "src.serving.app:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()

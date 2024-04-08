import logging

def setup_logger(
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup the logger with the given logging level.

    Args:
        level (int, optional): The logging level to use. Defaults to logging.INFO.

    Returns:
        logging.Logger: The configured logger.
    """
    logging.basicConfig(level=level)
    return logging.getLogger(__name__)

logger = setup_logger()
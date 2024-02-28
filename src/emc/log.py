import json
import logging.config
from pathlib import Path


def setup_logger(name: str) -> logging.Logger:
    """
    Setup up logging capabilities by reading the config as dictionary
    """
    # Read config
    if not setup_logger.loaded_config:
        LOG_DIR = Path(__file__).parent.parent.parent / 'log'
        CONFIG_PATH = LOG_DIR / 'config.json'

        with open(CONFIG_PATH, 'r') as file:
            config = json.load(file)

        # Set all file paths relative to log folder for file handlers
        for handler in config['handlers'].values():
            if 'filename' in handler:
                path = LOG_DIR / handler['filename']
                handler['filename'] = str(path)

        logging.config.dictConfig(config)
        setup_logger.loaded_config = True

    return logging.getLogger(name)


setup_logger.loaded_config = False

if __name__ == '__main__':
    logger = setup_logger(__name__)

    # Example log message (low -> high)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

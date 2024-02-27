import json
import logging.config

from emc.util import Paths


def setup_logging() -> None:
    """
    Setup up logging capabilities by reading the config as dictionary
    """
    # Read config
    CONFIG_PATH = Paths.log() / 'config.json'

    with open(CONFIG_PATH, 'r') as file:
        config = json.load(file)

    # Set all file paths relative to log folder for file handlers
    for handler in config['handlers'].values():
        if 'filename' in handler:
            path = Paths.log() / handler['filename']
            handler['filename'] = str(path)

    logging.config.dictConfig(config)


if __name__ == '__main__':
    setup_logging()
    logger = logging.getLogger(__name__)

    # Example log message (low -> high)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

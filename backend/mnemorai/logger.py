import logging
import os
import sys

from mnemorai.constants.config import config


class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        self.setStream(sys.stdout)
        self.stream = open(sys.stdout.fileno(), "w", encoding="utf-8", closefd=False)


# Configure the root logger
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    level=logging.WARNING,  # Set a higher level for the root logger
    handlers=[
        logging.FileHandler(
            "logs/mnemorai.log", encoding="utf-8"
        ),  # Ensure FileHandler uses UTF-8 encoding
        UTF8StreamHandler(),  # Use the custom UTF-8 StreamHandler
    ],
)

logging_lvl = config.get("LOGGING_LEVEL", "INFO")
if "-debug" in sys.argv:
    logging_lvl = "DEBUG"
logger = logging.getLogger("mnemorai-logger")
logger.setLevel(logging_lvl)
logger.info(f"LOGGING_LEVEL is set to {logging_lvl}")

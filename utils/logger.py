import logging
import os

class InstantFlushFileHandler(logging.FileHandler):
    """
    Subclasă de FileHandler care forțează scrierea imediată pe disc (flush)
    după fiecare linie generată.
    """
    def emit(self, record):
        super().emit(record)
        self.flush()

def get_script_logger(script_name: str):
    os.makedirs("logs", exist_ok=True)
    log_file_path = f"logs/{script_name}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Modificarea mode='a' asigură concatenarea logurilor fără suprascriere trunchiată
    file_handler = InstantFlushFileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
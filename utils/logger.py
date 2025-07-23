from loguru import logger as _logger


def prepare_logger():
    _logger.add("")
    return _logger


logger = prepare_logger()

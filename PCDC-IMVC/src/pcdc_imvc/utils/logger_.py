import logging
import datetime
from pathlib import Path


def get_logger(config):
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger('pcdc_imvc')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    plt_name = str(config['dataset']) + ' ' + str(config['missing_rate']).replace('.','') + ' ' + str(
        datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H-%M-%S'))

    project_root = Path(__file__).resolve().parents[3]
    log_dir = project_root / 'outputs' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / (
        str(config['dataset']) + ' ' + str(config['missing_rate']).replace('.', '') + ' ' +
        str(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H-%M-%S')) + '.log'
    )

    fh = logging.FileHandler(
        str(log_file))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, plt_name
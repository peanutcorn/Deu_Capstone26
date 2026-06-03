import os, sys
import logging
import time
from collections import namedtuple
from pathlib import Path

def create_logger(phase='train'):
    root_output_dir = Path("./output")
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = 'nia'
    model = "convlstm"


    final_output_dir = root_output_dir / dataset / model
    print(final_output_dir)

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format("ConvLSTM", time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)


    return logger, str(final_output_dir)
"""
This file has to be imported first when you use IUVS lib.
"""

## Add some directories to Pythonpath
import sys
from datetime import datetime
workpath = '/Users/masunaga/work/python_git/maven/iuvs/'
dirs = ['test', 'scripts', 'scripts/quicklook', 'lib', 'pfptools']
[sys.path.append(workpath + idir) for idir in dirs]

## import reload function
from importlib import reload

## Setting for ipython to automatically reload modificationds
from IPython import get_ipython
ipython = get_ipython()
ipython.magic("%load_ext autoreload")
ipython.magic("%autoreload 2")

# Make log file # check https://qiita.com/mimitaro/items/9fa7e054d60290d13bfc
from common.tools import MyLogger
import logging
from logging import getLogger, StreamHandler, Formatter
mylogger = MyLogger()
mylogger.start()
"""logger = getLogger("LogInit")
logger.setLevel(logging.DEBUG)
stream_handler = StreamHandler()
stream_handler.setLevel(logging.DEBUG)
handler_format = Formatter('%(asctime)s ---- %(name)s  %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)
logger.debug('---- Read init function ----')
(lambda Dt: '{:04d}/{:02d}/{:02d}-{:02d}:{:02d}:{:02d}'.format(Dt.year, Dt.month, Dt.day, Dt.hour, Dt.minute, Dt.second))(datetime.now())"""

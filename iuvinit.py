"""
This file has to be imported first when you use IUVS lib.
"""

## Add some directories to Pythonpath
import sys
workpath = '/Users/masunaga/work/python_git/maven/iuvs/'
dirs = ['test', 'scripts', 'scripts/quicklook', 'lib', 'pfptools']
[sys.path.append(workpath + idir) for idir in dirs]

## import reload function
from importlib import reload

## Setting for ipython to automatically reload modifications
from IPython import get_ipython
ipython = get_ipython()
ipython.magic("%load_ext autoreload")
ipython.magic("%autoreload 2")

print('---- Read init function ----')

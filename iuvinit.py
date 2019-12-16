"""
This file has to be imported first when you use IUVS lib.
"""

## Add some directories to Pythonpath
import sys
workpath = '/Users/masunaga/work/python_git/py_iuvs/'
dirs = ['test', 'script']
[sys.path.append(workpath + idir) for idir in dirs]

import sys
import os

def bar_system():
    bar = '?'
    if sys.platform.startswith("linux"):  # could be "linux", "linux2", "linux3", ...
        # linux
        bar = '/'
    elif sys.platform == "darwin":
        # MAC OS X
        bar = '/'
    elif os.name == "nt":
        # Windows, Cygwin, etc. (either 32-bit or 64-bit)
        bar = '\\'
    return bar
import platform

# Dependencies
import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib

# Export functions
from readECG import *
from signal import *
from misc import *
from stats import *


# Info
__version__ = "1.0.0"


# Maintainer info
__author__ = "Chinmaya Behera, Saunak Samantray"
__email__ = "chinmayabehera909@gmail.com"


# Citation
__bibtex__ = r"""
@misc{EGC Analysis Tool,
  author = {Chinmaya Behera, Saunak Samantray},
  title = {A Python Toolbox for ECG processing and classification},
  month={August},
  year = {2021},
}
"""


# =============================================================================
# Helper functions to retrieve info
# =============================================================================
def cite():
        return __bibtex__


def version(silent=False):
    
    if silent is False:
        print(
            "- OS: " + platform.system(),
            "(" + platform.architecture()[1] + " " + platform.architecture()[0] + ")",
            "\n- Python: " + platform.python_version(),
            "\n- ECG_Analysis: " + __version__,
            "\n\n- NumPy: " + np.__version__,
            "\n- Pandas: " + pd.__version__,
            "\n- SciPy: " + scipy.__version__,
            "\n- sklearn: " + sklearn.__version__,
            "\n- matplotlib: " + matplotlib.__version__,
        )
    else:
        return __version__

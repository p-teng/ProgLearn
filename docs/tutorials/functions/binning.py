import numpy as np
import math
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
from sklearn.datasets import load_breast_cancer

def KBinsDiscretize(data_x, n_bins=0, alpha=3.322, encode="ordinal", strategy="uniform"):
    """
        
    """
   # Makes n_bins optional, calculates optimal n_bins by default
   # Sturges Rule - num_bins = 1 + 3.322 * log_10(num_inputs)
    if n_bins == 0:
        # cap bins at 256
        n_bins = min(math.floor(1 + alpha * math.log10(data_x[0].shape), 256))

    kbins = KBinsDiscretizer(n_bins, encode='ordinal', strategy='uniform')
    kbins.fit(data_x)
    binned_x = kbins.transform(data_x)
    return binned_x
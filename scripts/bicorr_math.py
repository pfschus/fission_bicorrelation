# A few convenient math functions for the bicorr project



import os
import os.path
import time
import numpy as np
np.set_printoptions(threshold=np.nan) # print entire matrices
import pandas as pd
import scipy.io as sio
import sys
import matplotlib
import seaborn as sns
sns.set(style='ticks')

#matplotlib.use('agg') # for flux
import matplotlib.pyplot as plt
import time
from tqdm import *

from bicorr import *
from bicorr_sums import *
from bicorr_plot import * 


def prop_err_division(num,num_err,denom,denom_err):
    A = num/denom
    
    A_err = A*np.sqrt((num_err/num)**2+(denom_err/denom)**2)
    return A, A_err




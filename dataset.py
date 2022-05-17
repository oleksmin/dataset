import random, psutil
import matplotlib.pyplot as plt
import pandas as pd

def dataset():
    random.seed(int(psutil.cpu_percent(1)*psutil.virtual_memory()[3]%psutil.virtual_memory()[4]))

    



if __name__=="__main__":
    dataset()
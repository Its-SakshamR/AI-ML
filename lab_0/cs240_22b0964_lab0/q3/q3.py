from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def func(t, v, k):
    """ computes the function S(t) with constants v and k """
    
    # TODO: return the given function S(t)
    return (v*(t-(1/k)*(1-np.exp(-1*k*t))))

    # END TODO


def find_constants(df: pd.DataFrame, func: Callable):
    """ returns the constants v and k """

    v = 0
    k = 0

    # TODO: fit a curve using SciPy to estimate v and k
    a, b = curve_fit(func, df['t'], df['S'])
    v = a[0]
    k = a[1]
    # END TODO

    return v, k


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    v, k = find_constants(df, func)
    v = v.round(4)
    k = k.round(4)
    print(v, k)

    # TODO: plot a histogram and save to fit_curve.png
    plt.scatter(df['t'],df['S'], color='blue',marker="*", label='data')
    x=np.linspace(0,0.40,10)
    plt.plot(x, func(x,v,k),color='red', label = f'fit: v='+str(v)+' k='+str(k))
    plt.xlabel('t')
    plt.ylabel('S')
    plt.legend()
    plt.savefig('fit_curve.png')
    plt.clf()
    # END TODO

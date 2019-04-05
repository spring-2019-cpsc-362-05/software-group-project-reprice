import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("tuning_results/state_level").values
    plt.plot(df[:, 3])
    plt.show()


if __name__=='__main__':main()
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

## Correlation Matrix function
def CorrelationTable(df, width, height):

    # Create Correlation df from source df
    corr = df.corr()
    # Plot figsize
    fig, ax = plt.subplots(figsize=(width, height))
    # Drop self-correlations
    dropSelf = np.zeros_like(corr)
    dropSelf[np.triu_indices_from(dropSelf)] = True 

    # Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(corr, cmap="RdBu", annot=True, fmt=".2f", mask=dropSelf, 
        xticklabels=corr.columns, 
            yticklabels=corr.columns, ax=ax, linewidths=.5, cbar_kws={"shrink": .7},
            vmin = -1, vmax=1, center=0)
    plt.title('Correlation HeatMap',fontsize=14)
    plt.show()  
    return
 
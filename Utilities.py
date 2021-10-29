# Utilities.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def MSE(y,yh):
    mse = ((y-yh)*(y-yh)).mean()
    return mse

def ErrorRate(y,yh):
    err = (y!=yh).mean()
    return err

def PlotLearningCurves(learning_curves,title=''):
    ax1 = learning_curves.plot.line(x='epoch',y='loss')
    learning_curves.plot.line(x='epoch',y='val_loss',
            ax=ax1,
            grid=True,
            title=title)
    plt.show()

def PlotImages(ix_start,num_img,rows,cols,images,labels,color=True):
    plt.figure(figsize=(10,10))
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    for i in range(num_img):
        plt.subplot(rows,cols,i+1)
        plt.axis('off')
        if color:
            plt.imshow(images[ix_start+i])
        else: 
            plt.imshow(np.squeeze(images[ix_start+i]),cmap='Greys')
        plt.title(labels[ix_start+i])

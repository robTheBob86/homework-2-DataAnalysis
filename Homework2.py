#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 14:09:22 2018

@author: robertbaumgartner
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')

from nflparser import parse_data
from sklearn.cross_decomposition import PLSRegression
from copy import deepcopy

def do_boxplots(df, df_norm):

    plt.rcParams["figure.figsize"] = [15,8]
    df.iloc[:, 1:-1].boxplot(rot = 70, showfliers = False)
    plt.title('Boxplot of the data')
    plt.show()

    plt.rcParams["figure.figsize"] = [15,8]
    df_norm.boxplot(rot = 70, showfliers = False)
    plt.title('Normalized data')
    plt.show()


def extract_features(df, df_norm):
    
    # use the corr_limit to change the number of components considered
    corr_limit = 0.5
        
    data_norm = df_norm.values.astype(float)
    corr_coef = np.corrcoef(data_norm.T)
    
    fig = plt.figure(figsize=(15, 15))
    
    #ax = fig.add_subplot(111)
    fig, ax = plt.subplots()
    ax.set_title('Pearson Correlation Coefficients Colorplot')
    plt.imshow(corr_coef)
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()
    
    values = [d for d in corr_coef[0, :] if abs(d) >= corr_limit]
    #indexes = np.where(corr_coef[0, :] > 0.6) # or corr_coef[0,:] <= 0.6)
    ix =  np.isin(corr_coef[0, :], values)
    indexes = np.where(ix)[0]
    np.insert(indexes, 0, 0)
    indexes = np.add(indexes, np.ones(indexes.shape[0])).astype(int) # plus one, because we added the team-name thereafter
    
    headers = list(df)
    new_df = {}
    
    for i in range(len(indexes)):
        new_df[df_norm.columns[i]] = df_norm.iloc[:, i]
     
    new_df = pd.DataFrame(new_df, dtype = 'float') 
    new_df.insert(0, "team", df["team"].astype(str))
    
    return (indexes,new_df, values)


def crossplot_variables(df): # we pass the new_df
    
    var_names = list(df)
    data = df.iloc[:, 1:].values.astype(float)
    
    # NOTE: it is reasonable here to do subplots as well
    for i in range(len(data[0,:])):
        
        var_name = var_names[i + 1]
        
        axis = range(1, data.shape[0] + 1)
        fig, ax = plt.subplots(figsize = (10,10))
        ax.plot(axis, data[:,0], label = "Number of wins")
        ax.plot(axis, data[:,i], label = var_name)
        ax.set_title("Variables over Team")
        ax.set_xlabel("Team")
        ax.set_ylabel("Data")
        ax.legend()

def scores_plots_plsr(X, Y):

    plsr = PLSRegression(n_components = 2)
    plsr.fit(X,Y)
    T = plsr.x_scores_
    P = plsr.x_loadings_
    U = plsr.y_scores_
    Q = plsr.y_loadings_
    
    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(T[:, 0], Y[:, 0])
    ax.scatter(T[:, 0], Y[:, 0])
    ax.grid(b=True, which='major')
    ax.set_title("First scores vector vs. Wins, PLSR")
    ax.set_xlabel("Scores")
    ax.set_ylabel("Wins")
    
    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(T[:, 1], Y[:, 0])
    ax.scatter(T[:, 1], Y[:, 0])
    ax.grid(b=True, which='major')
    ax.set_title("Second scores vector vs. Wins, PLSR")
    ax.set_xlabel("Scores")
    ax.set_ylabel("Wins")
    
    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(T[:, 0], Y[:, 1])
    ax.scatter(T[:, 0], Y[:, 1])
    ax.grid(b=True, which='major')
    ax.set_title("First scores vector vs. Looses, PLSR")
    ax.set_xlabel("Scores")
    ax.set_ylabel("Wins")
    
    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(T[:, 1], Y[:, 1])
    ax.scatter(T[:, 1], Y[:, 1])
    ax.grid(b=True, which='major')
    ax.set_title("Second scores vector vs. Looses, PLSR")
    ax.set_xlabel("Scores")
    ax.set_ylabel("Wins")
    
def scores_plots_pcr(X, Y):
    
    [U, sigma, V] = np.linalg.svd(X)
    
    #accumulated singular values
    fig, ax = plt.subplots()
    ax.plot(np.arange(1,len(sigma)+1),100*np.cumsum(sigma)/np.sum(sigma),'o-')
    ax.set(xlabel='Number of Factors', ylabel='Accumulated Singular Values', title='Explained Variance')
    ax.grid(b=True, which='major')
    plt.xlim((0,len(sigma+1)))
    plt.ylim((0,105))
    
    # plot the wins and looses over scores
    scores = U.T
    Y_projected = np.dot(scores, Y)
    
    axis = range(0, Y_projected.shape[0])
    fig, ax = plt.subplots(figsize = (18,10))
    ax.plot(axis, Y_projected[:,0], label = "Wins projected, PCR")
    ax.plot(axis, Y_projected[:,1], label = "Looses projected, PCR")
    ax.grid(b=True, which='major')
    ax.set_title("Projected data, PCR")
    ax.set_xlabel("Scores number")
    ax.set_ylabel("Wins and looses")
    ax.legend()
    
    # As can be seen here from this plot, the first and second scores project 
    # the wins into a negative range. The opposite is of course true for the looses.
    # From this we can conclude that the first two scores somewhat gather the 
    # features that the wins correlate negatively with. From the Pearson Correlation
    # Coefficients Colorplot we can obtain the names of those.

    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(U[:, 0], Y[:, 0])
    ax.scatter(U[:, 0], Y[:, 0])
    ax.grid(b=True, which='major')
    ax.set_title("First scores vector vs. Wins, PCR")
    ax.set_xlabel("Scores")
    ax.set_ylabel("Wins")
    
    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(U[:, 1], Y[:, 0])
    ax.scatter(U[:, 1], Y[:, 0])
    ax.grid(b=True, which='major')
    ax.set_title("Second scores vector vs. Wins, PCR")
    ax.set_xlabel("Scores")
    ax.set_ylabel("Wins")
    
    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(U[:, 0], Y[:, 1])
    ax.scatter(U[:, 0], Y[:, 1])
    ax.grid(b=True, which='major')
    ax.set_title("First scores vector vs. Looses, PCR")
    ax.set_xlabel("Scores")
    ax.set_ylabel("Wins")
    
    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(U[:, 1], Y[:, 1])
    ax.scatter(U[:, 1], Y[:, 1])
    ax.grid(b=True, which='major')
    ax.set_title("Second scores vector vs. Looses, PCR")
    ax.set_xlabel("Scores")
    ax.set_ylabel("Wins")
    
    return sigma.shape[0]
    


if __name__ == "__main__":
    
    df = parse_data()
    
    means = df.mean()
    std = df.std()
    df_norm = df.iloc[:, 1:-1].subtract(means)
    df_norm = df_norm.divide(std)
    
    # ------------------------Kick out non-relevant features
    # NOTE on the following plot: The white area is the number of games. As each 
    # tema has played the same number of games, the standard deviation is zero,
    # and the normalization returns inf. we don't consider it here, it will be 
    # sorted out by the extract_features() function anyways.
    (indexes, new_df, correlations) = extract_features(df, df_norm)
    # from the correlation plot we do find that, in order to predict the wins and 
    # losses, variable 65 is the most important, as it correlates the most.
    # others we can take in are given by indexes. 
    
    
    # ---------------------- Do some plots here, find them in the .ppt-file
    do_boxplots(df, df_norm)
    do_boxplots(df, new_df)
    
    crossplot_variables(new_df)
    
    
    
    # ---------------------- Task number 3:PCR-----------------------------------
    
    number_of_teams = new_df.shape[0]
    X = new_df.iloc[:, 3:].values.astype(float)
    #Y = df.iloc[:, 1:3].values.astype(int)
    Y = new_df.iloc[:, 1:3].values.astype(float) 
    
    max_components = scores_plots_pcr(X, Y) # it may actually be confusing that this function returns the max number of components, but as it is we will need it and can save some values there
    
    # As can be seen here from this plot, the first and second scores project 
    # the wins into a negative range. The opposite is of course true for the looses.
    # From this we can conclude that the first two scores somewhat gather the 
    # features that the wins correlate negatively with. From the Pearson Correlation
    # Coefficients Colorplot we can obtain the names of those.
    
    # predict wins and losses using PCR and find the best model using cross validation
    errors = np.zeros(X.shape)
    for i in range(1, max_components):
        for j in range(max(number_of_teams, 30)): # because we have only 30 teams and little data, we can do leave-one-out cross validation 
            
            X_train = np.delete(X, j, 0)
            Y_train = np.delete(Y, j, 0) 
            [U_train, sigma_train, V_train] = np.linalg.svd(X_train, full_matrices = False)

            scores = U_train[:, 0:i] # for cross validation, we take the j-th row out
            loadings = np.dot(np.diag(sigma_train), V_train)[0:i, :]
            X_inv = np.linalg.pinv(np.dot(scores, loadings)) # scores*loadings is the X matrix we use for PCR
            w = np.dot(X_inv, Y_train[:, 0]) # the weight vector regarding the wins

            y_pred = np.dot(X[j, :], w)
            # take the error on the wins 
            errors[j, i] = abs(Y[j, 0] - y_pred)
    errors = np.delete(errors, 0, 1) # because we omit the 0-th component
    errors = np.mean(errors, axis = 0)
    pcr_errors = deepcopy(errors)# delete

    
    axis = range(1, errors.shape[0]+1)
    fig, ax = plt.subplots(figsize = (18,10))
    ax.plot(axis, errors)
    ax.grid(b=True, which='major')
    ax.set_title("Number of Components vs. Mean of Error, PCR")
    ax.set_xlabel("n_components")
    ax.set_ylabel("Mean error")
    
    # from the plot it appears that 3 or 10 are a good values for the components. 
    # while 10 has the lowest error, 3 has less components while performing very 
    # sightly better. Please note the small range on the y-axis
    
    
    # ---------------------- Task number 4: PLSR-------------------------------
    
    # first plot the scores vs wins and looses, like in the PCR part
    
    scores_plots_plsr(X, Y)
    
    # here we can see that the first scores are correlated positively with the wins, 
    # while negative to looses (symmetric). it must resemble variables that are 
    # positively correlated with wins. 
    # the second scores is hard to interpret.
    
    # cross validation and finding optimal number of components utilizing the mean error
    errors = np.zeros(X.shape)
    for i in range(1, max_components): 
        for j in range(max(number_of_teams, 30)):
            
            X_train = np.delete(X, j, 0)
            Y_train = np.delete(Y, j, 0) 
            
            plsr = PLSRegression(n_components = i)
            plsr.fit(X_train,Y_train)
            y_pred = plsr.predict(X[j, :].reshape(1, -1))[0]
            errors[j, i] = abs(Y[j, 0] - y_pred[0]) # y_pred here is an array of length 2, because y_train has wins and looses

    errors = np.delete(errors, 0, 1) # because we don't have zero components, we delete this columnn
    errors = np.mean(errors, axis = 0)
    
    axis = range(1, errors.shape[0]+1)
    fig, ax = plt.subplots(figsize = (18,10))
    ax.plot(axis, errors)
    ax.grid(b=True, which='major')
    ax.set_title("Number of Components vs. Mean of Error, PLSR")
    ax.set_xlabel("n_components")
    ax.set_ylabel("Mean error")
    
    # compare PCR and PLSR
    fig, ax = plt.subplots(figsize = (18,10))
    ax.plot(axis, errors, label = "PLSR")
    ax.plot(axis, pcr_errors, label = "PCR")
    ax.grid(b=True, which='major')
    ax.set_title("Number of Components vs. Mean of Error")
    ax.set_xlabel("n_components")
    ax.set_ylabel("Mean error")
    ax.legend()
    # as can be seen, the performance of the two methods is in about the same range. PCR performs slightly better
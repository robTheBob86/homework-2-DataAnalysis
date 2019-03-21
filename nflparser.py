#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 14:09:22 2018

@author: robTheBob86
"""

import pandas as pd
import numpy as np

def isint(n):
    try:
        int(n)
        return True
    except ValueError:
        return False
    
def return_row(row):
    
    try:
        row = row.split()
    except:
        return
    
    #NOTE: the names of the teams will only either have a length of two or three 
    # words, i.e. only two cases we do need to consider here
    if isint(row[3]):
        row[1] = row[1] + " " + row[2]
        row.remove(row[2])
    elif isint(row[4]): 
        row[1] = row[1] + " " + row[2] + " " + row[3]
        row.remove(row[3])
        row.remove(row[2]) 
    
    return np.array(row).T

def parse_data():
    
    data = pd.read_csv('NFL.txt', header = None) # actually, it is quite not right to use it like that, but i just realized now and the deadline is soon, I'll keep it and fix it when I find the time
    varnames = [] # variable names as described in NFL.txt
    intervals = [] # intervals of variables as described in NFL.txt
    num_of_observations = len(data.iloc[:,0])
    
    with open("NFL_description.txt", "r") as f:
        for line in f:
            line.strip()
            line = line.split()
            try:
                # rule out two distinct cases: team name is either 3 words long or two words long
                if isint(line[2]) and isint(line[0]):
                    varnames.append(line[3])
                    intervals.append([int(line[0])-1, int(line[2])])
                elif isint(line[0]):
                    varnames.append(line[1])
                    intervals.append([int(line[0])-1, int(line[0])])
            except: 
                pass
    
    data_matr = np.zeros([75])
    data_matr = np.vstack((data_matr, varnames))
    data_matr = np.delete(data_matr,0, 0) # take out all the zeros
    
    for i in range(num_of_observations):
        observation = data.iloc[i,0]
        data_matr = np.vstack((data_matr, return_row(observation)))
    
    labels = data_matr[0,:]
    data_matr = data_matr[1:, :]
    
    # Convert it to a dataframe 
    df = {}
    for i in range(1, len(labels)):
        df[labels[i]] = data_matr[:, i]
    
    df = pd.DataFrame(df, dtype = 'float') 
    
    return(df)

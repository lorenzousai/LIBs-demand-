import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import math
import pandas as pd


def assign_index_material (df):
    df = df.reset_index()
    df = df.rename(columns={'level_0': 'segment', 'level_1': 'chemistry', 'level_2': 'material' }) 
    df = df.set_index(['segment','chemistry','material'])
    #df = df.divide(1e9)
    return(df)

def assign_index_capacity (df):
    df = df.reset_index()
    df = df.rename(columns={'level_0': 'segment', 'level_1': 'chemistry' }) 
    df = df.set_index(['segment','chemistry'])
    #df = df.divide(1e6)
    return(df)

def stock_additions_segmented (share_segments, raw_vehicle_stock):
    
# This function takes the stock in each year and calculates the stock additions for the period 2016-2060
# Year 2015 is the total initial stock in that year
#Furthermore, it divides the df in segments
    
    #Clean up the df
    
    raw_vehicle_stock = raw_vehicle_stock.iloc[:,7:]
    raw_vehicle_stock = raw_vehicle_stock.reset_index()
    raw_vehicle_stock = raw_vehicle_stock.drop(['index'], axis = 1)
    raw_vehicle_stock.index = ['share']
    
    #Calculate capacity addition per year
   # stock_addition = raw_vehicle_stock.diff(axis = 1)
    #stock_addition[2015] = raw_vehicle_stock[2015]
    
    stock_segmented = share_segments.dot(raw_vehicle_stock)
    return(stock_segmented)


def calculate_eol(index, years_array, start_year, prob_data, stock_add_df):
   
    
    ## Trying to calculate eol for all the years
    prob_data = prob_data.transpose()
    prob_data = pd.concat([prob_data]*len(stock_add_df.groupby(level=0)))
    
    prob_data = prob_data.reset_index()
    prob_data.drop('index', inplace = True, axis = 1)
    prob_data = prob_data.reindex(index, level = 0)
    prob_data.columns = pd.Index(np.arange(start_year, start_year+25))
   
    testing_testing = pd.concat([stock_add_df[start_year]]*25, axis = 1)
    
    testing_testing.columns =  pd.Index(np.arange(start_year, start_year+25))
    
    eol = testing_testing.mul(prob_data)
    return(eol)

def get_share(stock, BEVs, PHEVs, ICEVG, ICEVD):
    BEV_share = BEVs.div(stock)
    PHEV_share = PHEVs.div(stock)
    ICEV_tot = ICEVG + ICEVD
    ICEV_share = ICEV_tot.div(stock)
    frames = [BEV_share,PHEV_share,ICEV_share]
    out = pd.concat(frames)
    out.index = ['BEV','PHEV','ICEV']
    return(out)


def main ():

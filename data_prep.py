# %%
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import math
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline


# %%
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
# %%
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


def data_read_manipulation():

    raw_data_inflows = pd.read_excel('ODYM_RECC.xls','Model_Results')
    segment = ['A','B','C','D','E','F','J']
    share = ['0.08','0.2056','0.2658','0.0679','0.0287', '0.0021','0.3499']
    share = np.array(share,dtype=float)
    share_df = pd.DataFrame(share,columns=['share'])


    # * Filter data and save it in different dataframes
    # * Starting with BEVs

    BEV_inflows = raw_data_inflows[raw_data_inflows['Indicator'].str.contains('final consumption (use phase inflow), Battery Electric Vehicles (BEV)', regex = False)]

    #Baseline - LED
    BEV_inflows_base_LED =  BEV_inflows[BEV_inflows['SocEc scen'].str.contains('LED', regex = False)]
    BEV_inflows_base_LED =  BEV_inflows_base_LED[BEV_inflows_base_LED['ClimPol scen'].str.contains('Baseline', regex = False)]

    #RCP2.6 - LED
    BEV_inflows_RCP26_LED =  BEV_inflows[BEV_inflows['SocEc scen'].str.contains('LED', regex = False)]
    BEV_inflows_RCP26_LED =  BEV_inflows_RCP26_LED[BEV_inflows_RCP26_LED['ClimPol scen'].str.contains('RCP2.6', regex = False)]

    #Baseline - SSP1
    BEV_inflows_base_SSP1 = BEV_inflows[BEV_inflows['SocEc scen'].str.contains('SSP1', regex = False)]
    BEV_inflows_base_SSP1 = BEV_inflows_base_SSP1[BEV_inflows_base_SSP1['ClimPol scen'].str.contains('Baseline', regex = False)]

    #RCP2.6 - SSP1
    BEV_inflows_RCP26_SSP1 = BEV_inflows[BEV_inflows['SocEc scen'].str.contains('SSP1', regex = False)]
    BEV_inflows_RCP26_SSP1 = BEV_inflows_RCP26_SSP1[BEV_inflows_RCP26_SSP1['ClimPol scen'].str.contains('RCP2.6', regex = False)]

    #RCP2.6 - SSP2
    BEV_inflows_RCP26_SSP2 = BEV_inflows[BEV_inflows['SocEc scen'].str.contains('SSP2', regex = False)]
    BEV_inflows_RCP26_SSP2 = BEV_inflows_RCP26_SSP2[BEV_inflows_RCP26_SSP2['ClimPol scen'].str.contains('RCP2.6', regex = False)]

    #Baseline - SSP2
    BEV_inflows_base_SSP2 = BEV_inflows[BEV_inflows['SocEc scen'].str.contains('SSP2', regex = False)]
    BEV_inflows_base_SSP2 = BEV_inflows_base_SSP2[BEV_inflows_base_SSP2['ClimPol scen'].str.contains('Baseline', regex = False)]

# * Now do the same to PHEVs

    PHEV_inflows = raw_data_inflows[raw_data_inflows['Indicator'].str.contains('final consumption (use phase inflow), Plugin Hybrid Electric Vehicles (PHEV)', regex = False)]

    ################################################# LED ###############################
    PHEV_inflows_base_LED =  PHEV_inflows[PHEV_inflows['SocEc scen'].str.contains('LED', regex = False)]
    PHEV_inflows_base_LED =  PHEV_inflows_base_LED[PHEV_inflows_base_LED['ClimPol scen'].str.contains('Baseline', regex = False)]

    PHEV_inflows_RCP26_LED =  PHEV_inflows[PHEV_inflows['SocEc scen'].str.contains('LED', regex = False)]
    PHEV_inflows_RCP26_LED =  PHEV_inflows_RCP26_LED[PHEV_inflows_RCP26_LED['ClimPol scen'].str.contains('RCP2.6', regex = False)]


    #########################################################################################
    #RCP2.6 - SSP1
    PHEV_inflows_base_SSP1 = PHEV_inflows[PHEV_inflows['SocEc scen'].str.contains('SSP1', regex = False)]
    PHEV_inflows_base_SSP1 = PHEV_inflows_base_SSP1[PHEV_inflows_base_SSP1['ClimPol scen'].str.contains('Baseline', regex = False)]

    #RCP2.6 - SSP1
    PHEV_inflows_RCP26_SSP1 = PHEV_inflows[PHEV_inflows['SocEc scen'].str.contains('SSP1', regex = False)]
    PHEV_inflows_RCP26_SSP1 = PHEV_inflows_RCP26_SSP1[PHEV_inflows_RCP26_SSP1['ClimPol scen'].str.contains('RCP2.6', regex = False)]

    #RCP2.6 - SSP2
    PHEV_inflows_RCP26_SSP2 = PHEV_inflows[PHEV_inflows['SocEc scen'].str.contains('SSP2', regex = False)]
    PHEV_inflows_RCP26_SSP2 = PHEV_inflows_RCP26_SSP2[PHEV_inflows_RCP26_SSP2['ClimPol scen'].str.contains('RCP2.6', regex = False)]

    #Baseline - SSP2
    PHEV_inflows_base_SSP2 = PHEV_inflows[PHEV_inflows['SocEc scen'].str.contains('SSP2', regex = False)]
    PHEV_inflows_base_SSP2 = PHEV_inflows_base_SSP2[PHEV_inflows_base_SSP2['ClimPol scen'].str.contains('Baseline', regex = False)]



    ## Create array of dfs for easier looping and manipulation
    # * BEVs
    BEVs_inflows_array = [BEV_inflows_RCP26_SSP2, BEV_inflows_RCP26_SSP1, BEV_inflows_RCP26_LED, 
        BEV_inflows_base_SSP2, BEV_inflows_base_SSP1, BEV_inflows_base_LED]

    # * PHEVs   
    PHEVs_inflows_array = [PHEV_inflows_RCP26_SSP2, PHEV_inflows_RCP26_SSP1, PHEV_inflows_RCP26_LED, 
        PHEV_inflows_base_SSP2, PHEV_inflows_base_SSP1, PHEV_inflows_base_LED]

    
    ### Read data for chemistries market share in given years
    chemistries = pd.read_excel('Test_chemistries.xlsx', sheet_name = 'Sheet1', skiprows=24, nrows = 8, usecols = 'B:AV')
    chemistries = chemistries.set_index(['chemistry'])
    chemistries = chemistries.interpolate(method = 'linear',  axis = 1)

    ### Read average battery size in each segment and forecasts for future battery size for BEV
    # * Interpolate battery size for missing years
    batt_size_BEV = pd.read_excel('Test_chemistries.xlsx', sheet_name = 'Batt_size', skiprows=2, nrows = 7, usecols = 'C:AW')
    batt_size_BEV = batt_size_BEV.set_index("Segment")
    batt_size_BEV = batt_size_BEV.interpolate(method = "linear", axis = 1)
    batt_size_BEV = batt_size_BEV.round()
    batt_size_BEV = batt_size_BEV.reset_index()
    batt_size_BEV.drop('Segment', inplace = True, axis = 1)


    ## Do a similar thing for PHEVs, but in this case we assume the battery size remains constant over time 
    batt_size_PHEV = pd.read_excel('Test_chemistries.xlsx', sheet_name = 'BEV_data', skiprows=20, nrows = 7, usecols = 'B:C')
    batt_size_PHEV = batt_size_PHEV.set_index("Segment")
    batt_size_PHEV = batt_size_PHEV.reset_index()
    batt_size_PHEV.drop('Segment', inplace = True, axis = 1)
    batt_size_PHEV.columns = [2015]
    
    ### Read material loading in kg/kwh for each chemistry analysed and prepare df
    material = pd.read_excel('Test_chemistries.xlsx', sheet_name = 'Material composition', skiprows=1, nrows = 8, usecols = 'B:M')
    material = material.set_index(['chemistry'])
    

    ### Prepare multiindex to be used for each BEV and PHEV inflow df
    segments_index = stock_additions_segmented(share_df, BEVs_inflows_array[0])
    
    chem_index = pd.MultiIndex.from_product([segments_index.index.to_list(),
        chemistries.index.to_list()])

    segments_chemistries_materials_index = pd.MultiIndex.from_product([segments_index.index.to_list(),
    chemistries.index.to_list(), 
    material.transpose().index.to_list()])   

    material = material.stack()
    material = material.to_frame()
    material.index.names = ['','']
    material.columns = ['']

## For loop steps: 
    # * Break down inflows by segment;
    # * convert inflows to units (from million units)
    # * Reindex dataframe to have chemistries as level 0
    # * multiply EVs inflows segmented with chemistries market share in each year

    BEV_split_chem_array = [None]*len(BEVs_inflows_array)
    PHEV_split_chem_array = [None]*len(PHEVs_inflows_array)

    for i in range(len(BEVs_inflows_array)):
        BEVs_inflows_array[i] = stock_additions_segmented(share_df, BEVs_inflows_array[i])
        BEVs_inflows_array[i] = BEVs_inflows_array[i].multiply(1e6).round()
        BEV_split_chem_array[i] = BEVs_inflows_array[i].reindex(chem_index, level = 0)
        BEV_split_chem_array[i] = BEV_split_chem_array[i].multiply(chemistries, level = 1)

    for i in range(len(PHEVs_inflows_array)):
        PHEVs_inflows_array[i] = stock_additions_segmented(share_df, PHEVs_inflows_array[i])
        PHEVs_inflows_array[i] = PHEVs_inflows_array[i].multiply(1e6)
        PHEV_split_chem_array[i] = PHEVs_inflows_array[i].reindex(chem_index, level = 0)
        PHEV_split_chem_array[i] = PHEV_split_chem_array[i].multiply(chemistries, level = 1) 

    # * Crate dataframe with materials loading in kg/kWh for all the chemistries and for all years
    materials_rep = pd.concat([material]*(len(chemistries.index)-1))
    materials_rep.index = segments_chemistries_materials_index
    materials_rep_PHEV = materials_rep
    materials_rep.columns = [2015]
    
    #materials_rep_PHEV = materials_rep_PHEV.reindex(columns = batt_size_BEV.columns, method = 'ffill')
    materials_rep = materials_rep.reindex(columns = batt_size_BEV.columns, method = 'ffill')

    # * Create dataframes with capacity of battery in each segment and for each chemistry 
    # * Assumed the same battery capacity for each chemistry within the segment
    # * Create another dataframe with material content in battery within segment
    # * Material content in battery pack grows over time as battery capacity grows yearly
    capacity_segmented_BEV = batt_size_BEV.reindex(chem_index, level = 0)
    capacity_segmented_PHEV = batt_size_PHEV.reindex(chem_index, level = 0)
    material_content_BEV = batt_size_BEV.reindex(segments_chemistries_materials_index, level = 0).mul(materials_rep)
    material_content_PHEV = batt_size_PHEV.reindex(segments_chemistries_materials_index, level = 0).mul(materials_rep_PHEV)

########################## CAPACITY #######################################
    ##### Calculate yearly EV capacity additions and store it in a new set of dfs.
    BEV_capacity_additions_yearly_array =  BEV_split_chem_array
    PHEV_capacity_additions_yearly_array = PHEV_split_chem_array

    ## Actual calculation
    for i in range(len(BEV_capacity_additions_yearly_array)):
        BEV_capacity_additions_yearly_array[i] = capacity_segmented_BEV.values * BEV_capacity_additions_yearly_array[i]
        PHEV_capacity_additions_yearly_array[i] = capacity_segmented_PHEV.values * PHEV_capacity_additions_yearly_array[i]

        ## Rename index of dfs
        BEV_capacity_additions_yearly_array[i] = (
            BEV_capacity_additions_yearly_array[i]
            .reset_index()
            .rename(columns={'level_0': 'segment', 'level_1': 'chemistry' }) 
            .set_index(['segment','chemistry'])
        )

        PHEV_capacity_additions_yearly_array[i] = (
            PHEV_capacity_additions_yearly_array[i]
            .reset_index()
            .rename(columns={'level_0': 'segment', 'level_1': 'chemistry' })
            .set_index(['segment','chemistry'])
        )
        

########################## MATERIALS #######################################
    # * Calculate material additions in a similar fashion as the capacity additions
    BEV_material_additions_yearly_array =  BEV_split_chem_array
    PHEV_material_additions_yearly_array = PHEV_split_chem_array

    ## Actual calculation
    for i in range(len(BEV_material_additions_yearly_array)):
        BEV_material_additions_yearly_array[i] =  BEV_material_additions_yearly_array[i].reindex(segments_chemistries_materials_index)
        PHEV_material_additions_yearly_array[i] = PHEV_material_additions_yearly_array[i].reindex(segments_chemistries_materials_index)    

        BEV_material_additions_yearly_array[i] = material_content_BEV.values * BEV_material_additions_yearly_array[i]
        PHEV_material_additions_yearly_array[i] = material_content_PHEV.values * PHEV_material_additions_yearly_array[i]

        ## Rename index of dfs
        BEV_material_additions_yearly_array[i] = (
            BEV_material_additions_yearly_array[i]
            .reset_index()
            .rename(columns={'level_0': 'segment', 'level_1': 'chemistry', 'level_2': 'material' }) 
            .set_index(['segment','chemistry','material'])
        )

        PHEV_material_additions_yearly_array[i] = (
            PHEV_material_additions_yearly_array[i]
            .reset_index()
            .rename(columns={'level_0': 'segment', 'level_1': 'chemistry', 'level_2': 'material' }) 
            .set_index(['segment','chemistry','material'])
        )

#This section is over  with the calculation of the yearly material and capacity additions
#Next is the calculation of the outflows   

########################## Outflows #######################################
    mu = 11 #average lifetime of vehicles in EU
    sigma = 3 
    x = np.linspace(1, 25, 25)
    
    probability = pd.DataFrame(stats.norm.pdf(x, mu, sigma))
    probability = probability.reset_index()
    probability['index'] = probability['index'] + 1 
    probability = probability.set_index('index')
    probability.columns = ['2015']

    years_index = BEV_material_additions_yearly_array[0].columns


    #! Create empty dataframes. This is not the best way perhaps. 
    #empty_df = calculate_eol(chem_index, years_index, 2015, probability, BEV_capacity_additions_yearly_array[0])
    return BEV_split_chem_array[0]

#%%
    cap_eol_PHEV_base_SSP2 = empty_df
    cap_eol_PHEV_base_SSP1 = empty_df
    cap_eol_PHEV_RCP26_SSP2 = empty_df
    cap_eol_PHEV_RCP26_SSP1 = empty_df
    cap_eol_PHEV_base_LED = empty_df
    cap_eol_PHEV_RCP26_LED = empty_df

    cap_eol_BEV_base_SSP2 = empty_df
    cap_eol_BEV_base_SSP1 = empty_df
    cap_eol_BEV_RCP26_SSP2 = empty_df
    cap_eol_BEV_RCP26_SSP1 = empty_df
    cap_eol_BEV_base_LED = empty_df
    cap_eol_BEV_RCP26_LED = empty_df

    for col in cap_eol_PHEV_base_SSP2.columns:
        cap_eol_PHEV_base_SSP2[col].values[:] = 0
        cap_eol_PHEV_base_SSP1[col].values[:] = 0
        cap_eol_PHEV_RCP26_SSP2[col].values[:] = 0
        cap_eol_PHEV_RCP26_SSP1[col].values[:] = 0
        cap_eol_PHEV_base_LED[col].values[:] = 0
        cap_eol_PHEV_RCP26_LED[col].values[:] = 0

        cap_eol_BEV_base_SSP2[col].values[:] = 0
        cap_eol_BEV_base_SSP1[col].values[:] = 0
        cap_eol_BEV_RCP26_SSP2[col].values[:] = 0
        cap_eol_BEV_RCP26_SSP1[col].values[:] = 0
        cap_eol_BEV_base_LED[col].values[:] = 0
        cap_eol_BEV_RCP26_LED[col].values[:] = 0

    capacity_BEV_eol_array = [
        cap_eol_BEV_base_SSP2,
        cap_eol_BEV_base_SSP1,
        cap_eol_BEV_RCP26_SSP2,
        cap_eol_BEV_RCP26_SSP1,
        cap_eol_BEV_base_LED,
        cap_eol_BEV_RCP26_LED
        ]

    capacity_PHEV_eol_array = [
        cap_eol_PHEV_base_SSP2,
        cap_eol_PHEV_base_SSP1,
        cap_eol_PHEV_RCP26_SSP2,
        cap_eol_PHEV_RCP26_SSP1,
        cap_eol_PHEV_base_LED,
        cap_eol_PHEV_RCP26_LED
        ]

    total_eol_BEV_cap = [None]*len(capacity_PHEV_eol_array)
    total_eol_PHEV_cap = [None]*len(capacity_PHEV_eol_array)

    for n in range(len(years_index)):
        for i in range(len(capacity_BEV_eol_array)):
            capacity_BEV_eol_array[i] = calculate_eol( 
                chem_index,
                years_index,
                years_index[n], 
                probability,
                capacity_BEV_eol_array[i]
                )
                    
            capacity_PHEV_eol_array[i] = calculate_eol( 
                chem_index,
                years_index,
                years_index[n], 
                probability,
                capacity_PHEV_eol_array[i]
                )
            
            total_eol_BEV_cap = capacity_BEV_eol_array[i].add(capacity_BEV_eol_array[i], fill_value = 0)
            total_eol_PHEV_cap = capacity_PHEV_eol_array[i].add(capacity_PHEV_eol_array[i], fill_value = 0)
        





    return probability

# %%
data_read_manipulation()

# %%

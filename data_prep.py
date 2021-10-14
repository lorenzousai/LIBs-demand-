# %%
from itertools import groupby
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pickle
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline
import os


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
    
    stock_segmented = share_segments.dot(raw_vehicle_stock)
    return(stock_segmented)

def calculate_eol(index, years_list, start_year, prob_data, stock_add_df):
   
    
    ## Trying to calculate eol for all the years
    prob_data = prob_data.transpose()
    prob_data = pd.concat([prob_data]*len(stock_add_df.groupby(level=0)))
    
    prob_data = prob_data.reset_index()
    prob_data.drop('index', inplace = True, axis = 1)
    prob_data = prob_data.reindex(index, level = 0)
    prob_data.columns = pd.Index(np.arange(start_year, start_year+25))
   
    stock_add_rep = pd.concat([stock_add_df[start_year]]*25, axis = 1)
    
    stock_add_rep.columns = pd.Index(np.arange(start_year, start_year+25))
    
    eol = stock_add_rep.mul(prob_data)
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

    raw_data_inflows = pd.read_excel('EVs_inflows_ODYM_RECC.xls','Model_Results')
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
    BEVs_inflows_list = [BEV_inflows_RCP26_SSP2, BEV_inflows_RCP26_SSP1, BEV_inflows_RCP26_LED, 
        BEV_inflows_base_SSP2, BEV_inflows_base_SSP1, BEV_inflows_base_LED]

    # * PHEVs   
    PHEVs_inflows_list = [PHEV_inflows_RCP26_SSP2, PHEV_inflows_RCP26_SSP1, PHEV_inflows_RCP26_LED, 
        PHEV_inflows_base_SSP2, PHEV_inflows_base_SSP1, PHEV_inflows_base_LED]

    
    ### Read data for chemistries market share in given years
    chemistries = pd.read_excel('BEVs_LIB_CAPEX_employment stats.xlsx', sheet_name = 'Chemistries_scenario', skiprows=4, nrows = 8, usecols = 'B:AV')
    chemistries = chemistries.set_index(['chemistry'])
    chemistries = chemistries.interpolate(method = 'linear',  axis = 1)

    #* Export chemistries data
    with open('Dat_Figures//chemistries.pkl','wb') as f:
        pickle.dump(chemistries,f)


    ### Read average battery size in each segment and forecasts for future battery size for BEV
    # * Interpolate battery size for missing years
    batt_size_BEV = pd.read_excel('BEVs_LIB_CAPEX_employment stats.xlsx', sheet_name = 'Battery_size', skiprows=2, nrows = 7, usecols = 'C:AW')
    batt_size_BEV = batt_size_BEV.set_index("Segment")
    batt_size_BEV = batt_size_BEV.interpolate(method = "linear", axis = 1)
    batt_size_BEV = batt_size_BEV.round()
    batt_size_BEV = batt_size_BEV.reset_index()
    batt_size_BEV.drop('Segment', inplace = True, axis = 1)


    ## Do a similar thing for PHEVs, but in this case we assume the battery size remains constant over time 
    batt_size_PHEV = pd.read_excel('BEVs_LIB_CAPEX_employment stats.xlsx', sheet_name = 'Battery_size', skiprows=13, nrows = 7, usecols = 'C:D')
    batt_size_PHEV = batt_size_PHEV.set_index("Segment")
    batt_size_PHEV = batt_size_PHEV.reset_index()
    batt_size_PHEV.drop('Segment', inplace = True, axis = 1)
    batt_size_PHEV.columns = [2015]
    
    ### Read material loading in kg/kwh for each chemistry analysed and prepare df
    material = pd.read_excel('BEVs_LIB_CAPEX_employment stats.xlsx', sheet_name = 'Material composition', skiprows=1, nrows = 8, usecols = 'B:M')
    material = material.set_index(['chemistry'])
    

    ### Prepare multiindex to be used for each BEV and PHEV inflow df
    segments_index = stock_additions_segmented(share_df, BEVs_inflows_list[0])
    
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

    BEV_split_chem_list = [None]*len(BEVs_inflows_list)
    PHEV_split_chem_list = [None]*len(PHEVs_inflows_list)

    for i in range(len(BEVs_inflows_list)):
        BEVs_inflows_list[i] = stock_additions_segmented(share_df, BEVs_inflows_list[i])
        BEVs_inflows_list[i] = BEVs_inflows_list[i].multiply(1e6).round()
        BEV_split_chem_list[i] = BEVs_inflows_list[i].reindex(chem_index, level = 0)
        BEV_split_chem_list[i] = BEV_split_chem_list[i].multiply(chemistries, level = 1)

    for i in range(len(PHEVs_inflows_list)):
        PHEVs_inflows_list[i] = stock_additions_segmented(share_df, PHEVs_inflows_list[i])
        PHEVs_inflows_list[i] = PHEVs_inflows_list[i].multiply(1e6)
        PHEV_split_chem_list[i] = PHEVs_inflows_list[i].reindex(chem_index, level = 0)
        PHEV_split_chem_list[i] = PHEV_split_chem_list[i].multiply(chemistries, level = 1) 

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
    BEV_capacity_additions_yearly_list =  BEV_split_chem_list.copy()
    PHEV_capacity_additions_yearly_list = PHEV_split_chem_list.copy()

    ## Actual calculation
    for i in range(len(BEV_capacity_additions_yearly_list)):
        BEV_capacity_additions_yearly_list[i] = capacity_segmented_BEV.values * BEV_capacity_additions_yearly_list[i]
        PHEV_capacity_additions_yearly_list[i] = capacity_segmented_PHEV.values * PHEV_capacity_additions_yearly_list[i]

        ## Rename index of dfs
        BEV_capacity_additions_yearly_list[i] = (
            BEV_capacity_additions_yearly_list[i]
            .reset_index()
            .rename(columns={'level_0': 'segment', 'level_1': 'chemistry' }) 
            .set_index(['segment','chemistry'])
        )

        PHEV_capacity_additions_yearly_list[i] = (
            PHEV_capacity_additions_yearly_list[i]
            .reset_index()
            .rename(columns={'level_0': 'segment', 'level_1': 'chemistry' })
            .set_index(['segment','chemistry'])
        )
        

########################## MATERIALS #######################################
    # * Calculate material additions in a similar fashion as the capacity additions
    BEV_material_additions_yearly_list =  BEV_split_chem_list.copy()
    PHEV_material_additions_yearly_list = PHEV_split_chem_list.copy()

    ## Actual calculation
    for i in range(len(BEV_material_additions_yearly_list)):
        BEV_material_additions_yearly_list[i] =  BEV_material_additions_yearly_list[i].reindex(segments_chemistries_materials_index)
        PHEV_material_additions_yearly_list[i] = PHEV_material_additions_yearly_list[i].reindex(segments_chemistries_materials_index)    

        BEV_material_additions_yearly_list[i] = material_content_BEV.values * BEV_material_additions_yearly_list[i]
        PHEV_material_additions_yearly_list[i] = material_content_PHEV.values * PHEV_material_additions_yearly_list[i]

        ## Rename index of dfs
        BEV_material_additions_yearly_list[i] = (
            BEV_material_additions_yearly_list[i]
            .reset_index()
            .rename(columns={'level_0': 'segment', 'level_1': 'chemistry', 'level_2': 'material' }) 
            .set_index(['segment','chemistry','material'])
        )

        PHEV_material_additions_yearly_list[i] = (
            PHEV_material_additions_yearly_list[i]
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
    #probability = probability.set_index('index')
    probability.columns = ['',2015]
    probability = probability.set_index('')
    years_index = BEV_material_additions_yearly_list[0].columns
    

    #! Create empty dataframes. This is not the best way perhaps. 
    empty_df = calculate_eol(chem_index, years_index, 2015, probability, BEV_capacity_additions_yearly_list[0])


########################## Calculate retired capacity (in kWh) #######################################

    cap_eol_PHEV_base_SSP2 = empty_df.copy()
    cap_eol_PHEV_base_SSP1 = empty_df.copy()
    cap_eol_PHEV_RCP26_SSP2 = empty_df.copy()
    cap_eol_PHEV_RCP26_SSP1 = empty_df.copy()
    cap_eol_PHEV_base_LED = empty_df.copy()
    cap_eol_PHEV_RCP26_LED = empty_df.copy()

    cap_eol_BEV_base_SSP2 = empty_df.copy()
    cap_eol_BEV_base_SSP1 = empty_df.copy()
    cap_eol_BEV_RCP26_SSP2 = empty_df.copy()
    cap_eol_BEV_RCP26_SSP1 = empty_df.copy()
    cap_eol_BEV_base_LED = empty_df.copy()
    cap_eol_BEV_RCP26_LED = empty_df.copy()

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

    capacity_BEV_eol_list = [
        cap_eol_BEV_RCP26_SSP2,
        cap_eol_BEV_RCP26_SSP1,
        cap_eol_BEV_RCP26_LED,
        cap_eol_BEV_base_SSP2,
        cap_eol_BEV_base_SSP1,
        cap_eol_BEV_base_LED
        ]

    capacity_PHEV_eol_list = [
        cap_eol_PHEV_RCP26_SSP2,
        cap_eol_PHEV_RCP26_SSP1,
        cap_eol_PHEV_RCP26_LED,
        cap_eol_PHEV_base_SSP2,
        cap_eol_PHEV_base_SSP1,
        cap_eol_PHEV_base_LED,
        ]

    eol_BEV_int = [None]*len(capacity_PHEV_eol_list)
    eol_PHEV_int = [None]*len(capacity_PHEV_eol_list)

    for i in range(len(capacity_BEV_eol_list)):
        for n in range(len(years_index)):
            eol_BEV_int[i] =  calculate_eol(chem_index, years_index, years_index[n], probability, BEV_capacity_additions_yearly_list[i])
            capacity_BEV_eol_list[i] = capacity_BEV_eol_list[i].add(eol_BEV_int[i], fill_value = 0)
            eol_PHEV_int[i] = calculate_eol(chem_index, years_index, years_index[n], probability, PHEV_capacity_additions_yearly_list[i])
            capacity_PHEV_eol_list[i] = capacity_PHEV_eol_list[i].add(eol_PHEV_int[i], fill_value = 0) 
        

    total_capacity_addition = [None]*len(BEV_material_additions_yearly_list)
    total_capacity_outflows = [None]*len(BEV_material_additions_yearly_list)

    for i in range(len(BEV_capacity_additions_yearly_list)):
        total_capacity_addition[i] = (
                BEV_capacity_additions_yearly_list[i].copy() + 
                PHEV_capacity_additions_yearly_list[i].copy() 
            )

        total_capacity_outflows[i] = (
                capacity_BEV_eol_list[i].copy() + 
                capacity_PHEV_eol_list[i].copy() 
            )



########################## Calculate retired capacity (in kg) #######################################

    material_BEV_eol_list = capacity_BEV_eol_list.copy()
    material_PHEV_eol_list = capacity_PHEV_eol_list.copy()

    materials_rep_BEV_eol = materials_rep.reindex(columns = material_BEV_eol_list[0].columns,
        method = 'ffill')
    materials_rep_PHEV_eol = materials_rep_PHEV.reindex(columns = material_PHEV_eol_list[0].columns, 
        method = 'ffill')


    for i in range(len(material_BEV_eol_list)):
        material_BEV_eol_list[i] = material_BEV_eol_list[i].reindex(segments_chemistries_materials_index)
        material_PHEV_eol_list[i] = material_PHEV_eol_list[i].reindex(segments_chemistries_materials_index)    

        material_BEV_eol_list[i] = materials_rep_BEV_eol.values * material_BEV_eol_list[i]
        material_PHEV_eol_list[i] = materials_rep_PHEV_eol.values * material_PHEV_eol_list[i]

        material_BEV_eol_list[i] = (
                material_BEV_eol_list[i]
                .reset_index()
                .rename(columns={'level_0': 'segment', 'level_1': 'chemistry', 'level_2': 'material' }) 
                .set_index(['segment','chemistry','material'])
            )

        material_PHEV_eol_list[i] = (
                material_PHEV_eol_list[i]
                .reset_index()
                .rename(columns={'level_0': 'segment', 'level_1': 'chemistry', 'level_2': 'material' }) 
                .set_index(['segment','chemistry','material'])
            )



    ####################################### SECTION END ##############################################

    # * Clean up data by for materials inflows and outflows by merging Cu and Al flows 
    # * Originally, Cu and Al are split in Al and Cu at a cell level and Al and Cu at a pack level
    # * Now Al and Cu are merged
    # * In addition, the additions of BEV and PHEVs are joined in one unique list of dataframes. Same for the retired flows

    # * Dataframes to work on: 
    material_total_inflows = [None]*len(BEV_material_additions_yearly_list)
    material_total_outflows = [None]*len(BEV_material_additions_yearly_list)
    
    for i in range(len(BEV_material_additions_yearly_list)):
        
        material_total_inflows[i] = BEV_material_additions_yearly_list[i].groupby('material').sum().copy() + PHEV_material_additions_yearly_list[i].groupby('material').sum().copy()
        material_total_outflows[i] = material_BEV_eol_list[i].groupby('material').sum().copy() + material_PHEV_eol_list[i].groupby('material').sum().copy()
    
        material_total_inflows[i].loc['Cu'] = (material_total_inflows[i].loc['Cu']) + (material_total_inflows[i].loc['Cu_pack'])

        material_total_inflows[i].loc['Al'] = (material_total_inflows[i].loc['Al']) + (material_total_inflows[i].loc['Al_pack'])

        material_total_outflows[i].loc['Cu'] = (material_total_outflows[i].loc['Cu']) + (material_total_outflows[i].loc['Cu_pack'])
        
        material_total_outflows[i].loc['Al'] = (material_total_outflows[i].loc['Al']) + (material_total_outflows[i].loc['Al_pack'])

        # * Remove old data to avoid double counting
        material_total_inflows[i] = material_total_inflows[i].drop(['Al_pack','Cu_pack'], axis = 0)
        material_total_outflows[i] = material_total_outflows[i].drop(['Al_pack','Cu_pack'], axis = 0)

    ####################################### Calculate employment and CAPEX ##############################################

    # * Read data from excel file. 
    # * employment_gwh --> employees needed per GWh or production capacity installed. 
    # * Assumed to be 120 employees/Gwh

    # * Employment loss --> decrease in employment demand (as a % of the total demand) as a results of 
    # * optimization of production lines and automation

    # * Start with employment  
    employment_gwh = pd.read_excel('BEVs_LIB_CAPEX_employment stats.xlsx', sheet_name = 'Employment and automation', 
                        skiprows = 1, nrows = 1, usecols = 'B:AO')
    employment_loss = pd.read_excel('BEVs_LIB_CAPEX_employment stats.xlsx', sheet_name = 'Employment and automation', 
                        skiprows = 5, nrows = 1, usecols = 'B:AO')

    employment_loss = employment_loss.interpolate(method = 'linear', axis = 1)

    employment_generated_yearly_list = [None]*len(BEV_material_additions_yearly_list)
    cumulative_capacity = [None]*len(BEV_material_additions_yearly_list)

    for i in range(len(employment_generated_yearly_list)):

        employment_generated_yearly_list[i] = (
                total_capacity_addition[i]
                .copy()
                .groupby('chemistry').sum()
                .sum(axis = 0)
                .divide(1e6)
                .diff()
            )

        employment_with_loss = employment_loss.mul(employment_gwh)
        employment_generated_yearly_list[i] = employment_generated_yearly_list[i].mul(employment_with_loss)
        employment_generated_yearly_list[i] = employment_generated_yearly_list[i].loc[:,2020:2050]

    # * Now the CAPEX
        cumulative_capacity[i] = (
            total_capacity_addition[i]
            .copy()
            .groupby('chemistry')
            .sum()
            .sum(axis = 0)
            .cumsum()
            .copy()
            )

    CAPEX_rates = [None]*len(BEV_material_additions_yearly_list)
    CAPEX_scenarios_list = [None]*len(BEV_material_additions_yearly_list) 
    b = float(-0.32) #Learning rate
    A = 140/(cumulative_capacity[1][2016]**b) #Estimate initial price for product 

    for i in range(len(cumulative_capacity)):
        CAPEX_rates[i] = cumulative_capacity[i]*0

    for j in range(len(CAPEX_rates)):    
        for i in range(len(BEV_capacity_additions_yearly_list[0][1:45].columns)):
            index = i + 2016
            if index <= 2060:
                CAPEX_rates[j][index] = A * (cumulative_capacity[j][index]**b)
            else: 
                break
            
        CAPEX_scenarios_list[j] = CAPEX_rates[j].mul(total_capacity_addition[j].groupby('chemistry').sum().sum(axis = 0).divide(1e6).diff())

####################################### Prep dataq for historical sales figure ##############################################

    PCs_prod = pd.read_excel('BEVs_LIB_CAPEX_employment stats.xlsx', sheet_name = 'Other_industries_mat demand',skiprows = 3, nrows = 1, usecols = 'C:R')
    smartphones_prod = pd.read_excel('BEVs_LIB_CAPEX_employment stats.xlsx', sheet_name = 'Other_industries_mat demand',skiprows = 4, nrows = 1, usecols = 'J:R')
    solar_PV = pd.read_excel('BEVs_LIB_CAPEX_employment stats.xlsx', sheet_name = 'Other_industries_mat demand',skiprows = 5, nrows = 1, usecols = 'C:R')
    PbA_EU = pd.read_excel('BEVs_LIB_CAPEX_employment stats.xlsx', sheet_name = 'Other_industries_mat demand', skiprows = 6, nrows = 1, usecols = 'C:R')
    LIBs_CN = pd.read_excel('BEVs_LIB_CAPEX_employment stats.xlsx', sheet_name = 'Other_industries_mat demand',skiprows = 7, nrows = 1, usecols = 'C:R')

    PCs_prod.columns = range(2000,2016)
    solar_PV.columns = range(2000,2016)
    smartphones_prod.columns = range(2007,2016)
    PbA_EU.columns = range(2000,2016)
    LIBs_CN.columns = range(2000,2016)

    smartphones_prod_list = smartphones_prod.stack().droplevel(0)
    solar_PV_list = solar_PV.stack().droplevel(0)
    PCs_prod_list = PCs_prod.stack().droplevel(0)
    LIBs_CN = LIBs_CN.stack().droplevel(0)
    PbA_EU = PbA_EU.stack().droplevel(0)


    materials_addition_historical = [
        PCs_prod_list/1e9, 
        smartphones_prod_list/1e9, 
        solar_PV_list/1e9, 
        LIBs_CN/1e9,
        PbA_EU/1e9
        ]

    historical_cars_segments = pd.read_excel('BEVs_LIB_CAPEX_employment stats.xlsx', sheet_name = 'BEV_data', skiprows = 22, nrows = 7, usecols = 'C:H')
    historical_EVs_sales = pd.read_excel('BEVs_LIB_CAPEX_employment stats.xlsx', sheet_name = 'BEV_data', skiprows = 36, nrows = 2, usecols = 'C:H')

    BEV_sales_historical = pd.DataFrame(historical_EVs_sales.loc[0,:])
    PHEV_sales_historical = pd.DataFrame(historical_EVs_sales.loc[1,:])

    BEV_sales_historical = BEV_sales_historical.transpose()
    PHEV_sales_historical = PHEV_sales_historical.transpose()

    historical_cars_segments.columns = BEV_sales_historical.columns.values

    historical_sales_segmented = [BEV_sales_historical.round(), PHEV_sales_historical.round()]
    historical_segments = historical_cars_segments.round(decimals=3)

    for i in range(2):
        historical_sales_segmented[i] = pd.concat([historical_sales_segmented[i]]*len(historical_segments.index))
        historical_sales_segmented[i] = historical_sales_segmented[i].set_index(historical_segments.index)
        historical_sales_segmented[i] = historical_sales_segmented[i].mul(historical_segments)
        historical_sales_segmented[i] = historical_sales_segmented[i].round()


    chem_cut = chemistries.loc[:,chemistries.columns.isin(range(2015,2021))]
    chem_cut = chem_cut.round(decimals = 3)
    for i in range(len(historical_sales_segmented)):
        historical_sales_segmented[i] = historical_sales_segmented[i].reindex(chem_index, level = 0)
        historical_sales_segmented[i] = historical_sales_segmented[i].mul(chem_cut, level = 1)
        historical_sales_segmented[i] = historical_sales_segmented[i].round()

    historical_capacity = historical_sales_segmented.copy()
    materials_loading_historical = materials_rep.loc[:,materials_rep.columns.isin(range(2015,2021))]

    battery_size_BEV_hist = batt_size_BEV.loc[:, batt_size_BEV.columns.isin(range(2015,2021))]
    battery_size_PHEV_hist = pd.concat([batt_size_PHEV]*(len(historical_segments.columns)), axis = 1)
    battery_size_PHEV_hist.columns = [2015,2016,2017,2018,2019,2020]

    historical_capacity[0] = historical_sales_segmented[0].mul(battery_size_BEV_hist, level = 0)
    historical_capacity[1] = historical_sales_segmented[1].mul(battery_size_PHEV_hist, level = 0)

    historical_materials = historical_capacity.copy()
    for i in range(len(historical_materials)):
        historical_materials[i] = historical_materials[i].reindex(segments_chemistries_materials_index)
        historical_materials[i] = historical_materials[i].mul(materials_loading_historical, level = 2)

    all_materials_historical = historical_materials[0]+historical_materials[1]
    capacity_historical = historical_capacity[0]+historical_capacity[1] 

####################################### Export data ##############################################
#* Employment 
    with open('Dat_Figures//employment.pkl','wb') as f:
        pickle.dump(employment_generated_yearly_list,f)

#* CAPEX
    with open('Dat_Figures//CAPEX.pkl','wb') as f:
        pickle.dump(CAPEX_scenarios_list,f)

#* Material additions
    with open('Dat_Figures//material_additions.pkl','wb') as f:
        pickle.dump(material_total_inflows,f)

#* Material outflows
    with open('Dat_Figures//material_outflows.pkl','wb') as f:
        pickle.dump(material_total_outflows,f)

#* Capacity additions
    with open('Dat_Figures//capacity_additions.pkl','wb') as f:
        pickle.dump(total_capacity_addition,f)

#* Capacity outflows
    with open('Dat_Figures//capacity_outflows.pkl','wb') as f:
        pickle.dump(total_capacity_outflows,f)


#* Material content battery BEVs [kg/kWh]
    with open('Dat_Figures//material_content_BEV.pkl','wb') as f:
        pickle.dump(material_content_BEV,f)

#* Material content battery PHEVs [kg/kWh]
    with open('Dat_Figures//material_content_PHEV.pkl','wb') as f:
        pickle.dump(material_content_PHEV,f)

#* Historical materials inflows other industries
    with open('Dat_Figures//materials_inflows_industries.pkl','wb') as f:
        pickle.dump(materials_addition_historical,f)

#* Material additions - Historical 
    with open('Dat_Figures//material_additions_historical.pkl','wb') as f:
        pickle.dump(all_materials_historical,f)


#* Capacity additions - Historical 
    with open('Dat_Figures//capacity_additions_historical.pkl','wb') as f:
        pickle.dump(capacity_historical,f)


    return 

# %%
if __name__ == '__main__':
    data_read_manipulation()


# %%

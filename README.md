# LIBs-demand-
Code for manuscript published in Environmental Research: Infrastructure and Sustainability.
The code is split in 2 main parts: 
  The script "data_prep.py" is used to: 
      - Read the raw data on EVs inflows for each scenario
      - Split the inflows in segments (e.g. vehicle sizes), assign a certain battery chemistry and battery size, and calculate the material inflows
      - Calculate, through the use of survival rates, the outflows from the stock for batteries (in capacity [TWh/year]) and materials (Mton)
      - Calculate the CAPEX evolution as a function of the installed capacity in each scenario
      - Estimate the employment generated in Li-ion  production facilties in each scenario
      - Export the data as pickle files for further use
      
  The visualization notebook is used to plot the data calculated in the script above. 
      

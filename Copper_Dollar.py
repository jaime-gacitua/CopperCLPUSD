import pandas as pd
#import pandas.io.data as iod
from pandas_datareader import data
#Also import numpy and matplotlib. Because ....
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp


import time
import ctypes  # An included library with Python install.

import urllib.request as ul #url.request lib for handling the url
from bs4 import BeautifulSoup #bs for parsing the page

url_copper = 'http://www.tradeservice.com/copper_pricing/index.html'

## Didnt work
#data_frame = pd.read_html(url_copper,header=0)
#print(data_frame)


#Do stuff necessary to get the page text into a string
url_response=ul.urlopen(url_copper,timeout=5)

web_copper = BeautifulSoup(url_response) #Soup stores the data in a structured way to make retrieval easy
#Soup also automatically decodes the page correctly (most of the time!)

#print(web_copper.prettify()) #Prints page contents 


## Import data
tables_copper = web_copper.find_all('table')
a = []
data = []
holiday_flag = False
for table in tables_copper:
    if(table.get('bgcolor') == '#ffffff'):
        for row in table.find_all('tr'):
            a = []
            holiday_flag = False
            for col in row.find_all('td'):
                if col.get_text() == 'HOLIDAY' or col.get_text() == 'None' :
                    # print(col.get_text())
                    holiday_flag = True
                    
                aux_txt = col.get_text().replace(" ", "") 
                aux_txt = aux_txt.replace("+", "")
                aux_txt = aux_txt.replace("$", "")
                a.append(aux_txt)
            if len(a)>0 and not holiday_flag:
                data.append(a)


# Column Names
data_pd = pd.DataFrame(data, columns=["Day", "Date", "Spot $/Pound", "Spot Change", "LMECopper $/Cathode", "FutureCOMEX Price", "Future Month"])

# Transform columns into Float numbers
for col in data_pd.columns.values:
    try:
        data_pd[col] = data_pd[col].astype(float)  
    except:
        pass


# Transform dates into proper format
j = 0
for date in data_pd["Date"]:
    try:
        date1 = time.strptime(str(date), "%b%d" )
#        print(time.strftime('%m/%d', date1) + "/2015")
        data_pd["Date"][j] = time.strftime('%m/%d', date1) + "/2015"
    except:
        pass
    j = j+1
    
print(data_pd)
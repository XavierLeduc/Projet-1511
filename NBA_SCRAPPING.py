from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Parameters
init_link = "https://www.nba.com/stats/leaders?Season=2022-23"


#CSV config

#Start of the program
driver = webdriver.Chrome()
driver.get(init_link)

time.sleep(5)
cookie_accept = driver.find_element(By.CSS_SELECTOR, "#onetrust-accept-btn-handler")
cookie_accept.click()
time.sleep(10)


soup = BeautifulSoup(driver.content, 'html.parser')

table = soup.find('table', attrs={'class':'Crom_table__p1iZz'})

if table:
    df = pd.read_html(str(table))[0]

    df.to_csv('NBA_Stats.csv', index=False)
else:
    print("No table found")

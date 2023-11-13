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
import io


#Parameters
init_link = "https://www.nba.com/stats/leaders?Season=2022-23"


#CSV config

#Start of the program
def get_data(link):
    driver = webdriver.Chrome()
    driver.get(init_link)

    time.sleep(1)
    cookie_accept = driver.find_element(By.CSS_SELECTOR, "#onetrust-accept-btn-handler")
    cookie_accept.click()
    time.sleep(1)


    page_source = driver.page_source

    soup = BeautifulSoup(page_source, 'html.parser')

    table = soup.find('table', attrs={'class':'Crom_table__p1iZz'})

    if table:
        df = pd.read_html(io.StringIO(str(table)))[0]

        df.to_csv('NBA_Stats_22-23.csv', index=False)

        print("Données sauvegardées.")
    else:
        print("Erreur lors de la récupération des données.")


    driver.quit()

get_data(init_link)
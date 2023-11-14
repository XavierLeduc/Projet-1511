from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import io



#Parameters
seasons = ["2022-23", "2021-22", "2020-21", "2019-20", "2018-19", "2017-18", "2016-17", "2015-16", "2014-15", "2013-14", "2012-13"]

#Start of the program
def get_data_per_year_per_player():
    i = 0
    for year in seasons:
        link = "https://www.nba.com/stats/leaders?Season={}".format(seasons[i])
        driver = webdriver.Chrome()
        driver.get(link)

        cookie_accept = driver.find_element(By.CSS_SELECTOR, "#onetrust-accept-btn-handler")
        cookie_accept.click()
        time.sleep(10)

        dropdown_div = driver.find_element(By.CSS_SELECTOR, ".Pagination_pageDropdown__KgjBU")
        dropdown_element = Select(dropdown_div.find_element(By.CSS_SELECTOR, ".DropDown_select__4pIg9"))
        dropdown_element.select_by_visible_text("All")

        page_source = driver.page_source

        soup = BeautifulSoup(page_source, 'html.parser')
        

        table = soup.find('table', attrs={'class':'Crom_table__p1iZz'})



        if table:
            df = pd.read_html(io.StringIO(str(table)))[0]

            df.to_csv('NBA_Stats_{}.csv'.format(seasons[i]), index=False)


            print("Données sauvegardées.")
        else:
            print("Erreur lors de la récupération des données.")
        
        time.sleep(10)
        i += 1




def assemble_all_seasons():
    i = 0
    for year in seasons:
        df = pd.read_csv('NBA_Stats_{}.csv'.format(seasons[i]))
        df['SEASON'] = seasons[i]
        if i == 0:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])
        i += 1
    #fais un to_csv en supprimant la colonne "#"
    df_all = df_all.drop(columns=['#'])
    df_all.to_csv('NBA_Stats_All_Seasons.csv', index=False)

#assemble_all_seasons()

#df = get_data(init_link)

df = pd.read_csv('NBA_Stats_22-23.csv')


print(df.describe())


#plt.figure(figsize=(20,10))
#plt.hist(df['PTS'], bins=70)
#plt.title("Distribution des points marqués par match")
#plt.xlabel("Points marqués")
#plt.ylabel("Nombre de joueurs")
#plt.show()


#sns.boxenplot(x="TEAM", y="PTS", data=df)
#plt.title("Distribution des points marqués par équipe")
#plt.xlabel("Equipe")
#plt.ylabel("Points marqués")
#plt.show()


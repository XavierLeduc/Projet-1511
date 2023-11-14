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
init_link = "https://www.nba.com/stats/leaders?Season=2022-23"



#Start of the program
def get_data(link):
    driver = webdriver.Chrome()
    driver.get(init_link)

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

        df.to_csv('NBA_Stats_22-23.csv', index=False)


        print("Données sauvegardées.")
    else:
        print("Erreur lors de la récupération des données.")
    

    driver.quit()

    return df





#df = get_data(init_link)

df = pd.read_csv('NBA_Stats_22-23.csv')


print(df.describe())


plt.figure(figsize=(20,10))
plt.hist(df['PTS'], bins=70)
plt.title("Distribution des points marqués par match")
plt.xlabel("Points marqués")
plt.ylabel("Nombre de joueurs")
plt.show()


sns.boxenplot(x="TEAM", y="PTS", data=df)
plt.title("Distribution des points marqués par équipe")
plt.xlabel("Equipe")
plt.ylabel("Points marqués")
plt.show()


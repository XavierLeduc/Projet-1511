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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score





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




#print(df.describe())



def pearson_correlation():
    data_to_analyse = ['FG%', '3P%', 'FT%']
    df = pd.read_csv('NBA_Stats_All_Seasons.csv')
    for data in data_to_analyse:
        pearson_correlation = np.corrcoef(df['PTS'], df[data])[0, 1]
        print("Corrélation entre PTS et {} : {}".format(data, pearson_correlation))
        correlation_matrix = np.corrcoef(df['PTS'], df[data])
        plt.figure(figsize=(10,10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.show()


#pearson_correlation()


def analysis_shooting_percentages():
    df = pd.read_csv('NBA_Stats_All_Seasons.csv')
    features = ['PTS', 'FGM', 'FGA', '3PM', '3PA', 'FTM', 'FTA', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV']
    target = ['EFF']

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    #r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    #print(f'R-squared: {r2}')


    plt.scatter(y_test, y_pred)
    plt.xlabel("Vraies valeurs")
    plt.ylabel("Prédictions")
    plt.title("Régression linéaire pour les tirs tentés")
    plt.show()

analysis_shooting_percentages()








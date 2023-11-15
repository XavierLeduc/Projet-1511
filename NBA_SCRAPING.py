#Xavier LEDUC @2023

##### Analysis of NBA players over the last 10 years.
##### It includes a linear regression model and classifications of players according to their actions in the game.

import os
os.system('cls' if os.name == 'nt' else 'clear')



#SCRAPING
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup



#DATA
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import io



#SKLEARN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report





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


def get_advanced_data():
    i = 0
    for year in seasons:
        link = "https://www.nba.com/stats/players/traditional?PerMode=Totals&sort=PTS&dir=-1&Season={}".format(seasons[i])
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

            df.to_csv('NBA_Stats_Advanced_{}.csv'.format(seasons[i]), index=False)


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

def assemble_all_seasons_advanced():
    i = 0
    for year in seasons:
        df = pd.read_csv('NBA_Stats_Advanced_{}.csv'.format(seasons[i]))
        df['SEASON'] = seasons[i]
        if i == 0:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])
        i += 1
    df_all.to_csv('NBA_Stats_Advanced_All_Seasons.csv', index=False)

assemble_all_seasons_advanced()




### Classifications des joueurs ###
def classification_joueurs(df):
    print(df['PTS'].describe())
    df['Level'] = pd.cut(df['PTS'], bins=[46, 208, 488, float('inf')], labels=['Faible', 'Moyen', 'Élevé'])

    features = df[['FGM', '3PM', 'FTM', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV', 'PF']]
    target = df['Level']

    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(target)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.3, random_state=42)

    n_neighbors = 3
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)

    knn_model.fit(X_train, y_train)

    predictions = knn_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    classification_report_output = classification_report(y_test, predictions)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report_output)


    selected_columns = ['Player', 'PTS', 'Level']
    df_selected = df[selected_columns]
    df_selected.to_csv('Classification_Players.csv', index=False)

    print("Fichier CSV enregistré avec succès.")







#### Modèle de prédiction ####




def data_cleaning(df):
    #df = df.iloc[:, :-26]
    df = df.dropna()
    return df


def pearson_correlation(df):
    df = data_cleaning(df)

    data_to_analyse = ['OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
    for data in data_to_analyse:
        pearson_correlation = np.corrcoef(df['PTS'], df[data])[0, 1]
        print("Corrélation entre PTS et {} : {}".format(data, pearson_correlation))



def analysis_shooting_percentages(df):
    df = data_cleaning(df)
    features = ['OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
    target = ['PTS']

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=42)

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')


    plt.scatter(y_test, y_pred)
    plt.xlabel("Vraies valeurs")
    plt.ylabel("Prédictions")
    plt.title("Régression linéaire simple par rapport aux points marqués selon l'attaque.")
    plt.show()





def assembly_by_player(df):
    df = df.drop(['Unnamed: 0', 'Team', 'GP RANK', 'W RANK', 'L RANK', 'MIN RANK', 'PTS RANK', 'FGM RANK', 'FGA RANK',
                'FG% RANK', '3PM RANK', '3PA RANK', '3P% RANK', 'FTM RANK', 'FTA RANK', 'FT% RANK', 'OREB RANK', 'DREB RANK',
                'REB RANK', 'AST RANK', 'TOV RANK', 'STL RANK', 'BLK RANK', 'PF RANK', 'FP RANK', 'DD2 RANK', 'TD3 RANK', '+/- RANK', 'SEASON'],
                axis=1)

    df_avg = df.groupby('Player').mean().reset_index()

    df_avg.to_csv('NBA_Stats_Advanced_Group_By_Player_All_Season.csv', index=False)





df = pd.read_csv('NBA_Stats_Advanced_Group_By_Player_All_Season.csv')
classification_joueurs(df)
#pearson_correlation(df)
#analysis_shooting_percentages(df)
#assembly_by_player(df)
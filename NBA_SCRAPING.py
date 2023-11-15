import os
os.system('cls' if os.name == 'nt' else 'clear')


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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    if len(df) == 0:
        print("Error: No samples in the data.")
        return

    features = df[['Age', 'GP', 'W', 'L', 'Min', 'PTS', 'FG%', '3P%', 'FT%', 'REB', 'AST', 'STL', 'BLK']]
    target = df['Player']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn_model = KNeighborsClassifier(n_neighbors=3)

    knn_model.fit(X_train, y_train)
    predictions = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=1)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)
    

    report_dict = classification_report(y_test, predictions, output_dict=True)
    df_report = pd.DataFrame(report_dict).T


    seuil_fort = 0.8
    seuil_faible = 0.5


    df_report['Groupe'] = pd.cut(df_report['precision'], bins=[-1, seuil_faible, seuil_fort, 1], labels=['Faible', 'Moyen', 'Fort'])


    df_report['Groupe'] = pd.Categorical(df_report['Groupe'])


    colors = df_report['Groupe'].cat.codes


    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_report.index, df_report['precision'], c=colors, cmap='viridis', s=100)

    plt.legend(scatter.legend_elements(), title='Groupes')


    plt.xlabel('Classes')
    plt.ylabel('Précision')
    plt.title('Précision par classe avec groupes')

    plt.show()




#### Modèle de prédiction ####




def data_cleaning(df):
    #surpprime les 26 dernières colonnes
    df = df.iloc[:, :-26]
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



df = pd.read_csv('NBA_Stats_Advanced_All_Seasons.csv')
#classification_joueurs(df)
pearson_correlation(df)
analysis_shooting_percentages(df)












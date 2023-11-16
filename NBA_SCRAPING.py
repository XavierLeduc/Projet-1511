# Théo Porzio, Xavier Leduc, Eliot Leleu, Alexandre Deprez

##### Analysis of NBA datas over the last 10 years.
##### It includes linear regression models, classifications of players according to their actions in the game.

import os

from sklearn.tree import DecisionTreeRegressor

os.system('cls' if os.name == 'nt' else 'clear')

# SCRAPING
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup

# DATA
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# SKLEARN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def menu():
    while True:
        print("Menu")
        print("0) Scraping et formatage.")
        print("1) Classification des joueurs.")
        print("2) Modèle de prédiction.")
        print("3) Corrélation de Paerson.")
        print("4) Caractéristiques des données.")
        print("5) Graphiques.")
        print("6) Statistiques de tous les matchs.")
        print("7) Modèle de prédiction 2 .")
        print("8) Quitter le programme.")
        time.sleep(1)

        try:
            choice = input("Choisissez l'option:")
        except EOFError:
            print("\nError reading input. Please make sure you are running the script in a compatible environment.")
            break

        choice = choice.strip()

        if (choice == "0"):
            get_data_per_year_per_player()
            get_advanced_data()
            get_data_per_year_per_match()
            assemble_all_seasons()
            assemble_all_seasons_advanced()
            assemble_all_matchs()
            assemble_by_player(df)
        elif (choice == "1"):
            classification_joueurs(df)
        elif (choice == "2"):
            analysis_shooting_percentages(df)
        elif (choice == "3"):
            pearson_correlation(df)
        elif (choice == "4"):
            print(df.describe())
        elif (choice == "5"):
            graphiques(df)
        elif (choice == "6"):
            match_stats()
        elif (choice == "7"):
            glFromTurnover()
        elif (choice == "8"):
            print("Au revoir.")
            break


# Parameters
seasons = ["2022-23", "2021-22", "2020-21", "2019-20", "2018-19", "2017-18", "2016-17", "2015-16", "2014-15", "2013-14",
           "2012-13"]


# Start of the program
def get_data_per_year_per_player():
    i = 0
    for year in seasons:
        link = "https://www.nba.com/stats/leaders?Season={}".format(seasons[i])
        driver = webdriver.Chrome()
        driver.get(link)

        cookie_accept = driver.find_element(By.CSS_SELECTOR, "#onetrust-accept-btn-handler")
        if cookie_accept:
            time.sleep(1)
            cookie_accept.click()
            time.sleep(6)

        dropdown_div = driver.find_element(By.CSS_SELECTOR, ".Pagination_pageDropdown__KgjBU")
        dropdown_element = Select(dropdown_div.find_element(By.CSS_SELECTOR, ".DropDown_select__4pIg9"))
        dropdown_element.select_by_visible_text("All")

        page_source = driver.page_source

        soup = BeautifulSoup(page_source, 'html.parser')

        table = soup.find('table', attrs={'class': 'Crom_table__p1iZz'})

        if table:
            df = pd.read_html(io.StringIO(str(table)))[0]

            df.to_csv('NBA_Stats_{}.csv'.format(seasons[i]), index=False)

            print("Données sauvegardées.")
        else:
            print("Erreur lors de la récupération des données.")

        time.sleep(8)
        i += 1


def get_advanced_data():
    i = 0
    for year in seasons:
        link = "https://www.nba.com/stats/players/traditional?PerMode=Totals&sort=PTS&dir=-1&Season={}".format(
            seasons[i])
        driver = webdriver.Chrome()
        driver.get(link)

        cookie_accept = driver.find_element(By.CSS_SELECTOR, "#onetrust-accept-btn-handler")
        if cookie_accept:
            time.sleep(1)
            cookie_accept.click()
            time.sleep(6)

        dropdown_div = driver.find_element(By.CSS_SELECTOR, ".Pagination_pageDropdown__KgjBU")
        dropdown_element = Select(dropdown_div.find_element(By.CSS_SELECTOR, ".DropDown_select__4pIg9"))
        dropdown_element.select_by_visible_text("All")

        page_source = driver.page_source

        soup = BeautifulSoup(page_source, 'html.parser')

        table = soup.find('table', attrs={'class': 'Crom_table__p1iZz'})

        if table:
            df = pd.read_html(io.StringIO(str(table)))[0]

            df.to_csv('NBA_Stats_Advanced_{}.csv'.format(seasons[i]), index=False)

            print("Données sauvegardées.")
        else:
            print("Erreur lors de la récupération des données.")

        time.sleep(8)
        i += 1


def get_data_per_year_per_match():
    driver = webdriver.Chrome()
    driver.get("https://www.nba.com/stats/teams/boxscores")

    cookie_accept = driver.find_element(By.CSS_SELECTOR, "#onetrust-accept-btn-handler")
    if cookie_accept:
        time.sleep(1)
        cookie_accept.click()
        time.sleep(6)

    for season in seasons:
        dropdown_div = driver.find_element(By.CSS_SELECTOR, ".DropDown_content__Bsm3h")
        dropdown_element = Select(dropdown_div.find_element(By.CSS_SELECTOR, ".DropDown_select__4pIg9"))
        dropdown_element.select_by_visible_text(season)

        time.sleep(2)

        dropdown_div = driver.find_element(By.CSS_SELECTOR, ".Pagination_pageDropdown__KgjBU")
        dropdown_element = Select(dropdown_div.find_element(By.CSS_SELECTOR, ".DropDown_select__4pIg9"))
        dropdown_element.select_by_visible_text("All")

        page_source = driver.page_source

        soup = BeautifulSoup(page_source, 'html.parser')

        table = soup.find('table', attrs={'class': 'Crom_table__p1iZz'})

        if table:
            df = pd.read_html(io.StringIO(str(table)))[0]

            df.to_csv('NBA_Matchs_Stats_{}.csv'.format(season), index=False)

            print("Données sauvegardées.")
        else:
            print("Erreur lors de la récupération des données.")

    driver.quit()

    return df


def assemble_all_matchs():
    i = 0
    for year in seasons:
        df = pd.read_csv('NBA_Matchs_Stats_{}.csv'.format(seasons[i]))
        df['SEASON'] = seasons[i]
        if i == 0:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])
        i += 1
    df_all.to_csv('NBA_Matchs_Stats_All_Seasons.csv', index=False)


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
    # Création d'un csv en supprimant la colonne "#"
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


def assemble_by_player(df):
    columns_to_drop = [
        'Unnamed: 0', 'Team', 'GP RANK', 'W RANK', 'L RANK', 'MIN RANK', 'PTS RANK', 'FGM RANK', 'FGA RANK',
        'FG% RANK', '3PM RANK', '3PA RANK', '3P% RANK', 'FTM RANK', 'FTA RANK', 'FT% RANK', 'OREB RANK',
        'DREB RANK', 'REB RANK', 'AST RANK', 'TOV RANK', 'STL RANK', 'BLK RANK', 'PF RANK', 'FP RANK',
        'DD2 RANK', 'TD3 RANK', '+/- RANK', 'SEASON'
    ]

    df = df.drop(columns=columns_to_drop, errors='ignore')

    df_avg = df.groupby('Player').mean().reset_index()

    df_avg.to_csv('NBA_Stats_Advanced_Group_By_Player_All_Season.csv', index=False)


### Graphiques ###
def graphiques(df):
    df_classification = pd.read_csv('Classification_Players.csv')

    # Crée un diagramme à barres montrant la distribution des joueurs en fonction de leur niveau
    sns.countplot(x='Level', data=df_classification)
    plt.title("Répartition des joueurs selon leur niveau.")
    plt.show()

    # Crée un diagramme en boîte montrant la distribution des points en fonction du niveau des joueurs
    sns.boxplot(x='Level', y='PTS', data=df_classification)
    plt.show()

    pearson_correlation(df)


### Graphiques Matchs ###
def match_stats():
    df = pd.read_csv('NBA_Matchs_Stats_All_Seasons.csv')

    # Diagramme de dispersion entre les points marqués et les rebonds
    sns.scatterplot(x='PTS', y='REB', data=df)
    plt.title('Scatter Plot - Points vs Rebonds')
    plt.show()

    # Régression linéaire entre les points marqués et les tirs tentés
    sns.regplot(x='FGA', y='PTS', data=df)
    plt.title('Régression Linéaire - Points vs Tirs Tentés')
    plt.show()

    # Régression arborescente entre les points marqués et les rebonds
    X_train, X_test, y_train, y_test = train_test_split(df[['REB']], df['PTS'], test_size=0.2, random_state=42)
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(X_train, y_train)
    predictions = tree_reg.predict(X_test)

    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, predictions, color='blue', linewidth=3)
    plt.title('Régression Arborescente - Points vs Rebonds')
    plt.xlabel('Rebonds')
    plt.ylabel('Points')
    plt.show()

    # Boxplot des points marqués par équipe
    sns.boxplot(x='Team', y='PTS', data=df)
    plt.title('Boxplot - Points par Équipe')
    plt.xticks(rotation=45)
    plt.show()

    # Boxplot des points marqués par résultat du match
    sns.boxplot(x='W/L', y='PTS', data=df)
    plt.title('Boxplot - Points par Résultat du Match')
    plt.show()

    # Histogramme des points marqués
    sns.histplot(df['PTS'], bins=20, kde=True)
    plt.title('Histogramme des Points Marqués')
    plt.show()

    # Diagramme en barres du pourcentage de victoires par équipe
    team_win_percentage = df.groupby('Team')['W/L'].value_counts(normalize=True).unstack()
    team_win_percentage.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Pourcentage de Victoires par Équipe')
    plt.xlabel('Équipe')
    plt.ylabel('Pourcentage de Victoires')
    plt.legend(title='Résultat du Match', bbox_to_anchor=(1, 1))
    plt.show()

    # Diagramme en barres des moyennes de points marqués par équipe
    team_avg_points = df.groupby('Team')['PTS'].mean().sort_values(ascending=False)
    team_avg_points.plot(kind='bar', color='skyblue')
    plt.title('Moyenne de Points Marqués par Équipe')
    plt.xlabel('Équipe')
    plt.ylabel('Moyenne de Points Marqués')
    plt.show()

    # Heatmap de la corrélation entre les différentes statistiques donnée par le tableau (sur le site)
    numeric_columns = df.select_dtypes(include=['number'])

    correlation_matrix = numeric_columns.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Heatmap de Corrélation entre les Statistiques')
    plt.show()

    # Diagramme en barres des moyennes de points marqués par date de match
    df['Game Date'] = pd.to_datetime(df['Game Date'], format='%m/%d/%Y')
    df['Month'] = df['Game Date'].dt.month
    monthly_avg_points = df.groupby('Month')['PTS'].mean()
    monthly_avg_points.plot(kind='bar', color='salmon')
    plt.title('Moyenne de Points Marqués par Mois')
    plt.xlabel('Mois')
    plt.ylabel('Moyenne de Points Marqués')
    plt.show()


def glFromTurnover():
    df = pd.read_csv('NBA_Matchs_Stats_All_Seasons.csv')

    # Ici, on crée une nouvelle colonne qui contient 1 si l'équipe a perdu et 0 si elle a gagné
    df['Percentage_Loss'] = df['W/L'].apply(lambda x: 1 if x == 'L' else 0)

    X = df[['TOV']]
    y = df['Percentage_Loss']

    # On sépare les données en données d'entraînement et données de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # On crée le modèle
    model = LinearRegression()
    model.fit(X_train, y_train)

    # On fait des prédictions sur les données de test
    y_pred = model.predict(X_test)

    # On évalue le modele
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')


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
    # df = df.iloc[:, :-26]
    df = df.dropna()
    return df


def pearson_correlation(df):
    df = data_cleaning(df)

    data_to_analyse = ['OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
    for data in data_to_analyse:
        pearson_correlation = np.corrcoef(df['PTS'], df[data])[0, 1]
        print("Corrélation entre PTS et {} : {}".format(data, pearson_correlation))

        sns.heatmap(df[['PTS', data]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Heatmap de la corrélation entre PTS et {}".format(data))
        plt.show()


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

    plt.scatter(y_test, y_pred, s=1)
    plt.xlabel("Vraies valeurs")
    plt.ylabel("Prédictions")
    plt.title("Régression linéaire des points selon le type d'attaque.")
    plt.show()


df = pd.read_csv('NBA_Stats_Advanced_Group_By_Player_All_Season.csv')

menu()

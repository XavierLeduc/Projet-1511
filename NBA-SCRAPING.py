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
waiting = time.sleep(1)

#CSV config

#Start of the program
driver = webdriver.Chrome()
driver.get(init_link)

waiting
cookie_accept = driver.find_element(By.CSS_SELECTOR, "#onetrust-accept-btn-handler")
cookie_accept.click()
waiting

import requests
import bs4
import util
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
import re
import time
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

NUM_REV = 10

def get_soup(url):
    '''
    Returns soup object, given URL
    Inputs:
        url: url to request
    Output:
        soup: Beautiful Soup Object
        page_url: url from the request object
    '''
    #request_obj = util.get_request(url)
    # request_obj = requests.get(url, timeout=15)
    # soup = bs4.BeautifulSoup(request_obj.text, 'html.parser')
    
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--incognito')
    options.add_argument('--headless')
    driver = webdriver.Chrome(executable_path = r'C:\Users\Dhruval\Downloads\chromedriver_win32\chromedriver.exe', chrome_options=options)
    driver.get(url)
    
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    return soup


def parse_reviews(soup_obj):
    '''
    '''

    df = pd.DataFrame()
    df_title = pd.DataFrame()
    df_text = pd.DataFrame()
    df_rating = pd.DataFrame()
    
    # Whole Div
    # soup_obj.find('div', class_="hotels-community-tab-common-Card__ui_card--mBW-w hotels-community-tab-common-Card__card--ihfZB hotels-community-tab-common-Card__section--4r93H")

    # Find Name and Data from Text
    # soup_obj.find('div', class_="social-member-event-MemberEventOnObjectBlock__event_type--3njyv").text

    #title
    k = 0
    for item in soup_obj.find_all('a', class_="hotels-review-list-parts-ReviewTitle__reviewTitleText--3QrTy"):
        df_title = df_title.append(pd.DataFrame({'key': k, 'Title': item.text}, index=[0]), ignore_index=True)
        k = k + 1

    #Text
    k = 0
    for item in soup_obj.find_all('q', class_="hotels-review-list-parts-ExpandableReview__reviewText--3oMkH"):
        df_text = df_text.append(pd.DataFrame({'key': k, 'Text': item.text}, index=[0]), ignore_index=True)
        k = k + 1

    # Rating
    k = 0
    for item in soup_obj.find_all('span', class_="ui_bubble_rating"):
        bubble = item.attrs['class'][1]
        print(bubble)
        rating = re.search(r'\d', bubble)[0]
        print(rating)
        df_rating = df_rating.append(pd.DataFrame({'key': k, 'Rating': rating}, index=[0]), ignore_index=True)
        k = k + 1
    
    #print(df_title)

    df = pd.merge(df_title, df_rating, on='key')
    # df = pd.merge(df_title, df_text, on='key')
    # df = pd.merge(df, df_rating, on='key')
    print(df)
    return df



def parse(url):
    '''
    Parse for the main page and all pages of reviews
    '''

    df = parse_reviews(get_soup(url))

    url_update = url.replace('-Reviews-', '-Reviews-or{}-')

    # Loop through all other review pages
    for offset in range(5, NUM_REV, 5):
        print('offset', offset)
        url_ = url_update.format(offset)
        df2 = parse_reviews(get_soup(url_))
        df = pd.concat([df, df2], ignore_index=True)

    df = df.drop(columns =['key'])
    
    print(df)
    #df.to_csv('hostel_reviews.csv')


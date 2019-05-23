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

NUM_REV = 607

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
    # driver = webdriver.PhantomJS(executable_path = r'C:\Users\Dhruval\Downloads\phantomjs-2.1.1-windows\phantomjs-2.1.1-windows\bin\phantomjs')
    driver.get(url)

    #Code to click on traveltype checkbox
    # wait = WebDriverWait(driver, 10)
    # element = wait.until(EC.element_to_be_clickable((By.ID, 'TravelTimeFilter_0')))

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    return soup


def parse_reviews(soup_obj):
    '''
    '''
    if soup_obj == None:
        print("Obj is empty")
    else:
        print("there is object")

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
    # k = 0
    # for item in soup_obj.find_all('span', class_="ui_bubble_rating"):
    #     bubble = item.attrs['class'][1]
    #     rating = re.search(r'\d', bubble)[0]
    #     df_rating = df_rating.append(pd.DataFrame({'key': k, 'Rating': rating}, index=[0]), ignore_index=True)
    #     k = k + 1
    
    print("title", df_title)
    print("text", df_text)
    if df_title.empty and df_text.empty:
        e_df = pd.DataFrame
        return e_df
    else:
        #df = pd.merge(df_title, df_rating, on='key')
        df = pd.merge(df_title, df_text, on='key')
        # df = pd.merge(df, df_rating, on='key')
        #print(df)
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
        if df2.empty:
            continue
        else:
            df = pd.concat([df, df2], ignore_index=True)

    df = df.drop(columns =['key'])
    
    #print(df)
    df.to_csv('hotel_all_reviews.csv')


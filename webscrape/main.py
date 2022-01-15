#!/usr/bin/env python

from selenium import webdriver
from time import sleep
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException

def main():

    start = 0
    end = 100 # Not inclusive

    path = f'output{7}-{10}.txt'
    file = open(path, "w", encoding='utf-8')

    url = "https://www.imdb.com/search/title/?user_rating=7.1,10.0"

    #driver = webdriver.Chrome('/home/marno874/web-scrape/chromedriver')
    driver = webdriver.Chrome(r'C:\Users\Marcus\webscrape\chromedriver.exe')
    
    driver.get(url)
    (driver.page_source).encode('utf-8')

    for i in range(start, end):

        print("i = ", i)
       
        movie_list = driver.find_elements(By.XPATH, "//div[contains(@class,'lister-list')]//div[contains(@class,'lister-item')]//div[contains(@class,'lister-item-content')]") 
        
        movie = movie_list[i]

        wait = WebDriverWait(movie, 10)

        movie_link =  wait.until(EC.element_to_be_clickable((By.XPATH, "./h3[contains(@class,'lister-item-header')]//a")))
        movie_link.click()

        wait = WebDriverWait(driver, 10)
        user_reviews = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[contains(@class,'UserReviewsHeader__Header-k61aee-0')]//a")))
        user_reviews.click()

        try:
            while True:
                wait = WebDriverWait(driver, 10)
                elem = wait.until(EC.element_to_be_clickable((By.ID, 'load-more-trigger')))
                elem.click()

        except TimeoutException:
            reviews = driver.find_elements(By.CLASS_NAME, 'lister-item-content')

        #reviews = driver.find_elements(By.CLASS_NAME, 'lister-item-content')

        for review in reviews:
            try:
                score = review.find_element(By.XPATH, "./div[contains(@class,'ipl-ratings-bar')]//span[contains(@class,'rating-other-user-rating')]//span")
                review_text = review.find_element(By.XPATH, "./div[contains(@class,'content')]//div[contains(@class,'text')]")
                text = review_text.get_attribute('innerHTML').replace('<br>', ' ')
                file.write(f'{text}\n--score-- {score.text}\n')
            except NoSuchElementException:
                continue


        driver.get(url)
    file.close()

main()

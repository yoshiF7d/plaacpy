import argparse
import os
import time
#from Colors import Colors
import selenium
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome import service
import getpass

driverPath = '/Users/star-yoshi/plaac/chromedriver'
url = 'https://www.wdb.com/login'
urlContract = 'https://www.wdb.com/woker/attendance/select-contract'
urlCal = 'https://www.wdb.com/woker/attendance/calendar'
cssUser = 'input#email.form-control'
cssPass = 'input#password.form-control'
cssButtonLogin = 'button[type="submit"]'
cssCal = '.wdb-timesheet-calendar'
cssButtonTimeSheet = '#attendanceDescription > div > div.col.text-right > a'
cssCheck = '#inputWorkTime > div:nth-child(5) > div > div.w-100.px-md-3.py-md-2 > div:nth-child(1) > label > span.wdb-checkbox-icon.px-3'
cssCheckInput = '#inputWorkTime > div:nth-child(5) > div > div.w-100.px-md-3.py-md-2 > div:nth-child(1) > label > input'
cssSubmit = '#submitBtn'
cssConfirm = '#applyContactForm > div.card.border-0 > div > form > div.col-md-8.mx-auto.mt-4 > div > div.col-8 > button'
cssReConfirm = 'body > div > div.wdb-main-container > div.wdb-l-main > main > div.wdb-container.container.container-768.pt-md-4 > div.card.border-0 > div > form > div > div > div.col-8 > button'
cssBriefing = '#report > div > div.w-100.px-md-3.py-md-2 > textarea'

def login(driver,usern,passw):
    driver.find_element(By.CSS_SELECTOR,cssUser).send_keys(usern)
    driver.find_element(By.CSS_SELECTOR,cssPass).send_keys(passw)
    driver.find_element(By.CSS_SELECTOR,cssButtonLogin).click()

def clickDate(driver,date):
    cal = driver.find_element(By.CSS_SELECTOR,cssCal)
    xpathDate = f'./tbody/tr/td/a/span/span[text()={date}]/../..' 
    hits = cal.find_elements(By.XPATH,xpathDate)
    if hits:
        hits[0].click()

def parseList(strl):
    return strl.strip('[{()}]').split(',')
    
class Report():
    def __init__(self,date,*,start,end,exclude,briefing):
        self.date = date
        self.start = start
        self.end = end
        self.exclude = exclude,
        self.briefing = briefing
    
    def submit(self,driver):
        driver.get(urlCal)
        cal = driver.find_element(By.CSS_SELECTOR,cssCal)
        xpathDate = f'./tbody/tr/td/a/span/span[text()={self.date}]/../..' 
        hits = cal.find_elements(By.XPATH,xpathDate)
        if hits:
            hits[0].click()
        driver.find_element(By.CSS_SELECTOR,cssButtonTimeSheet).click()
        if not driver.find_element(By.CSS_SELECTOR,cssCheckInput).is_selected():
            driver.find_element(By.CSS_SELECTOR,cssCheck).click()

        if self.briefing is not None:
            driver.find_element(By.CSS_SELECTOR,cssBriefing).send_keys(self.briefing)
    
        driver.find_element(By.CSS_SELECTOR,cssSubmit).click()
        driver.find_element(By.CSS_SELECTOR,cssConfirm).click()
        driver.find_element(By.CSS_SELECTOR,cssReConfirm).click()
    
parser = argparse.ArgumentParser()
parser.add_argument('--user',help='username')
parser.add_argument('--dateList',help='dateList')
#parser.add_argument('pass',help='password')
args = parser.parse_args()

if args.dateList is not None:
    args.dateList = parseList(args.dateList)
else:
    args.dateList = []

os.system('')
print('username : ' + args.user)
passw = getpass.getpass(prompt='password:')

chrome_service = service.Service(executable_path=driverPath)
options = ChromeOptions()
#options.add_argument('--headless')
#options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})
#options.binary_location = chromiumPath
driver = selenium.webdriver.Chrome(service=chrome_service,options=options)

try:    
    driver.get(url)
    login(driver,args.user,passw)
    driver.get(urlContract)
    
    for date in args.dateList:
        driver.get(urlCal)
        clickDate(driver,date)
        driver.find_element(By.CSS_SELECTOR,cssButtonTimeSheet).click()
        if not driver.find_element(By.CSS_SELECTOR,cssCheckInput).is_selected():
            driver.find_element(By.CSS_SELECTOR,cssCheck).click()
        driver.find_element(By.CSS_SELECTOR,cssSubmit).click()
        driver.find_element(By.CSS_SELECTOR,cssConfirm).click()
        driver.find_element(By.CSS_SELECTOR,cssReConfirm).click()
   #time.sleep(5)
finally:
    driver.quit()

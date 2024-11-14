pip install selenium
from selenium import webdriver
driver = webdriver.Chrome()
driver.get("https://www.twitch.tv/directory/game/Art")
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
...

# configure webdriver
options = Options()
options.headless = True  # hide GUI
options.add_argument("--window-size=1920,1080")  # set window size to native GUI size
options.add_argument("start-maximized")  # ensure window is full-screen

...
driver = webdriver.Chrome(options=options)
#                         ^^^^^^^^^^^^^^^
driver.get("https://www.twitch.tv/directory/game/Art")
from parsel import Selector
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

from parsel import Selector

sel = Selector(text=driver.page_source)
parsed = []
for item in sel.xpath("//div[contains(@class,'tw-tower')]/div[@data-target]"):
    parsed.append({
        'title': item.css('h3::text').get(),
        'url': item.css('.tw-link::attr(href)').get(),
        'username': item.css('.tw-link::text').get(),
        'tags': item.css('.tw-tag ::text').getall(),
        'viewers': ''.join(item.css('.tw-media-card-stat::text').re(r'(\d+)')),
    })

parsed

from selenium import webdriver
from selenium.webdriver.common.by import By

# Initialize the driver
driver = webdriver.Chrome()
driver.get("https://www.twitch.tv/")

# Locate the search box using By.CSS_SELECTOR
search_box = driver.find_element(By.CSS_SELECTOR, 'input[aria-label="Search Input"]') 
search_box.send_keys('fast painting')

from selenium import webdriver

driver = webdriver.Chrome()
driver.get("https://www.twitch.tv/directory/game/Art")
# find last item and scroll to it
driver.execute_script("""
let items=document.querySelectorAll('.tw-tower>div');
items[items.length-1].scrollIntoView();
""")

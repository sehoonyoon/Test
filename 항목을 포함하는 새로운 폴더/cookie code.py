import requests
from bs4 import BeautifulSoup
import pandas as pd

cookies = {
    'PHPSESSID': 'p68sraolt6gphr1qacb0h05fd2',
    'browser_id': '1389e2bc-46cd-43b5-b250-6904add442d1',
}

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
}

session = requests.Session()
session.cookies.update(cookies)
session.headers.update(headers)

# ✅ 실제 URL
url = 'https://www.statiz.co.kr/stats/?m=main&m2=pitching'

res = session.get(url)
print(res.status_code)
print(res.text[:500])

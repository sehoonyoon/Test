from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from io import StringIO
import pandas as pd
import time
import os

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# ✅ 연도별 투수 크롤링
all_pitchers = []
years = [2023, 2024, 2025]

for year in years:
    url = f'https://www.statiz.co.kr/stats/?m=main&m2=pitching&year={year}'
    driver.get(url)
    print(f"{year} 시즌 - 60초 안에 로그인 해주세요!")
    time.sleep(60)

    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, 'table'))
        )
        print(f"{year} 테이블 감지됨!")
        time.sleep(2)
        html = driver.page_source
        tables = pd.read_html(StringIO(html), flavor='html5lib')
        if len(tables) > 0:
            df_year = tables[0]
            df_year['season'] = year  # ✅ 시즌 컬럼 추가
            all_pitchers.append(df_year)
            print(f"{year} 투수 {len(df_year)}명 수집완료!")
    except Exception as e:
        print(f"{year} 크롤링 실패: {e}")

driver.quit()

# ✅ 전체 합치기
if all_pitchers:
    os.makedirs('data', exist_ok=True)
    df_all = pd.concat(all_pitchers, ignore_index=True)
    df_all.to_csv('data/kbo_pitcher_all.csv', index=False, encoding='utf-8-sig')
    print(f"\n전체 투수 {len(df_all)}명 저장완료!")
    print(df_all['season'].value_counts())
    # ✅ 연도별 타자 크롤링
    driver2 = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    all_batters = []

    for year in years:
        url = f'https://www.statiz.co.kr/stats/?m=main&m2=batting&year={year}'
        driver2.get(url)
        print(f"{year} 타자 - 60초 안에 로그인 해주세요!")
        time.sleep(60)

        try:
            WebDriverWait(driver2, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, 'table'))
            )
            print(f"{year} 타자 테이블 감지됨!")
            time.sleep(2)
            html2 = driver2.page_source
            tables2 = pd.read_html(StringIO(html2), flavor='html5lib')
            if len(tables2) > 0:
                df_year2 = tables2[0]
                df_year2['season'] = year
                all_batters.append(df_year2)
                print(f"{year} 타자 {len(df_year2)}명 수집완료!")
        except Exception as e:
            print(f"{year} 타자 크롤링 실패: {e}")

    driver2.quit()

    if all_batters:
        df_batter_all = pd.concat(all_batters, ignore_index=True)
        df_batter_all.to_csv('data/kbo_batter_all.csv', index=False, encoding='utf-8-sig')
        print(f"\n전체 타자 {len(df_batter_all)}명 저장완료!")
        print(df_batter_all['season'].value_counts())

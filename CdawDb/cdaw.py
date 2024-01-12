import os
import requests
import time
from bs4 import BeautifulSoup

def month_name_to_number(month_name):
    from datetime import datetime
    datetime_object = datetime.strptime(month_name, "%b")
    month_number = datetime_object.month
    return month_number


class cdaw:
    def get_univ_all(self):
        url = "https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver1/text_ver/univ_all.txt"
        filename = "univ_all.txt"

        if not os.path.exists(filename):
            response = requests.get(url)
            with open(filename, 'wb') as file:
                file.write(response.content)

        with open(filename, 'r') as file:
            content = file.read().split("\n")[4:]

        return content

    def get_catalog_by_months(self):
        result = {}

        url = "https://cdaw.gsfc.nasa.gov/CME_list/"
        html = requests.get(url).text
        parsed_html = BeautifulSoup(html)
        trs = parsed_html.find_all('tr')
        for tr in trs:
            th = tr.find('th')
            if th is None: continue

            year = th.text
            tds = tr.find_all('td')
            for td in tds:
                links = td.find_all('a')
                for link in links:
                    href = link.get('href')
                    month = link.text

                    if href is None or month is None: continue

                    month_number = month_name_to_number(month)
                    url = "https://cdaw.gsfc.nasa.gov/CME_list/" + href

                    result[(year, month_number)] = url

        return result

    def get_observations_from_month_url(self, url):
        result = []

        success = False
        while not success:
            try:
                html = requests.get(url).text
                parsed_html = BeautifulSoup(html)
                success = True
            except Exception as e:
                print("Error downloading " + url + ":", e)
                print("Retrying in 5 seconds...")
                time.sleep(5)
                pass

        trs = parsed_html.find_all('tr')
        for tr in trs:
            # get th where headers="hd_date_time"
            tds = tr.find_all('td', attrs={'headers': 'hd_date_time'})
            for td in tds:
                link = td.find('a')
                if link is None or link.text.count(":") != 2: continue

                href = link.get('href')
                url2 = url.rsplit('/', 1)[0] + "/" + href
                result.append(url2)

        return result

import os
import requests
import csv
from bs4 import BeautifulSoup
from models import Article
from utils import ResponseStatusCodeException
from tqdm import tqdm
from urllib.parse import urljoin


class ConsultantPlusParser:

    def __init__(self, config):
        self.base_url = config['base_url']
        self.articles_info_file = config['articles_info_file']
        self.session = requests.Session()
        self.session.headers.update({
            'User-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36'
        })
        self.codex_urls = self.__get_codex_urls()
        self.articles_info = self.__get_articles_info()
        self.sorted_articles_info = self.__get_articles_info_sorted_by_codexes()

    def __make_request(self, method, location='', status_code=200, headers=None,
                       params=None, data=None, json_convert=False,
                       custom_location=False, allow_redirects=True, json=None):
        if not custom_location:
            if location:
                url = urljoin(self.base_url, location)
            else:
                url = self.base_url
        else:
            url = location

        response = self.session.request(method, url, headers=headers,
                                        params=params, data=data,
                                        allow_redirects=allow_redirects,
                                        json=json)

        if response.status_code != status_code:
            raise ResponseStatusCodeException(
                f' Got {response.status_code} {response.reason} for URL "{url}"')
        if json_convert:
            json_response = response.json()
            return json_response
        return response

    def __get_articles_info(self):
        articles_info = list()
        if not os.path.exists(self.articles_info_file):
            self.parse()
        with open(self.articles_info_file) as articles_info_file:
            csv_reader = csv.reader(articles_info_file, delimiter=',')
            for row in tqdm(csv_reader):
                articles_info.append(Article(id=int(row[0]), url=row[1], title=row[2], codex_type=row[3]))
        return articles_info

    def __get_articles_info_sorted_by_codexes(self):
        sorted_articles_info = dict()
        for article_info in tqdm(self.articles_info):
            if article_info.codex_type not in sorted_articles_info:
                sorted_articles_info[article_info.codex_type] = list()
            sorted_articles_info[article_info.codex_type].append(article_info)
        return sorted_articles_info

    def __get_codex_urls(self):
        home_page = BeautifulSoup(self.__make_request('GET').text, 'html.parser')
        home_page = home_page.find_all(
            class_='useful-links__list useful-links__list_dashed')
        return list(
            map(lambda element: (
            element.get_text(), element.find('a').get('href')),
                home_page[0].find_all('li')))

    def parse(self):
        current_article_id = 0
        articles_info = list()
        for codex_type, url in tqdm(self.codex_urls):
            codex_page = BeautifulSoup(self.__make_request('GET', 'https:' + url,
                                                           custom_location=True).text, 'html.parser')
            codex_page_content = codex_page.find('div', class_='contents').find('contents')
            for article in tqdm(codex_page_content.find_all('a')):
                if article.get_text().startswith('Статья'):
                    articles_info.append(Article(id=current_article_id, url=article.get('href'), title=article.get_text(), codex_type=codex_type))
                    current_article_id += 1
        if os.path.exists(self.articles_info_file):
            os.remove(self.articles_info_file)
        with open(self.articles_info_file, mode='w') as articles_info_file:
            articles_info_writer = csv.writer(articles_info_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for article in tqdm(articles_info):
                articles_info_writer.writerow([article.id, article.url, article.title, article.codex_type])

    def get_article_text_by_id(self, article_id):
        article = list(filter(lambda item: item.id == article_id, self.articles_info))[0]
        article_page = BeautifulSoup(self.__make_request('GET', self.base_url + article.url, custom_location=True).text, 'html.parser')
        text = ''
        for part_of_text in article_page.find('div', class_='text').find_all('span'):
            text += part_of_text.get_text()
        return text

    def get_article_url_by_id(self, article_id):
        return list(filter(lambda item: item.id == article_id, self.articles_info))[0].url

    def get_links_on_target_words_by_id_and_target_words(self, article_id, target_words):
        article = list(filter(lambda item: item.id == article_id, self.articles_info))[0]
        article_page = BeautifulSoup(self.__make_request('GET', self.base_url + article.url, custom_location=True).text, 'html.parser')
        for part_of_text in article_page.find('div', class_='text').find_all('span'):
            if target_words in part_of_text.get_text() and 'если иное не предусмотрено' in str(part_of_text) and 'href' in str(part_of_text).split('если иное не предусмотрено')[1].split('.')[0]:
                return str(part_of_text).split('если иное не предусмотрено')[1].split('.')[0].split('href="')[1].split('">')[0]

    def get_text_by_url(self, url):
        page = BeautifulSoup(self.__make_request('GET', self.base_url + url, custom_location=True).text, 'html.parser')
        text = ''
        for part_of_text in page.find('div', class_='text').find_all('span'):
            text += part_of_text.get_text()
        return text

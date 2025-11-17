import pandas as pd
import requests

class StockUniverse(object) :
    @staticmethod
    def get_sp500_stock_info() -> pd.DataFrame :
        url = 'https://en.wikipedia.org/wiki/List_of_S&P_500_companies'
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/100.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()  # 응답 오류있으면 예외
        dfs = pd.read_html(resp.text, header=0)
        res = dfs[1]
        return res

    @staticmethod
    def get_sp100_stock_info() -> pd.DataFrame :
        url = 'https://en.wikipedia.org/wiki/S%26P_100'
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/100.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()  # 응답 오류있으면 예외
        dfs = pd.read_html(resp.text, header=0)
        res = dfs[2]
        return res

    @staticmethod
    def get_sp400_stock_info() -> pd.DataFrame :
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/100.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()  # 응답 오류있으면 예외
        dfs = pd.read_html(resp.text, header=0)
        res = dfs[0]
        return res

    @staticmethod
    def get_sp600_stock_info() -> pd.DataFrame :
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/100.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()  # 응답 오류있으면 예외
        dfs = pd.read_html(resp.text, header=0)
        res = dfs[1]
        return res

    @staticmethod
    def get_nasdaq100_stock_info() -> pd.DataFrame :
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/100.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()  # 응답 오류있으면 예외
        dfs = pd.read_html(resp.text, header=0)
        res = dfs[4]
        return res
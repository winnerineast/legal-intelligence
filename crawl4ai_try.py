import asyncio
from crawl4ai import AsyncWebCrawler, CacheMode

async def main(my_url):
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url=my_url)
        # Soone will be change to result.markdown
        print(result.markdown_v2.raw_markdown)

if __name__ == "__main__":
    url_list=[
        'https://www.agbi.com/',
        'https://asahichinese-j.com/',
        'https://brazilian.report/',
        'https://chinese.aljazeera.net/',
        'http://www.dw.com/',
        'https://ekstrabladet.dk/',
        'https://www.foxnews.com/',
        'http://www.ftchinese.com/',
        'https://www.huffpost.com/',
        'https://indianexpress.com/',
        'https://www.medcom.id/',
        'https://www.mingpao.com/',
        'https://www.nationthailand.com/',
        'https://www.nbcnews.com/business',
        'https://nepalnews.com/',
        'https://news.google.com/',
        'https://nournews.ir/zh',
        'https://www.nytimes.com/',
        'https://www.prothomalo.com/',
        'https://www.qna.org.qa/en',
        'http://www.rfi.fr/',
        'https://www.rnz.co.nz/',
        'https://www.sbs.com.au/',
        'https://www.sinchew.com.my/',
        'https://thefrontierpost.com/',
        'https://www.theguardian.com/',
        'https://www.thenationalnews.com/uae/'
        'https://www.timesofisrael.com/',
        'https://en.vietnamplus.vn/',
        'https://www.washingtonpost.com/',
        'https://www.wsj.com/',
        'https://news.yahoo.com/',
        'https://www.zaobao.com/']

    for item in url_list:
        asyncio.run(main(item))
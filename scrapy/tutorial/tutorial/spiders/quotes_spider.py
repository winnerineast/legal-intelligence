from pathlib import Path
import scrapy_splash
import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            "https://www.jiqizhixin.com/users/27999d5c-8072-4eb7-8f45-f4c1bcc1d0b9"
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse, meta={
                'splash':{
                    'args':{
                        'html':1,
                        'png': 1,
                    },
                    # optional parameters
                    'endpoint': 'render.json',  # optional; default is render.json
                    'splash_url': '<url>',  # optional; overrides SPLASH_URL
                    'slot_policy': scrapy_splash.SlotPolicy.PER_DOMAIN,
                    'splash_headers': {},  # optional; a dict with headers sent to Splash
                    'dont_process_response': True,  # optional, default is False
                    'dont_send_headers': True,  # optional, default is False
                    'magic_response': False,  # optional, default is True
                }
            })

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = f"quotes-{page}.html"
        Path(filename).write_bytes(response.body)
        self.log(f"Saved file {filename}")
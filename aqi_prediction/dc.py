from openaq import OpenAQ, AsyncOpenAQ
import asyncio
import pandas as pd
import csv
import time
from math import ceil

API_KEY = '02bfbbe947ca3f9e84ca526ede43052c4d5a3c20ae05a65552efd93272bb3cd6'

with OpenAQ(api_key=API_KEY) as client:
    locations = client.measurements.list(
        sensors_id=24434, datetime_from='2024-01-01', datetime_to='2024-12-24', page=2)
    meta = locations.meta
    found = meta.found
    limit = 1000

    # pages = ceil(found/limit)
    print(meta)
    print(locations.results[0])
    print(f'Found {found} locations, {0} pages')
    print('Getting data...')

import requests
import time
import os
import threading
from datetime import datetime, timedelta

from CdawDb.sunpy_source import sunpy_source
from cdaw import cdaw
from cdaw_cme_repository import cdaw_cme_repository

def get_timespans():
    a = datetime(1996, 12, 7, 0, 0, 0)
    b = datetime(2000, 12, 31, 23, 59, 59)
    c = datetime(2001, 12, 31, 23, 59, 59)
    d = datetime(2002, 12, 31, 23, 59, 59)
    start = a
    end = b + timedelta(days=365)
    #end = datetime(2024, 1, 1, 23, 59, 59)

    # a + 1yr completed
    # b; c completed

    # get timespans for each day
    timespans = []
    current = start
    while current < end:
        next = current + timedelta(days=1)
        timespans.append((current, next))
        current = next

    return timespans

class fits_manager:
    def __init__(self, dbconn):
        self.dbconn = dbconn
        self.repo = cdaw_cme_repository(dbconn)
        self.sunpy_source = sunpy_source()

    def download_all_fits(self):
        timespans = get_timespans()

        # create directory if it doesn't exist ./downloads/fits/
        if not os.path.exists("./downloads/fits/"):
            os.makedirs("./downloads/fits/")

        for (start, end) in timespans:
            print("Downloading data for " + str(start) + " - " + str(end))
            obs = self.sunpy_source.get_fits_from_timespan(start, end)
            if obs is None or len(obs) == 0:
                print("No data found")
                continue

            obsd = self.sunpy_source.download_fits(obs)

            pass


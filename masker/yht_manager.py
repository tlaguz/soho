import os
from datetime import datetime

import numpy as np

from astropy.io import fits as astrofits

from masker.repositories.yht_repository import YhtRepository
from masker.yht_reader import yht_reader


class yht_manager:
    def __init__(self, dbconn):
        self.dbconn = dbconn
        self.repo = YhtRepository(dbconn)

    def scan_directory(self, directory):
        # find all files in this directory
        for filename in os.listdir(directory):
            filename = os.path.join(directory, filename)
            if not filename.endswith('.yht'):
                continue

            print(f"Scanning {filename}")

            # read metadata from the file
            yht_f = yht_reader(filename)
            yht_f.parse_data()

            date = datetime.strptime(f"{yht_f.file.date_obs} {yht_f.file.time_obs}", '%Y/%m/%d %H:%M:%S')

            basefilename = os.path.basename(filename)

            try_get = self.repo.get_yht(basefilename)
            if try_get is not None:
                print(f"File {basefilename} already exists in the database")
            else:
                # add it to the database
                self.repo.add_yht(yht(
                    filename=basefilename,
                    date=date.strftime('%Y-%m-%d %H:%M:%S'),
                    detector=yht_f.file.detector,
                    observer=yht_f.file.observer,
                    halo=yht_f.file.halo
                ))

            try_points = self.repo.get_yht_points(basefilename)
            # observations are doubled. The first is used, the second is ignored
            added_dates = []

            # add all points to the database
            for obs in yht_f.file.observations:
                date = datetime.strptime(f"{obs.date} {obs.time}", '%Y/%m/%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
                if len(list(filter(lambda x: x[1] == date, try_points))) == 0 and date not in added_dates:
                    self.repo.add_yht_point(basefilename, date, obs.row, obs.col, obs.height, obs.tel)
                    added_dates.append(date)

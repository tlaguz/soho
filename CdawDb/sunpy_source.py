import astropy.units as u
from astropy.coordinates import SkyCoord

from sunpy.map import Map
from sunpy.map.maputils import all_coordinates_from_map
from sunpy.net import Fido
from sunpy.net import attrs as a

import time

class sunpy_source():
    def get_fits_from_timespan(self, start, end):
        result = Fido.search(a.Time(start, end),
                             a.Instrument.lasco,
                             a.Detector.c2 | a.Detector.c3)

        return result

    def download_fits(self, obs):
        success = False
        while not success:
            try:
                result = Fido.fetch(obs, path="./downloads/sunpy/{fileid}")

                if result.errors is not None and len(result.errors) > 0:
                    print("Error downloading, retrying in 5 seconds")
                    time.sleep(5)
                    continue

                success = True
            except:
                print("Error downloading, retrying in 5 seconds")
                time.sleep(5)


        return result

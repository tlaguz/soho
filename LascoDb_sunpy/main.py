import sqlite3
import argparse

import astropy.units as u
from astropy.coordinates import SkyCoord

from sunpy.map import Map
from sunpy.map.maputils import all_coordinates_from_map
from sunpy.net import Fido
from sunpy.net import attrs as a

if __name__ == '__main__':
    result = Fido.search(a.Time('1997-01-01 00:00', '1997-01-01 06:36'),
                         a.Instrument.lasco,
                         a.Detector.c2)
    pass
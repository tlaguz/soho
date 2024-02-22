import os
from datetime import datetime
from astropy.io import fits as astrofits

from masker.repositories.fits_repository import FitsRepository, FitsDto
from masker.yht_reader import yht_reader


class fits_manager:
    def __init__(self, dbconn):
        self.dbconn = dbconn
        self.repo = FitsRepository(dbconn)

    def scan_directory(self, directory):
        for subdir, dirs, files in os.walk(directory):
            for file in files:
                filename = os.path.join(subdir, file)
                if not filename.endswith('.fits') and not filename.endswith('.fts'):
                    continue

                path = os.path.relpath(filename, directory)
                if os.path.getsize(filename) == 0:
                    print(f"File {path} has zero length, ignoring")
                    continue

                if os.path.getsize(filename) != 2108160:
                    print(f"File {path} has invalid length, deleting from previous scans and ignoring")
                    self.repo.remove_fits_by_filename(path)
                    continue

                if self.repo.check_fits_exists(path):
                    print(f"File {path} already exists in the database")
                    continue

                print(f"Scanning {filename}")

                with astrofits.open(filename) as hdul:
                    header = hdul[0].header
                    date = datetime.strptime(f"{header['DATE-OBS']} {header['TIME-OBS']}", '%Y/%m/%d %H:%M:%S.%f')
                    self.repo.add_fits(FitsDto(
                        filename=path,
                        date=date.strftime('%Y-%m-%d %H:%M:%S.%f'),
                        detector=header['DETECTOR'],
                        naxis1=header['NAXIS1'],
                        naxis2=header['NAXIS2'],
                    ))

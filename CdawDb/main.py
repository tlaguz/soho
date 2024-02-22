import sqlite3
import argparse

from CdawDb.fits_manager import fits_manager
from cdaw_cme_manager import cdaw_cme_manager
from dbstruct import create_db

dbconn = sqlite3.connect('db.sqlite3')
cursor = dbconn.cursor()

parser = argparse.ArgumentParser(description='Interact with CDAW database')
parser.add_argument('--create-db', action='store_true', help='Create database')
parser.add_argument('--download-yhts', action='store_true', help='Download data')
parser.add_argument('--download-fits', action='store_true', help='Download data')

args = parser.parse_args()

if __name__ == '__main__':
    if args.create_db:
        create_db(dbconn)
        exit(0)
    if args.download_yhts:
        manager = cdaw_cme_manager(dbconn)
        manager.download_all_yhts()
        exit(0)
    if args.download_fits:
        manager = fits_manager(dbconn)
        manager.download_all_fits()
        exit(0)

    parser.print_help()

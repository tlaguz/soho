import sqlite3
import argparse

from cdaw_cme_manager import observations_manager
from dbstruct import create_db

dbconn = sqlite3.connect('db.sqlite3')
cursor = dbconn.cursor()

parser = argparse.ArgumentParser(description='Interact with CDAW database')
parser.add_argument('--create', action='store_true', help='Create database')
parser.add_argument('--download-data', action='store_true', help='Download data')

args = parser.parse_args()

if __name__ == '__main__':
    if args.create:
        create_db(dbconn)
        exit(0)
    if args.download_data:
        manager = observations_manager(dbconn)
        manager.download_all_yhts()
        exit(0)

    parser.print_help()

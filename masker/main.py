import argparse
import sqlite3

from dbstruct import create_db
from fits_manager import fits_manager
from masker.animator import Animator
from masker.previewer import Previewer
from yht_reader import yht_reader
from yht_manager import yht_manager

db_file = "/home/tlaguz/db.sqlite3"
yhts_directory = "/mnt/mlpool/yhts/"
fits_directory = "/mnt/mlpool/soho_seiji/"

dbconn = sqlite3.connect(db_file, timeout=300000.0)
dbconn.row_factory = sqlite3.Row
cursor = dbconn.cursor()

parser = argparse.ArgumentParser(description='Scan yht and fits files and create a database of them')
parser.add_argument('--create', action='store_true', help='Create database')
parser.add_argument('--scan-yht', action='store_true', help='Scan yht files')
parser.add_argument('--scan-fits', action='store_true', help='Scan fits files')
parser.add_argument('--animate', action='store_true', help='Animate specified observation files')
parser.add_argument('--train-status', action='store_true', help='Prints plots of the training status of the models')
parser.add_argument('--yht', action='append', help='YHT file to animate')
parser.add_argument('--model', action='append', help='Model path to use for animation')

args = parser.parse_args()

if __name__ == '__main__':
    if args.create:
        create_db(dbconn)
        exit(0)
    if args.scan_yht:
        manager = yht_manager(dbconn)
        manager.scan_directory(yhts_directory)
        exit(0)
    if args.scan_fits:
        manager = fits_manager(dbconn)
        manager.scan_directory(fits_directory)
        exit(0)
    if args.animate:
        animator = Animator(dbconn, fits_directory)
        animator.animate(args.yht, args.model)
        exit(0)
    if args.train_status:
        previewer = Previewer()
        previewer.preview_metadata(args.model[0])

    parser.print_help()
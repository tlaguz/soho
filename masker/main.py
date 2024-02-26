import argparse
import sqlite3

from dbstruct import create_db
from fits_manager import fits_manager
from masker.animator import Animator
from masker.datasets.cache_warmer import CacheWarmer
from masker.datasets.cached_dataset import CachedDataset
from masker.datasets.create_dataset import create_dataset, create_validation_dataset
from masker.previewer import Previewer
from masker.utils import get_paths
from yht_reader import yht_reader
from yht_manager import yht_manager

paths = get_paths()

dbconn = sqlite3.connect(paths.db_file, timeout=300000.0)
dbconn.row_factory = sqlite3.Row
cursor = dbconn.cursor()

parser = argparse.ArgumentParser(description='Scan yht and fits files and create a database of them')
parser.add_argument('--create', action='store_true', help='Create database')
parser.add_argument('--scan-yht', action='store_true', help='Scan yht files')
parser.add_argument('--scan-fits', action='store_true', help='Scan fits files')
parser.add_argument('--animate', action='store_true', help='Animate specified observation files')
parser.add_argument('--train-status', action='store_true', help='Prints plots of the training status of the models')
parser.add_argument('--warm-cache', action='store_true', help='Warm up the cache')
parser.add_argument('--drop-cache', action='store_true', help='Drop the cache')
parser.add_argument('--yht', action='append', help='YHT file to animate')
parser.add_argument('--model', action='append', help='Model path to use for animation')

args = parser.parse_args()

if __name__ == '__main__':
    if args.create:
        create_db(dbconn)
        exit(0)
    if args.scan_yht:
        manager = yht_manager(dbconn)
        manager.scan_directory(paths.yhts_directory)
        exit(0)
    if args.scan_fits:
        manager = fits_manager(dbconn)
        manager.scan_directory(paths.fits_directory)
        exit(0)
    if args.animate:
        animator = Animator(dbconn)
        animator.animate(args.yht, args.model)
        exit(0)
    if args.train_status:
        previewer = Previewer()
        previewer.preview_metadata(args.model[0])
        exit(0)
    if args.warm_cache:
        def create_dataset_lambda():
            dbconn = sqlite3.connect(paths.db_file, timeout=300000.0)
            dbconn.row_factory = sqlite3.Row
            dataset = create_dataset(dbconn, augment=False)
            return dataset

        def create_validation_dataset_lambda():
            dbconn = sqlite3.connect(paths.db_file, timeout=300000.0)
            dbconn.row_factory = sqlite3.Row
            dataset = create_validation_dataset(dbconn, augment=False)
            return dataset

        print("Warming training cache")
        warmer = CacheWarmer(create_dataset_lambda)
        warmer.warm()

        print("Warming validation cache")
        warmer = CacheWarmer(create_validation_dataset_lambda)
        warmer.warm()

        exit(0)

    if args.drop_cache:
        import shutil
        shutil.rmtree(paths.valid_cache, ignore_errors=True)
        shutil.rmtree(paths.train_cache, ignore_errors=True)

        print("Caches dropped")
        exit(0)

    parser.print_help()
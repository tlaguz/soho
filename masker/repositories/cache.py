import hashlib
import os

from masker.utils import get_paths
import pickle

class Cache():
    def __init__(self, name):
        self.name = name
        self.paths = get_paths()
        os.makedirs(self.paths.repository_cache, exist_ok=True)

    def _get_cache_path(self, key):
        return self.paths.repository_cache + '/' + self.name + '_' + str(key) + '.cache'

    def get(self, key):
        # open file pickle
        try:
            with open(self._get_cache_path(key), 'rb') as f:
                return pickle.load(f)
        except:
            return None

    def set(self, key, value):
        # write to file pickle
        with open(self._get_cache_path(key), 'wb') as f:
            pickle.dump(value, f)


    def get_hash(self, key):
        if isinstance(key, str):
            key = key.encode('utf-8')

        return hashlib.md5(key).hexdigest()

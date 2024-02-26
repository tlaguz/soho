import os

from astropy.io import fits

from masker.utils import get_paths


class FitsLoader():
    def __init__(self):
        self.paths = get_paths()
        os.makedirs(self.paths.fits_cache, exist_ok=True)

    def _load_fits(self, path):
        try:
            return fits.open(path)
        except Exception as e:
            print(f"Error loading fits file {path}: {e}")
            return None

    def get_cached_fits(self, path):
        cache_file = os.path.join(self.paths.fits_cache, path)
        if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
            return self._load_fits(cache_file)

        fits_file = os.path.join(self.paths.fits_directory, path)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        # copy the file to the cache
        with open(fits_file, "rb") as f:
            with open(cache_file, "wb") as f2:
                f2.write(f.read())

        return self._load_fits(cache_file)


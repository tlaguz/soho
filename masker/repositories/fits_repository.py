from dataclasses import dataclass

from masker.repositories.cache import Cache


@dataclass
class FitsDto:
    filename: str
    date: str
    date2: str
    detector: str
    naxis1: int
    naxis2: int

@dataclass
class FitsPointDto(FitsDto):
    row: int
    col: int

class FitsRepository:
    def __init__(self, dbconn):
        self.dbconn = dbconn
        self.cache = Cache("fits_repository")

    def add_fits(self, fits: FitsDto):
        cursor = self.dbconn.cursor()
        cursor.execute("""
        INSERT INTO fits (filename, date, detector, naxis1, naxis2) VALUES (?, ?, ?, ?, ?)
        """, (fits.filename, fits.date, fits.detector, fits.naxis1, fits.naxis2))
        self.dbconn.commit()

    def get_fits(self, filename):
        cursor = self.dbconn.cursor()
        cursor.execute("""
        SELECT * FROM fits WHERE filename = ?
        """, (filename,))
        result = cursor.fetchone()
        return FitsDto(*result) if result is not None else None

    def check_fits_exists(self, filename):
        return self.get_fits(filename) is not None

    def remove_fits_by_filename(self, filename):
        cursor = self.dbconn.cursor()
        cursor.execute("""
        DELETE FROM fits WHERE filename = ?
        """, (filename,))
        self.dbconn.commit()

    def get_fits_points_by_yht_filename(self, filename_yht):
        hash = self.cache.get_hash(filename_yht)
        key = f"fits_by_yht_{hash}"

        result = self.cache.get(key)
        if result is not None:
            return result

        cursor = self.dbconn.cursor()
        cursor.execute("""
        select                     
            fits.filename,
            fits.date,
            fits.date2,
            fits.detector,
            fits.naxis1,
            fits.naxis2,
            
            yht_point.row,
            yht_point.col
        from
            fits
        join
            yht_point on yht_point.date = fits.date2
            and yht_point.detector = fits.detector
        where
            yht_point.filename = ?;
        """, (filename_yht,))
        result = cursor.fetchall()
        result = [FitsPointDto(*row) for row in result]
        self.cache.set(key, result)
        return result

    def get_previous_fits(self, filename):
        hash = self.cache.get_hash(filename)
        key = f"previous_fits_{hash}"

        result = self.cache.get(key)
        if result is not None:
            return result

        cursor = self.dbconn.cursor()
        cursor.execute("""
WITH temp_table AS (
     SELECT 
        fits.filename,
        fits.date,
        fits.date2,
        fits.detector,
        fits.naxis1,
        fits.naxis2,
        ROW_NUMBER() OVER (ORDER BY fits.date2) as row_number
     FROM 
        fits
     WHERE
        fits.naxis1 = 1024 AND
        fits.naxis2 = 1024 AND
        detector = (
            SELECT detector FROM fits WHERE filename = ?
        )
)
SELECT 
    filename,
    date,
    date2,
    detector,
    naxis1,
    naxis2 
FROM 
    temp_table
WHERE 
    row_number = (
       SELECT row_number FROM temp_table WHERE filename = ?
    ) - 1;
        """, (filename,filename))
        result = cursor.fetchone()
        result = FitsDto(*result) if result is not None else None
        self.cache.set(key, result)
        return result
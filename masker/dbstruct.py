
def create_db(dbconn):
    cursor = dbconn.cursor()
    cursor.execute("""
CREATE TABLE IF NOT EXISTS yht (
    filename TEXT PRIMARY KEY,
    date TEXT,
    detector TEXT,
    observer TEXT,
    halo BOOLEAN
    );
    """)

    cursor.execute("""
CREATE TABLE IF NOT EXISTS yht_point (
    filename TEXT,
    date TEXT,
    row INTEGER,
    col INTEGER,
    height REAL,
    detector TEXT,
    PRIMARY KEY (filename, date),
    FOREIGN KEY (filename) REFERENCES yht(filename)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fits (
        filename TEXT PRIMARY KEY,
        date TEXT,
        date2 TEXT,
        detector TEXT,
        naxis1 INTEGER,
        naxis2 INTEGER
        );
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_observations_begin_date ON yht (date);
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_fits_date ON fits (date);
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_fits_naxis1 ON fits (naxis1);
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_fits_date2 ON fits (date2);
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_fits_date2_naxis ON fits (date2, naxis1, naxis2);
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_fits_naxis2 ON fits (naxis2);
        """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_yht_point_date ON yht_point (date);
    """)

    cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_yht_point_detector ON yht_point (detector);
    """)

    cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fits_detector ON fits (detector);
    """)

    # create multi index on yht_point date and detector
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_yht_point_date_detector ON yht_point (date, detector);
    """)

    # create multi index on fits date and detector
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_fits_date_detector ON fits (date, detector);
    """)


    dbconn.commit()


###
# -- Add a new column `date2`
# -- Update the new column with date values without milliseconds
# UPDATE fits SET date2 = strftime('%Y-%m-%d %H:%M:%S', date);
#
#

# select count(*) from fits join yht_point on yht_point.date = fits.date2 and yht_point.detector = fits.detector;




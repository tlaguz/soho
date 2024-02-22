    DROP TABLE IF EXISTS training_points_c2;
    DROP TABLE IF EXISTS training_points_c3;

    CREATE TABLE IF NOT EXISTS training_points_c2 (
        id INTEGER PRIMARY KEY,
        filename TEXT,
        filename_yht TEXT,
        date TEXT,
        date2 TEXT,
        detector TEXT,
        naxis1 INTEGER,
        naxis2 INTEGER,
        observer TEXT,
        aggregated_row TEXT,
        aggregated_col TEXT,
        aggregated_height TEXT
        );

    CREATE TABLE IF NOT EXISTS training_points_c3 (
        id INTEGER PRIMARY KEY,
        filename TEXT,
        filename_yht TEXT,
        date TEXT,
        date2 TEXT,
        detector TEXT,
        naxis1 INTEGER,
        naxis2 INTEGER,
        observer TEXT,
        aggregated_row TEXT,
        aggregated_col TEXT,
        aggregated_height TEXT
        );

    CREATE INDEX IF NOT EXISTS idx_fits_tpc2_id ON training_points_c2 (id);
    CREATE INDEX IF NOT EXISTS idx_fits_tpc2_date ON training_points_c2 (date);
    CREATE INDEX IF NOT EXISTS idx_fits_tpc2_detector ON training_points_c2 (detector);
    CREATE INDEX IF NOT EXISTS idx_fits_tpc2_naxis1 ON training_points_c2 (naxis1);
    CREATE INDEX IF NOT EXISTS idx_fits_tpc2_date2 ON training_points_c2 (date2);
    CREATE INDEX IF NOT EXISTS idx_fits_tpc2_date2_naxis ON training_points_c2 (date2, naxis1, naxis2);
    CREATE INDEX IF NOT EXISTS idx_fits_tpc2_naxis2 ON training_points_c2 (naxis2);

    CREATE INDEX IF NOT EXISTS idx_fits_tpc3_id ON training_points_c3 (id);
    CREATE INDEX IF NOT EXISTS idx_fits_tpc3_date ON training_points_c3 (date);
    CREATE INDEX IF NOT EXISTS idx_fits_tpc3_detector ON training_points_c3 (detector);
    CREATE INDEX IF NOT EXISTS idx_fits_tpc3_naxis1 ON training_points_c3 (naxis1);
    CREATE INDEX IF NOT EXISTS idx_fits_tpc3_date2 ON training_points_c3 (date2);
    CREATE INDEX IF NOT EXISTS idx_fits_tpc3_date2_naxis ON training_points_c3 (date2, naxis1, naxis2);
    CREATE INDEX IF NOT EXISTS idx_fits_tpc3_naxis2 ON training_points_c3 (naxis2);


INSERT INTO training_points_c2 (filename, date, date2, detector, naxis1, naxis2, filename_yht, observer, aggregated_row, aggregated_col, aggregated_height)
SELECT
    fits.filename, fits.date, fits.date2, fits.detector, fits.naxis1, fits.naxis2,
    min(yht_point.filename) as filename_yht, min(yht.observer), group_concat(yht_point.row, ', '),
    group_concat(yht_point.col, ', '), group_concat(yht_point.height, ', ')
FROM fits
JOIN
    yht_point ON yht_point.date = fits.date2
    AND yht_point.detector = fits.detector
JOIN
    yht ON yht.filename = yht_point.filename
WHERE
    fits.date2 IN (SELECT fits2.date2 FROM fits AS fits2 GROUP BY fits2.date2 HAVING count(fits2.date2) = 1)
    AND fits.naxis1 = 1024
    AND fits.naxis2 = 1024
    AND fits.detector = 'C2'
GROUP BY fits.filename, fits.date, fits.date2, fits.detector, fits.naxis1, fits.naxis2
ORDER BY fits.date2;

INSERT INTO training_points_c3 (filename, date, date2, detector, naxis1, naxis2, filename_yht, observer, aggregated_row, aggregated_col, aggregated_height)
SELECT
    fits.filename, fits.date, fits.date2, fits.detector, fits.naxis1, fits.naxis2,
    min(yht_point.filename) as filename_yht, min(yht.observer), group_concat(yht_point.row, ', '),
    group_concat(yht_point.col, ', '), group_concat(yht_point.height, ', ')
FROM fits
JOIN
    yht_point ON yht_point.date = fits.date2
    AND yht_point.detector = fits.detector
JOIN
    yht ON yht.filename = yht_point.filename
WHERE
    fits.date2 IN (SELECT fits2.date2 FROM fits AS fits2 GROUP BY fits2.date2 HAVING count(fits2.date2) = 1)
    AND fits.naxis1 = 1024
    AND fits.naxis2 = 1024
    AND fits.detector = 'C3'
GROUP BY fits.filename, fits.date, fits.date2, fits.detector, fits.naxis1, fits.naxis2
ORDER BY fits.date2;


ALTER TABLE training_points_c2 ADD COLUMN filename_prev TEXT;

ALTER TABLE training_points_c3 ADD COLUMN filename_prev TEXT;

WITH temp_table AS (
    SELECT curr.filename, LAG(prev.filename) OVER (ORDER BY prev.date2) AS filename_prev
    FROM fits AS curr
    INNER JOIN fits AS prev
        ON curr.date2 = prev.date2
        AND curr.detector = prev.detector
        AND prev.naxis1 = 1024
        AND prev.naxis2 = 1024
        AND curr.detector = 'C2'
)
UPDATE training_points_c2
SET filename_prev = (SELECT filename_prev FROM temp_table WHERE training_points_c2.filename = temp_table.filename);

WITH temp_table AS (
    SELECT curr.filename, LAG(prev.filename) OVER (ORDER BY prev.date2) AS filename_prev
    FROM fits AS curr
    INNER JOIN fits AS prev
        ON curr.date2 = prev.date2
        AND curr.detector = prev.detector
        AND prev.naxis1 = 1024
        AND prev.naxis2 = 1024
        AND curr.detector = 'C3'
)
UPDATE training_points_c3
SET filename_prev = (SELECT filename_prev FROM temp_table WHERE training_points_c3.filename = temp_table.filename);

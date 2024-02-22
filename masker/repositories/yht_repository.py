from dataclasses import dataclass

@dataclass
class YhtDto:
    filename: str
    date: str
    detector: str
    observer: str
    halo: bool

class YhtRepository:
    def __init__(self, dbconn):
        self.dbconn = dbconn

    def add_yht(self, yht: YhtDto):
        cursor = self.dbconn.cursor()
        cursor.execute("""
        INSERT INTO yht (filename, date, detector, observer, halo) VALUES (?, ?, ?, ?, ?)
        """, (yht.filename, yht.date, yht.detector, yht.observer, yht.halo))
        self.dbconn.commit()

    def get_yht(self, filename):
        cursor = self.dbconn.cursor()
        cursor.execute("""
        SELECT * FROM yht WHERE filename = ?
        """, (filename,))
        return cursor.fetchone()

    def get_yht_paged(self, page, per_page=50):
        cursor = self.dbconn.cursor()
        cursor.execute("""
        SELECT * FROM yht ORDER BY filename LIMIT ? OFFSET ?
        """, (per_page, (page)*per_page))
        return cursor.fetchall()

    def add_yht_point(self, filename, date, row, col, height, detector):
        cursor = self.dbconn.cursor()
        cursor.execute("""
        INSERT INTO yht_point (filename, date, row, col, height, detector) VALUES (?, ?, ?, ?, ?, ?)
        """, (filename, date, row, col, height, detector))
        self.dbconn.commit()

    def get_yht_points(self, filename):
        cursor = self.dbconn.cursor()
        cursor.execute("""
        SELECT * FROM yht_point WHERE filename = ?
        """, (filename,))
        return cursor.fetchall()

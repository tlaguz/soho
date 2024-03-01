from dataclasses import dataclass

@dataclass
class TrainingPointDto:
    id: int
    filename: str
    filename_prev: str
    filename_yht: str
    date: str
    date2: str
    detector: str
    naxis1: int
    naxis2: int
    observer: str
    aggregated_row: str
    aggregated_col: str
    aggregated_height: [float]

class TrainingPointsRepository:
    def __init__(self, dbconn):
        self.dbconn = dbconn

    @staticmethod
    def detector_to_table(detector):
        if detector == 'C2':
            return 'training_points_c2'
        elif detector == 'C3':
            return 'training_points_c3'
        else:
            return 'training_points'

    def get_training_data_len(self, detector):
        cursor = self.dbconn.cursor()
        cursor.execute(f"""
            select count(*)
            from {self.detector_to_table(detector)}
        """)
        return cursor.fetchone()[0]

    def get_training_data_by_id(self, id, detector):
        cursor = self.dbconn.cursor()
        cursor.execute(f"""
        select 
            id, 
            filename,
            filename_prev,
            date,
            date2, 
            detector, 
            naxis1, 
            naxis2, 
            filename_yht, 
            observer, 
            aggregated_row, 
            aggregated_col, 
            aggregated_height
        from {self.detector_to_table(detector)}
        where id = ?
        """, (id,))
        return cursor.fetchone()

    def get_training_data_by_filename(self, filename, detector):
        cursor = self.dbconn.cursor()
        cursor.execute(f"""
        select 
            id, 
            filename,
            filename_prev,
            date,
            date2, 
            detector, 
            naxis1, 
            naxis2, 
            filename_yht, 
            observer, 
            aggregated_row, 
            aggregated_col, 
            aggregated_height
        from {self.detector_to_table(detector)}
        where filename = ?
        """, (filename,))
        return cursor.fetchone()
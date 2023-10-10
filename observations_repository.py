
class observations_repository:

    def __init__(self, dbconn):
        self.dbconn = dbconn
        self.cursor = dbconn.cursor()

    def insert_observation(
        self,
        begin_date,
        calibrated,
        campaign_name,
        campaign_number,
        campaign_oid,
        detector_name,
        detector_name_description,
        end_date,
        file_format,
        file_name,
        file_path,
        file_size,
        instrument_name,
        observation_oid,
        observation_type,
        observatory_name,
        postcard,
        processing_level,
        processing_level_priority,
        science_objective,
        science_object_oid,
        science_slit_oid,
        study_name,
        study_oid,
        wavelength_range
    ):
        self.cursor.execute("""
INSERT INTO observations (
    begin_date                   ,
    calibrated                   ,
    campaign_name                ,
    campaign_number              ,
    campaign_oid                 ,
    detector_name                ,
    detector_name_description    ,
    end_date                     ,
    file_format                  ,
    file_name                    ,
    file_path                    ,
    file_size                    ,
    instrument_name              ,
    observation_oid              ,
    observation_type             ,
    observatory_name             ,
    postcard                     ,
    processing_level             ,
    processing_level_priority    ,
    science_objective            ,
    science_object_oid           ,
    science_slit_oid             ,
    study_name                   ,
    study_oid                    ,
    wavelength_range
) VALUES (
    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
)
        """, (
            begin_date,
            calibrated,
            campaign_name,
            campaign_number,
            campaign_oid,
            detector_name,
            detector_name_description,
            end_date,
            file_format,
            file_name,
            file_path,
            file_size,
            instrument_name,
            observation_oid,
            observation_type,
            observatory_name,
            postcard,
            processing_level,
            processing_level_priority,
            science_objective,
            science_object_oid,
            science_slit_oid,
            study_name,
            study_oid,
            wavelength_range
        ))

    def get_observation(self, oid):
        return self.cursor.execute("""
SELECT * FROM observations WHERE observation_oid = ?
        """, (oid,)).fetchone()

    def get_count(self):
        return self.cursor.execute("""
SELECT COUNT(*) FROM observations
        """).fetchone()[0]

    def get_observations(self, page):
        return self.cursor.execute("""
SELECT * FROM observations LIMIT 1000 OFFSET ?
        """, ((page - 1) * 1000,)).fetchall()
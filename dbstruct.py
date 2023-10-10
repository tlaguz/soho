
def create_db(dbconn):
    cursor = dbconn.cursor()
    cursor.execute("""
CREATE TABLE IF NOT EXISTS observations (
  begin_date                   TEXT    ,
  calibrated                   INTEGER ,
  campaign_name                TEXT    ,
  campaign_number              INTEGER ,
  campaign_oid                 INTEGER ,
  detector_name                TEXT    ,
  detector_name_description    TEXT    ,
  end_date                     TEXT    ,
  file_format                  TEXT    ,
  file_name                    TEXT    ,
  file_path                    TEXT    ,
  file_size                    INTEGER ,
  instrument_name              TEXT    ,
  observation_oid              INTEGER PRIMARY KEY ,
  observation_type             TEXT    ,
  observatory_name             TEXT    ,
  postcard                     INTEGER ,
  processing_level             TEXT    ,
  processing_level_priority    INTEGER ,
  science_objective            TEXT    ,
  science_object_oid           INTEGER ,
  science_slit_oid             INTEGER ,
  study_name                   TEXT    ,
  study_oid                    INTEGER ,
  wavelength_range             TEXT    
    )""")

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_observations_begin_date ON observations (begin_date);
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_observations_end_date ON observations (end_date);
    """)

    dbconn.commit()
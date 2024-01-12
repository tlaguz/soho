
def create_db(dbconn):
    cursor = dbconn.cursor()
    cursor.execute("""
CREATE TABLE IF NOT EXISTS cdaw_cme (
  occurence_time              TEXT    ,
  central_pa                  TEXT    ,
  angular_width               TEXT    ,
  linear_speed                REAL    ,
  second_order_speed_initial  REAL    ,
  second_order_speed_final    REAL    ,
  second_order_speed_20R      REAL    ,
  acceleration                TEXT    ,
  mass                        TEXT    ,
  kinetic_energy            TEXT    ,
  mpa                         REAL    ,
  remarks                     TEXT 
)""")

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_observations_occurence_time ON cdaw_cme (occurence_time);
    """)

    dbconn.commit()
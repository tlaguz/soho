class cdaw_cme_repository:

    def __init__(self, dbconn):
        self.dbconn = dbconn
        self.cursor = dbconn.cursor()

    def insert_cme_event(self, observation_data):
        self.cursor.execute("""
INSERT INTO cdaw_cme (
    occurence_time              ,
    central_pa                  ,
    angular_width               ,
    linear_speed                ,
    second_order_speed_initial  ,
    second_order_speed_final    ,
    second_order_speed_20R      ,
    acceleration                ,
    mass                        ,
    kinetic_energy            ,
    mpa                         ,
    remarks
    ) VALUES (
    ?,?,?,?,?,?,?,?,?,?,?,?
    )
        """, (
            observation_data['occurence_time'],
            observation_data['central_pa'],
            observation_data['angular_width'],
            observation_data['linear_speed'],
            observation_data['second_order_speed_initial'],
            observation_data['second_order_speed_final'],
            observation_data['second_order_speed_20R'],
            observation_data['acceleration'],
            observation_data['mass'],
            observation_data['kinetic_energy'],
            observation_data['mpa'],
            observation_data['remarks']
        ))

    def get_count(self):
        return self.cursor.execute("""
SELECT COUNT(*) FROM cdaw_cme
        """).fetchone()[0]
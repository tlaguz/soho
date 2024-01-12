import requests
import time
import os
import threading
from datetime import datetime

from cdaw import cdaw
from cdaw_cme_repository import cdaw_cme_repository

class observations_manager:
    def __init__(self, dbconn):
        self.dbconn = dbconn
        self.repo = cdaw_cme_repository(dbconn)
        self.cdaw = cdaw()

    def download_all_yhts(self):
        all_obs = []
        catalog = self.cdaw.get_catalog_by_months()
        for (year, month) in catalog:
            url = catalog[(year, month)]
            print("Downloading data for " + str(year) + "/" + str(month))
            obs = self.cdaw.get_observations_from_month_url(url)
            all_obs += obs
            pass

        # create directory if it doesn't exist ./downloads/yhts/
        if not os.path.exists("./downloads/yhts/"):
            os.makedirs("./downloads/yhts/")

        for obs in all_obs:
            filename = "./downloads/yhts/" + obs.rsplit('/', 1)[-1]
            if os.path.exists(filename): continue

            success = False
            while not success:
                try:
                    print("Downloading " + filename)
                    response = requests.get(obs, stream=True)
                    response.raise_for_status()

                    with open(filename, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)

                    success = True
                except Exception as e:
                    print("Error downloading " + filename + ":", e)
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                    pass


    def import_from_cdaw_to_db(self):
        univ_all = self.cdaw.get_univ_all()
        print("Total events: " + str(len(univ_all)))

        for record in univ_all:
            if record == "":
                continue

            fields = record.split()

            try:
                # combine date and time, then parse into datetime object
                datetime_str = fields[0] + " " + fields[1]
                datetime_obj = datetime.strptime(datetime_str, "%Y/%m/%d %H:%M:%S")

                # Convert datetime object to ISO8601 format
                occurence_time = datetime_obj.isoformat()

                central_pa = fields[2]
                angular_width = fields[3]
                linear_speed = fields[4]

                # Second order speed
                second_order_speed_initial = fields[5]
                second_order_speed_final = fields[6]
                second_order_speed_20R = fields[7]

                # Acceleration
                accel = fields[8]

                # Mass is parsed, checking for dashes to denote no value
                mass = fields[9]
                mass = None if "-" in mass else mass

                # Kinetic energy is parsed, checking for dashes to denote no value
                kinetic_energy = fields[10]
                kinetic_energy = None if "-" in kinetic_energy else kinetic_energy

                # Mass Position Angle (MPA) and Remarks
                mpa = fields[11]
                remarks = " ".join(fields[12:])

                observation_data = {
                    'occurence_time': occurence_time,
                    'central_pa': central_pa,
                    'angular_width': angular_width,
                    'linear_speed': linear_speed,
                    'second_order_speed_initial': second_order_speed_initial,
                    'second_order_speed_final': second_order_speed_final,
                    'second_order_speed_20R': second_order_speed_20R,
                    'acceleration': accel,
                    'mass': mass,
                    'kinetic_energy': kinetic_energy,
                    'mpa': mpa,
                    'remarks': remarks
                }

                self.repo.insert_cme_event(observation_data)
            except IndexError as e:
                print("Record is not in expected format or is incomplete: ", record)

        print("Finished importing data into the database.")
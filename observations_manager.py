import requests
import time
import os

from observations_repository import observations_repository
from ssa import ssa


class observations_manager:
    def __init__(self, dbconn):
        self.dbconn = dbconn
        self.repo = observations_repository(dbconn)
        self.ssa = ssa()

    def download_metadata(self):
        count = self.ssa.get_count()
        print("Total observations: " + str(count))
        pages = int(count / 1000) + 1
        print("Total pages: " + str(pages))
        for page in range(1, pages + 1):
            print("Downloading page " + str(page) + "...")
            observations = self.ssa.get_observations(page)
            for observation in observations:
                if self.repo.get_observation(observation[13]) is not None:
                    print("!!! Skipping " + str(observation[13]))
                    continue
                self.repo.insert_observation(
                    observation[0],
                    observation[1],
                    observation[2],
                    observation[3],
                    observation[4],
                    observation[5],
                    observation[6],
                    observation[7],
                    observation[8],
                    observation[9],
                    observation[10],
                    observation[11],
                    observation[12],
                    observation[13],
                    observation[14],
                    observation[15],
                    observation[16],
                    observation[17],
                    observation[18],
                    observation[19],
                    observation[20],
                    observation[21],
                    observation[22],
                    observation[23],
                    observation[24],
                )
            self.dbconn.commit()

    def make_path(self, path, filename):
        return './data/' + str(path) + "/" + filename

    def download_file(self, observation_oid, path, filename):
        f = self.ssa.get_file(observation_oid)

        if not os.path.exists('./data/' + path):
            os.makedirs('./data/' + path)

        open(self.make_path(path, filename), 'wb').write(f)

    def download_data(self):
        count = self.repo.get_count()
        print("Total observations: " + str(count))
        pages = int(count / 1000) + 1
        print("Total pages: " + str(pages))
        for page in range(1, pages + 1):
            print("Downloading page " + str(page) + "...")
            observations = self.repo.get_observations(page)
            for observation in observations:
                path = observation[10]
                filename = observation[9]
                oid = observation[13]

                if os.path.exists(self.make_path(path, filename)):
                    print("!!! Skipping " + str(observation[13]))
                    continue

                print("Downloading " + str(oid) + "...")

                self.download_file(oid, path, filename)

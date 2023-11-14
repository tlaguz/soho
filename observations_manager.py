import requests
import time
import os
import threading

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

        #print("Downloaded " + str(observation_oid) + " to " + self.make_path(path, filename))

    def download_data(self):
        count = self.repo.get_count()
        i = 0
        page_size = 20
        print("Total observations: " + str(count))
        pages = int(count / page_size) + 1
        print("Total pages: " + str(pages))
        for page in range(1, pages + 1):
            threads = []
            start = time.time()
            print("Downloading page " + str(page) + "...")
            observations = self.repo.get_observations(page, page_size)
            for observation in observations:
                i = i+1
                path = observation[10]
                filename = observation[9]
                oid = observation[13]

                if os.path.exists(self.make_path(path, filename)):
                    #print("["+str(i)+"/"+str(count)+" ; page "+str(page)+"/"+str(pages)+"] Skipping " + str(observation[13]))
                    continue

                #print("["+str(i)+"/"+str(count)+" ; page "+str(page)+"/"+str(pages)+"] Downloading " + str(oid) + "...")

                download_thread = threading.Thread(target=self.download_file, args=(oid, path, filename))
                download_thread.start()
                threads.append(download_thread)

            for thread in threads:
                thread.join()

            end = time.time()
            print("Page "+str(page)+"/"+str(pages)+" downloaded in " + str(end - start) + " seconds")
            print("Observations per second: " + str(page_size / (end - start)))
            print("Estimated time left: " + str((pages - page) * (end - start) / 60/60) + " hours")

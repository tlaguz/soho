from dataclasses import dataclass
from typing import List


@dataclass
class YhtObservation:
    height: str
    date: str
    time: str
    angle: str
    tel: str
    fc: str
    col: str
    row: str

@dataclass
class YhtFile:
    date_obs: str
    time_obs: str
    detector: str
    filter: str
    observer: str
    feat_code: str
    image_type: str
    yht_id: str
    orig_htfile: str
    orig_wdfile: str
    universal: str
    wdata: str
    halo: str
    onset1: str
    onset2: str
    onset2_rsun: str
    cen_pa: str
    width: str
    speed: str
    accel: str
    feat_pa: str
    feat_qual: str
    quality_index: str
    remark: str
    comment: str
    observations: List[YhtObservation]

class yht_reader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.read_file()

    def read_file(self):
        with open(self.file_path, 'r') as f:
            self.data = f.read()
            # parse the data
            self.parse_data()

    def parse_data(self):
        lines = self.data.split("\n")
        # find on which line there is DATE
        for i, line in enumerate(lines):
            if "DATE-OBS" in line:
                date_obs = line.split(":")[1].strip()
            if "TIME-OBS" in line:
                time_obs = line.split(":", 1)[1].strip()
            if "DETECTOR" in line:
                detector = line.split(":")[1].strip()
            if "FILTER" in line:
                filter = line.split(":")[1].strip()
            if "OBSERVER" in line:
                observer = line.split(":")[1].strip()
            if "FEAT_CODE" in line:
                feat_code = line.split(":")[1].strip()
            if "IMAGE_TYPE" in line:
                image_type = line.split(":")[1].strip()
            if "YHT_ID" in line:
                yht_id = line.split(":")[1].strip()
            if "ORIG_HTFILE" in line:
                orig_htfile = line.split(":")[1].strip()
            if "ORIG_WDFILE" in line:
                orig_wdfile = line.split(":")[1].strip()
            if "UNIVERSAL" in line:
                universal = line.split(":")[1].strip()
            if "WDATA" in line:
                wdata = line.split(":")[1].strip()
            if "HALO" in line:
                halo = line.split(":")[1].strip()
            if "ONSET1" in line:
                onset1 = line.split(":")[1].strip()
            if "ONSET2" in line:
                onset2 = line.split(":")[1].strip()
            if "ONSET2_RSUN" in line:
                onset2_rsun = line.split(":")[1].strip()
            if "CEN_PA" in line:
                cen_pa = line.split(":")[1].strip()
            if "WIDTH" in line:
                width = line.split(":")[1].strip()
            if "SPEED" in line:
                speed = line.split(":")[1].strip()
            if "ACCEL" in line:
                accel = line.split(":")[1].strip()
            if "FEAT_PA" in line:
                feat_pa = line.split(":")[1].strip()
            if "FEAT_QUAL" in line:
                feat_qual = line.split(":")[1].strip()
            if "QUALITY_INDEX" in line:
                quality_index = line.split(":")[1].strip()
            if "REMARK" in line:
                remark = line.split(":")[1].strip()
            if "COMMENT" in line:
                comment = line.split(":")[1].strip()
            if "HEIGHT" in line:
                observations = []
                for j in range(i+1, len(lines)):
                    if lines[j] == "":
                        continue
                    height, date, time, angle, tel, fc, col, row = lines[j].split()
                    observations.append(YhtObservation(height, date, time, angle, tel, fc, col, row))
                break

        self.file = YhtFile(date_obs, time_obs, detector, filter, observer, feat_code, image_type, yht_id, orig_htfile, orig_wdfile, universal, wdata, halo, onset1, onset2, onset2_rsun, cen_pa, width, speed, accel, feat_pa, feat_qual, quality_index, remark, comment, observations)


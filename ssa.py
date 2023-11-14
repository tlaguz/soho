# https://ssa.esac.esa.int/ssa-sl-tap/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=json&QUERY=select count(*) from (SELECT * FROM soho.mv_observation observation WHERE observation.end_date >= '1992-01-01 00:00:00' AND observation.begin_date <= '2025-01-01 23:59:59' AND observation.instrument_name='LASCO' order by observation.processing_level_priority asc, observation.begin_date desc) as counter&TAPCLIENT=angular
#
# https://ssa.esac.esa.int/ssa-sl-tap/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=json&QUERY=SELECT * from soho.mv_observation observation WHERE observation.end_date >= '1992-01-01 00:00:00' AND observation.begin_date <= '2025-01-01 23:59:59' AND observation.instrument_name='LASCO' order by observation.processing_level_priority asc, observation.begin_date desc&TAPCLIENT=angular&PAGE=1&PAGE_SIZE=1
#
# https://ssa.esac.esa.int/ssa-sl-tap/data?retrieval_type=OBSERVATION&QUERY=WHERE observation.observation_oid IN (190628945)&compress=false&TAPCLIENT=angular

import requests

class ssa:

    def get_count(self):
        url = "https://ssa.esac.esa.int/ssa-sl-tap/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=json&QUERY=\
select count(*) from \
(SELECT * FROM soho.mv_observation observation \
WHERE observation.end_date >= '1992-01-01 00:00:00' AND observation.begin_date <= '2025-01-01 23:59:59' \
AND observation.instrument_name='LASCO' order by observation.processing_level_priority asc, observation.begin_date desc)\
as counter"
        return requests.get(url).json()['data'][0][0]

    def get_observations(self, page):
        url = "https://ssa.esac.esa.int/ssa-sl-tap/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=json&QUERY=\
SELECT * from soho.mv_observation observation \
WHERE observation.end_date >= '1992-01-01 00:00:00' AND observation.begin_date <= '2025-01-01 23:59:59' \
AND observation.instrument_name='LASCO' order by observation.begin_date desc&PAGE="+str(page)+"&PAGE_SIZE=1000"
        return requests.get(url).json()['data']


    def get_file(self, oid):
        url = "https://ssa.esac.esa.int/ssa-sl-tap/data?retrieval_type=OBSERVATION&QUERY=\
WHERE observation.observation_oid="+str(oid)+"&compress=false"
        while True:
            try:
                return requests.get(url).content
            except:
                print("Retrying...")
                continue

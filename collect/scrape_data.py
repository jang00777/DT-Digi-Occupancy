import sys
import urllib2
import time
import csv
import simplejson as json
import numpy as np
from DQMX509 import X509

IDENT = "DQMToJson/1.0 python/%d.%d.%d" % sys.version_info[:3]
SERVER = 'https://cmsweb.cern.ch/dqm/online'
GOLDEN_RUNS = [273158, 273730, 274388, 274422, 274968, 274969, 275310, 275311,
               275847, 275890, 276244, 276283, 276384, 276587, 276775, 276776,
               276950, 278509, 278820, 278822, 279694, 279766, 279794, 280018,
               281693, 281727, 281976, 282735, 282814, 276582]
COLLISION_RUNS = [272011, 272012, 272014, 272017, 272021, 272774, 284044,
                  284043, 284042, 284041]
SESSION = ''


def dqm_request(url, x509):
    datareq = urllib2.Request(url)
    datareq.add_header('User-agent', IDENT)
    return eval(urllib2.build_opener(x509).open(datareq).read(),
                {"__builtins__": None}, {})


def get_observation_json(run, wheel, sector, station, X509):
    wheel_string = 'Wheel%s' % wheel
    url = '%s/jsonfairy/archive/%s/Global/Online/ALL/DT/01-Digi/%s/Sector%s/Station%s/OccupancyAllHits_perCh_W%s_St%s_Sec%s' % (SERVER, run, wheel_string, sector, station, wheel, station, sector)
    return dqm_request(url, X509)


def get_lumi(run, x509):
    url = "https://cmsweb.cern.ch/dqm/online/session/%s/chooseSample" % SESSION
    response = dqm_request(url, x509)
    for r in response[1]['items'][1]['items']:
        if r['run'] == str(run):
            got_it = r
            break
    url = "https://cmsweb.cern.ch/dqm/online/session/%s/select?type=online_data;dataset=/Global/Online/ALL;runnr=%s;importversion=%s" % (SESSION, run, got_it['importversion'])
    response = dqm_request(url, x509)
    return response[1]['lumi'].replace("'", "")


if __name__ == "__main__":
    SESSION = sys.argv[1]
    print("Using %s session" % SESSION)

    x509 = X509.x509()

    runs = GOLDEN_RUNS + COLLISION_RUNS
    wheels = range(-2, 3)
    sectors = range(1, 15)
    stations = range(1, 5)

    for run in runs:
        run_data = []
        lumisections = get_lumi(run, x509)
        if lumisections is None:
            continue

        for wheel in wheels:
            for sector in sectors:
                for station in stations:
                    if sector > 12 and station < 4:
                        continue

                    print("Scraping RUN %s, Wheel: %s, Station: %s, Sector: %s"
                          % (run, wheel, station, sector))
                    time.sleep(2)
                    raw = get_observation_json(run, wheel, sector,
                                               station, x509)["hist"]

                    if not "bins" in raw:
                        continue

                    raw = raw["bins"]["content"]
                    content = np.array(raw)
                    for layer in range(0, 12):
                        data = {}

                        data_layer = [float(i) for i in content[layer]]
                        data_layer = [i for i in data_layer if i != -1]

                        if not len(data_layer):
                            continue

                        data['lumi'] = str(lumisections)
                        data['run'] = str(run)
                        data['wheel'] = str(wheel)
                        data['sector'] = str(sector)
                        data['station'] = str(station)
                        data['layer'] = str(layer + 1)
                        data['content'] = str(data_layer)
                        run_data.append(data)

        with open(('../data/%s.json' % run), 'wb') as file_:
            file_.write(json.dumps(run_data))

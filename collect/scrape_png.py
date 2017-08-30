import sys
import urllib2
import time
from DQMX509 import X509

SERVER = 'https://cmsweb.cern.ch/dqm/online'
GOLDEN_RUNS = [273158, 273730, 274388, 274422, 274968, 274969, 275310, 275311,
               275847, 275890, 276244, 276283, 276384, 276587, 276775, 276776,
               276950, 278509, 278820, 278822, 279694, 279766, 279794, 280018,
               281693, 281727, 281976, 282735, 282814, 276582]
COLLISION_RUNS = [272011, 272012, 272014, 272017, 272021, 272774, 284044,
                  284043, 284042, 284041]
SESSION = ''


def dqm_request(url, X509):
    res = urllib2.build_opener(X509).open(urllib2.Request(url)).read()
    return res


def get_png(run, wheel, sector, station, X509):
    url = "%s/plotfairy/archive/%s/Global/Online/ALL/DT/01-Digi/Wheel%s/Sector%s/Station%s/OccupancyAllHits_perCh_W%s_St%s_Sec%s?session=%s;w=1426;h=718" % (SERVER, run, wheel, sector, station, wheel, station, sector, SESSION)
    return dqm_request(url, X509)

if __name__ == "__main__":
    SESSION = sys.argv[1]
    print("Using %s session" % SESSION)
    x509 = X509.x509()

    runs = GOLDEN_RUNS + COLLISION_RUNS
    wheels = range(-2, 3)
    sectors = range(1, 15)
    stations = range(1, 5)

    for run in runs:
        print "Now Run: ", run
        for wheel in wheels:
            for sector in sectors:
                for station in stations:
                    if sector > 12 and station < 4:
                        continue
                    time.sleep(2)
                    name = "%s_%s_%s_%s" % (run, wheel, sector, station)
                    png = get_png(run, wheel, sector, station, x509)
                    with open(('../static/images/%s.png' % name), 'wb') as file_:
                        file_.write(png)

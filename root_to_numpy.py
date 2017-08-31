import ROOT as root
import os


def get_numpy_from_root(run, lumi, wheel):

    data = []

    s = 'ls00000000'
    lumi = str(lumi)
    lumi = s[::-1].replace('0'*len(lumi), lumi[::-1], 1)[::-1]

    wheel = int(wheel)
    run = str(run)

    for _, _, filenames in os.walk('data'):
        filenames = [f for f in filenames if ('R000' + run) in f]
        filenames = [f for f in filenames if lumi in f]
        if len(filenames) == 0:
            raise Exception
        _file = 'data/' + filenames[0]

    for station in range(1, 5):
        for sector in range(1, 15):
            if sector > 12 and station != 4:
                continue
            directory = ('DQMData/Run %s/DT/Run summary/01-Digi/Wheel%s/Sector%s/Station%s/' %
                        (run, wheel, sector, station))
            histogram = ('OccupancyAllHits_perCh_W%s_St%s_Sec%s' %
                        (wheel, station, sector))

            f = root.TFile(_file)
            d = f.GetDirectory(directory)
            h = d.Get(histogram)

            dimx = h.GetNbinsX()
            dimy = h.GetNbinsY()

            contents = []
            for y in range(1, dimy + 1):
                layer = []
                for x in range(1, dimx + 1):
                    layer.append(h.GetBinContent(x, y))
                contents.append(layer)

            data.append({'id': 'w' + str(wheel) + 's' + str(sector) +
                         'st' + str(station), 'content': contents})

    return data

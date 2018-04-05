import ROOT
import copy
import numpy as np
import simplejson as json
import pprint
import os

def readROOT(path,target,json_name):
    f = ROOT.TFile(path)
    tree = f
    for i in range(len(target)-1):
        tree = tree.Get(target[i])
    
    wheels=[(-1,"_r-1"),(1,"_r1")]
    sectors=[0]
    stations=[(1,"_st1")]
    #stations=[(1,"_st1"),(2,"_st2")]
    chambers=[range(0,36),range(0,18)]
    run_data = []
    for wheel in wheels:
        target_wheel = target[-1]+wheel[1]
        for sector in sectors:
            for station in stations:
                target_hist = target_wheel+station[1]
                t = tree.Get(target_hist)
                print target_hist
                for layer in [1,2]:
                    #shape = (len(chambers[station[0]-1]),t.GetYaxis().GetNbins())
                    shape = (73,12)
                    data_layer = np.array(np.zeros(shape)).astype(int)
                    for chamber in chambers[station[0]-1]:
                        for j in range(t.GetYaxis().GetNbins()):
                            data_layer[chamber,j] = t.GetBinContent(chamber*2+layer-1,j)
                    data={}
                    data['lumi']=str(1829)
                    data['run']=str(1)
                    data['wheel']=str(wheel[0])
                    data['sector']=str(sector)
                    data['station']=str(station[0])
                    data['layer']=str(layer+1)
                    data['content']=str(list(data_layer)).replace("array(","").replace(")","")
    run_data.append(data)
    print data['content']
    with open(("../data/%s.json" % json_name),"wb") as file_:
        file_.write(json.dumps(run_data))

target_digi = ["DQMData","Run 1","MuonGEMDigisV","Run summary","GEMDigisTask"]
target_rechit = ["DQMData","Run 1","MuonGEMRecHitsV","Run summary","GEMRecHitsTask"]
target = copy.copy(target_rechit)
target.append("rh_dcEta")
for i in range (0,301):
    file_name = './input/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO000'+str(i).zfill(3)+'.root'
    if os.path.isfile(file_name): readROOT(file_name,target,str(i))

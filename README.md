Online DQM with Machine Learning: Drift Tubes Occupancy Tests

To run it yourself:

- To collect the data run:

```
$ cd collect
$ python scrape_data.py <SESSION>
```
This assumes you have your `.pem` files in `~/.globus` and you can access DQMGUI. To get the `SESSION` parameter, copy it from DQMGUI URL (`.../online/session/SESSION`)

- To label some data:
```
$ mkdir -p static/images
$ cd collect
$ python scrape_png.py <SESSION>
$ cd ..
$ python label.py
```
The `label` site runs on Flask. Navigate to `<Your URL>/label` to start scoring the data.

- To train the model use the `model_training.ipynb` notebook.

- To verify stability of the results run:
```
$ ssh lxplus
$ ssh cmsusr
$ scp /globalscratch/dqm4ml_production/*<RUN>.root.ls* <PATH_TO_REPO>/data/
```
This will copy the ROOT files with the occupancy data stored every 10 lumisections. Next, generate csv `python stability.py <RUN>`. You can run `stability.ipynb` now.



If you are happy with the performance, try plugging it in to the DQM GUI. To generate DQM ROOT file use the steps from the `example`. Substitute `INPUT` in with the list of files you can get from DAS, for example `https://cmsweb.cern.ch/dbs/prod/global/DBSReader/files?dataset=/DoubleMuon/Run2016H-v1/RAW&run_num=284068`. You will need to clone the `https://gitlab.cern.ch/mrieger/CMSSW-DNN` repository to your `src`and apply last commit from `https://github.com/AdrianAlan/cmssw/tree/feature/cnn-for-dt-occupancy`.

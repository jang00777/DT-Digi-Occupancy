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

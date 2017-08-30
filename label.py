from flask import Flask, render_template, url_for, redirect, request
import pandas as pd
import random

app = Flask(__name__)
app.url_map.strict_slashes = False

RUNS = [273158, 273730, 274388, 274422, 274968, 274969, 275310, 275311, 275847,
        275890, 276244, 276283, 276384, 276587, 276775, 276776, 276950, 278509,
        278820, 278822, 279694, 279766, 279794, 280018, 281693, 281727, 281976,
        282735, 282814, 276582, 272011, 272012, 272014, 272017, 272021, 272774,
        284044, 284043, 284042, 284041]


@app.route('/label/')
def label():
    # Draw random chamber
    wheel = random.randint(-2, 2)
    station = random.randint(1, 4)
    if station == 4:
        sec_max = 14
    else:
        sec_max = 12
    sector = random.randint(1, sec_max)
    run = RUNS[random.randint(0, len(RUNS)-1)]

    # Check if not scored already
    labels = pd.read_csv('data/labels.csv', names=['wheel', 'station', 'sector', 'run', 'layer', 'score'])
    already = labels[(labels.run == run) & (labels.sector == sector) &
                     (labels.wheel == wheel) & (labels.station == station)]
    if len(already):
        return redirect(url_for('label'))

    # Get the image
    id_name = (str(run) + "_" + str(wheel) + "_" + str(sector) + "_" + str(station))
    return render_template('label.html', run=run, imgsrc=url_for('static', filename='images/' + id_name + '.png'), id=id_name)


@app.route('/result/')
def result():
    pos = request.args['idplot'].split("_")

    layers = []
    for i in range(12):
        if 'l' + str(i+1) in request.args:
            layers.append((request.args['l' + str(i+1)] == "on") + 0)
        else:
            layers.append(0)

    foo = ''
    for x, l in enumerate(layers):
        csv = pos[1] + "," + pos[3] + "," + pos[2] + "," + pos[0]
        csv += "," + str(x + 1) + "," + str(l)
        foo += "\n" + csv

    with open('data/labels.csv', 'ab') as file_:
        file_.write(foo)

    return redirect(url_for('label'))

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=80)

import root_to_numpy as rtn
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
import math
import tensorflow as tf
import sys
from scipy import misc

RUN = None
SAMPLE_SIZE = 47

def resize_occupancy(layer):
    return misc.imresize(np.array(layer).reshape(1, -1), (1, SAMPLE_SIZE), interp='bilinear', mode="F")

def scale_occupancy(layer):
    layer = layer.reshape(-1, 1)
    scaler = MaxAbsScaler().fit(layer)
    return scaler.transform(layer).reshape(1, -1)

def preprocess(data_layer):
    data_layer = [i for i in data_layer if i != -1]
    if not len(data_layer):
        return data_layer

    data_layer = resize_occupancy(data_layer)
    data_layer = scale_occupancy(data_layer)
    return data_layer

def get_layer_stack(data):
    layer_stack = np.zeros(47)

    for chamber_data in data:
        chamber_data = chamber_data['content']
        for layer in chamber_data:
            layer = preprocess(layer)
            if len(layer):
                layer_stack = np.vstack([layer_stack, layer])

    return layer_stack


def get_current_dqm_alarms(data):
    total_bad = 0

    for chamber_data in data:
        chamber_data = chamber_data["content"]
        chamber_data = np.array(chamber_data)
        score = float(len(np.where(chamber_data == 0)[0])) / len(np.where(chamber_data != -1)[0])
        if score > 0.25:
            total_bad = total_bad + 1

    return total_bad

if __name__ == "__main__":
    RUN = sys.argv[1]
    CSV_NAME = ("data/%sST.csv" % RUN)
    sess = tf.Session()
    tf.saved_model.loader.load(sess,
                              [tf.saved_model.tag_constants.SERVING],
                              "model")
    _input = tf.get_default_graph().get_tensor_by_name('input_3:0')
    output = tf.get_default_graph().get_tensor_by_name('output_1/Softmax:0')

    with open(CSV_NAME, 'w') as file_:
        file_.write('lumi,run,total,emerging,current_dqm')
        file_.write("\n")

    p_faults = np.zeros(2721)

    for lumi in range(10, 400, 10):
        raw_data = np.array([])
        for wheel in range(-2, 3):
            try:
                raw_data = np.append(raw_data,
                                     rtn.get_numpy_from_root(RUN, lumi, wheel),
                                     axis=0)
            except:
                sys.exit(0)
        current_dqm = get_current_dqm_alarms(raw_data)
        layer_stack = get_layer_stack(raw_data).reshape(-1, SAMPLE_SIZE, 1)

        prediction = sess.run(output, {_input: layer_stack})
        faults = np.argmax(prediction, axis=1)
        emerging = sum(faults - p_faults == 1)
        p_faults = faults

        with open(CSV_NAME, 'a') as file_:
            file_.write("%s,%s,%s,%s,%s\n" % (lumi, RUN, sum(faults), emerging, current_dqm))

        print("Tested lumisection %d" % lumi)

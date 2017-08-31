import root_to_numpy as rtn
import numpy as np
import math
import tensorflow as tf
import sys


CSV_NAME = "data/stability.csv"
RUN = None


def smooth_reshape_occupancy(layer):
    target_size = 47
    Y = []
    stop = 0
    step = len(layer)/float(target_size)

    for i in range(target_size):
        hook = math.floor(stop)
        stop = step + stop
        mean = np.median(layer[int(hook):int(math.ceil(stop))])
        Y.append(mean)
    return Y


def scale_occupancy(layer):
    denominator = np.max(layer)
    if denominator:
        return layer/denominator
    return layer


def preprocess(data_layer):
    data_layer = [i for i in data_layer if i != -1]
    if not len(data_layer):
        return data_layer

    data_layer = smooth_reshape_occupancy(data_layer)
    data_layer = np.array(scale_occupancy(data_layer))
    return data_layer


def get_layer_stack(data):
    layer_stack = np.zeros(47)

    for chamber_data in raw_data:
        chamber_data = chamber_data['content']

        for layer in chamber_data:
            postlayer = preprocess(layer)
            if len(postlayer):
                layer_stack = np.vstack([layer_stack, postlayer])

    return layer_stack


if __name__ == "__main__":
    RUN = sys.argv[1]
    sess = tf.Session()
    tf.saved_model.loader.load(sess,
                              [tf.saved_model.tag_constants.SERVING],
                              "model")
    _input = tf.get_default_graph().get_tensor_by_name('input:0')
    output = tf.get_default_graph().get_tensor_by_name('output/Softmax:0')

    with open(CSV_NAME, 'w') as file_:
        file_.write('lumi,total,emerging')
        file_.write("\n")

    p_faults = np.zeros(2721)
    for lumi in range(10, 1000, 10):
        raw_data = np.array([])
        for wheel in range(-2, 3):
            try:
                raw_data = np.append(raw_data,
                                     rtn.get_numpy_from_root(RUN, lumi, wheel),
                                     axis=0)
            except:
                sys.exit(0)

        layer_stack = get_layer_stack(raw_data)
        prediction = sess.run(output, {_input: layer_stack})
        faults = np.argmax(prediction, axis=1)
        emerging = sum(faults - p_faults == 1)
        p_faults = faults

        with open(CSV_NAME, 'a') as file_:
            file_.write("%s,%s,%s\n" % (lumi, sum(faults), emerging))
        print("Tested lumisection %d" % lumi)

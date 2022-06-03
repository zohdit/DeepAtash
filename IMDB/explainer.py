# import tensorflow as tf
# import numpy as np
import json
from tensorflow.keras.datasets import imdb
from alibi.explainers import IntegratedGradients
# from properties import MODEL, VOCAB_SIZE
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import pickle
from datasets import load_dataset

import numpy as np
# For Python 3.6 we use the base keras
import tensorflow as tf
# import tensorflow_hub as hub

import pickle
from config import MODEL
from tensorflow.keras.preprocessing.sequence import pad_sequences

from imdb_michael import MyTokenAndPositionEmbedding
from config import VOCAB_SIZE


model = tf.keras.models.load_model(MODEL)

layer = model.layers[2]

index = imdb.get_word_index()
reverse_index = {value: key for (key, value) in index.items()}

def decode_sentence(x, reverse_index):
    # the `-3` offset is due to the special tokens used by keras
    # see https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset
    return " ".join([reverse_index.get(i - 3, 'UNK') for i in x])


max_features = 10000
maxlen = 2000

DATASET_DIR = "data"
test_ds = load_dataset('imdb', 

cache_dir=f"{DATASET_DIR}/imdb", split='test')
x_test, y_test = test_ds['text'], test_ds['label']

# index = imdb.get_word_index()
# reverse_index = {value: key for (key, value) in index.items()}


n_steps = 50
method = "gausslegendre"
internal_batch_size = 100
ig  = IntegratedGradients(model,
                          layer=layer,
                          n_steps=n_steps,
                          method=method,
                          internal_batch_size=internal_batch_size)


# loading
with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def explain_integrated_gradiant(X):
    #Predictions vector
    seq = tokenizer.texts_to_sequences([X])
    padded_texts = pad_sequences(seq, maxlen=VOCAB_SIZE)
    predictions = model.predict(padded_texts).argmax(axis=1)


    explanation = ig.explain(padded_texts,
                            baselines=None,
                            target=predictions,
                            attribute_to_layer_inputs=False)

    # Get attributions values from the explanation object
    attrs = explanation.attributions[0]
    attrs = attrs.sum(axis=2)
    expl = attrs[0]
    expl[expl<0] = 0
    return expl


from IPython.display import HTML
def  hlstr(string, color='white'):
    """
    Return HTML markup highlighting text with the desired color.
    """
    return f"<mark style=background-color:{color}>{string} </mark>"



def colorize(attrs, cmap='PiYG'):
    """
    Compute hex colors based on the attributions for a single instance.
    Uses a diverging colorscale by default and normalizes and scales
    the colormap so that colors are consistent with the attributions.
    """
    import matplotlib as mpl
    cmap_bound = attrs.max()
    norm = mpl.colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
    cmap = mpl.cm.get_cmap(cmap)

    # now compute hex values of colors
    colors = list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))
    return colors


if __name__ == "__main__":
    json_data_file = "../../DeepHyperion-IMDB/logs/DH-CS/VerbCount_NegCount/RUN_6_1000_VerbCount-NegCount_3600_20220519083119_346/archive/mbr8988.json"
    with open(json_data_file, 'r') as input_file:
            # Get the JSON
            map_dict = json.load(input_file)
    x_i = map_dict["text"]
    attrs_i = explain_integrated_gradiant(x_i)

    # words = decode_sentence(x_i, reverse_index).split()
    words = x_i.split()
    colors = colorize(attrs_i)

    _data = HTML("".join(list(map(hlstr, words, colors))))

    with open("data1.html", "w") as file:
        file.write(_data.data)
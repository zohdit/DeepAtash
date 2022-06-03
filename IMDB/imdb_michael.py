
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from datasets import load_dataset

CASE_STUDY = "imdb"

from config import VOCAB_SIZE

INPUT_MAXLEN = 2000


SA_ACTIVATION_LAYERS = [5]

NC_ACTIVATION_LAYERS = [
    (1, lambda x: x.token_emb),  # Embedding layers
    (1, lambda x: x.pos_emb),  # Embedding layers
    (2, lambda x: x.ffn[0]),  # Dense feed forward layers in transformer
    (2, lambda x: x.ffn[1]),  # Dense feed forward layers in transformer
    3, 5  # Dense layers in classifier
]

BADGE_SIZE = 128


# Taken from https://keras.io/examples/nlp/text_classification_with_transformer/
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim), ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.embed_dim, self.num_heads, self.ff_dim, self.rate = embed_dim, num_heads, ff_dim, rate

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config["embed_dim"] = self.embed_dim
        config["num_heads"] = self.num_heads
        config["ff_dim"] = self.ff_dim
        config["rate"] = self.rate
        config["att"] = self.att.get_config()
        config["ffn"] = self.ffn.get_config()
        config["layernorm1"] = self.layernorm1.get_config()
        config["layernorm2"] = self.layernorm2.get_config()
        config["dropout1"] = self.dropout1.get_config()
        config["dropout2"] = self.dropout2.get_config()
        return config

    @classmethod
    def from_config(cls, config):
        instance = cls(config["embed_dim"], config["num_heads"], config["ff_dim"], config["rate"])
        instance.att = tf.keras.layers.MultiHeadAttention.from_config(config["att"])
        instance.ffn = tf.keras.Sequential.from_config(config["ffn"])
        instance.layernorm1 = tf.keras.layers.LayerNormalization.from_config(config["layernorm1"])
        instance.layernorm2 = tf.keras.layers.LayerNormalization.from_config(config["layernorm2"])
        instance.dropout1 = tf.keras.layers.Dropout.from_config(config["dropout1"])
        instance.dropout2 = tf.keras.layers.Dropout.from_config(config["dropout2"])
        return instance


# Taken from https://keras.io/examples/nlp/text_classification_with_transformer/
@tf.keras.utils.register_keras_serializable()
class MyTokenAndPositionEmbedding(tf.keras.layers.Layer):

    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        # super(MyTokenAndPositionEmbedding, self).__init__()
        super(MyTokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen, self.vocab_size, self.embed_dim = maxlen, vocab_size, embed_dim

    def build(self, input_shape):
        self.token_emb = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        return {'maxlen': self.maxlen,
                'vocab_size': self.vocab_size,
                'embed_dim': self.embed_dim}




DATASET_DIR = "data"
if __name__ == '__main__':
    # prepare data
    train_ds = load_dataset('imdb', cache_dir=f"{DATASET_DIR}/imdb", split='train')
    x_train, y_train = train_ds['text'], train_ds['label']

    test_ds = load_dataset('imdb', cache_dir=f"{DATASET_DIR}/imdb", split='test')
    x_test, y_test = test_ds['text'], test_ds['label']
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(x_train)


    import pickle

    # saving
    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)


    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=INPUT_MAXLEN)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=INPUT_MAXLEN)


    if not os.path.exists(f"{DATASET_DIR}/imdb-cached"):
        os.makedirs(f"{DATASET_DIR}/imdb-cached")
    np.save(f"{DATASET_DIR}/imdb-cached/x_train.npy", x_train)
    np.save(f"{DATASET_DIR}/imdb-cached/y_train.npy", y_train)
    np.save(f"{DATASET_DIR}/imdb-cached/x_test.npy", x_test)
    np.save(f"{DATASET_DIR}/imdb-cached/y_test.npy", y_test)


    x_train = np.load(f"{DATASET_DIR}/imdb-cached/x_train.npy")
    y_train = np.load(f"{DATASET_DIR}/imdb-cached/y_train.npy")
    x_test = np.load(f"{DATASET_DIR}/imdb-cached/x_test.npy")
    y_test = np.load(f"{DATASET_DIR}/imdb-cached/y_test.npy")

    y_train = tf.keras.utils.to_categorical(y_train)


    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = tf.keras.layers.Input(shape=(INPUT_MAXLEN,))
    embedding_layer = MyTokenAndPositionEmbedding(INPUT_MAXLEN, VOCAB_SIZE, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(20, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    # As opposed to the keras tutorial, we use categorical_crossentropy,
    #   and we run 10 instead of 2 epochs, but with early stopping.
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        x_train, y_train, batch_size=32, epochs=10, validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    )
    model.save("models/imdb_michael_new.h5")


    # accuracy: 0.8834 - val_loss: 0.6041
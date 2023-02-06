
import tensorflow as tf
from config import MODEL
from alibi.explainers import IntegratedGradients

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


model = tf.keras.models.load_model(MODEL)


# Initialize IntegratedGradients instance
n_steps = 50
method = "gausslegendre"
ig  = IntegratedGradients(model,
                          n_steps=n_steps,
                          method=method)


def explain_integrated_gradiant(X):
    predictions = model(X).numpy().argmax(axis=1)
    explanation = ig.explain([X],
                            baselines=None,
                            target=predictions)

    attrs = explanation.attributions[0]
    
    attr = attrs[0]
    # positive attributions
    attr_pos = attr.clip(0, 1)
    im_pos = attr_pos.squeeze()   

    return im_pos    






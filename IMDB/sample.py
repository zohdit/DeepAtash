
import json
from os.path import join
import numpy as np
from explainer import explain_integrated_gradiant

from config import XAI_METHOD


class Sample:
    COUNT = 0

    def __init__(self, text, label, seed):
        self.id = Sample.COUNT
        self.seed = seed
        self.features = {}
        self.text = text
        self.expected_label = label
        self.predicted_label = None
        self.confidence = None
        self.ff = None
        self.distance_to_target = np.inf
        self.sparseness = np.inf
        self.coordinate = None
        self.latent_vector = None
        self.explanation = None
        Sample.COUNT += 1

    def to_dict(self):
        return {'id': str(self.id),
                'seed': str(self.seed),
                'expected_label': str(self.expected_label),
                'predicted_label': str(self.predicted_label),
                'misbehaviour': self.is_misbehavior(),
                'performance': str(self.ff),
                'features': self.features,
                'distance to target': str(self.distance_to_target),
                'sparseness': str(self.sparseness),
                'text': self.text,
    }

    def compute_explanation(self):
        if XAI_METHOD == "IG":
            self.explanation = explain_integrated_gradiant(self.purified)
        # elif XAI_METHOD == "CEM":
        #     self.explanation = explain_cem(self.purified)


    def compute_latent_vector(self, encoder):
        mean, _, _ = encoder.predict(self.purified)
        self.latent_vector = mean


    def from_dict(self, the_dict):
        for k in self.__dict__.keys():
            if k in the_dict.keys():
                setattr(self, k, the_dict[k])
        return self

    def dump(self, filename):
        data = self.to_dict()
        filedest = filename+".json"
        with open(filedest, 'w') as f:
            (json.dump(data, f, sort_keys=True, indent=4))

    def save_txt(self, filename):
        data = self.text
        filedest = filename + ".txt"
        with open(filedest, 'w') as f:
            f.write(data)


    def is_misbehavior(self):
        if str(self.expected_label) == str(self.predicted_label):
            return False
        else:
            return True

    def export(self, dst):
        dst = join(dst, "mbr"+str(self.id))
        self.dump(dst)


    def clone(self):
        clone_text = Sample(self.text, self.expected_label, self.seed)
        return clone_text
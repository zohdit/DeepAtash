import random
import mutation_manager
from utils import levenshtein



class TextMutator:


    def __init__(self, text):
        self.text = text

    def mutate(self):
        condition = True
        counter_mutations = 0
        while condition:
            # Select mutation operator.
            mutation = random.choice([1,2,3,4,5])

            counter_mutations += 1

            mutant_vector = mutation_manager.mutate(self.text.text, mutation)

            distance_inputs = levenshtein(self.text.text, mutant_vector)

            if distance_inputs != 0:
                condition = False


        self.text.text = mutant_vector
        self.text.predicted_label = None
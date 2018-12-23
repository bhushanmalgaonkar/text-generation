#!/usr/bin/env python3

from models.probabilistic import Probabilistic

model = Probabilistic()
model.fit_from('data/harry_potter.txt')

generated_text = model.generate_text()
print(generated_text)

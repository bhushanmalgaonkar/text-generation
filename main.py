#!/usr/bin/env python3

from models.probabilistic import Probabilistic

model = Probabilistic()
model.fit_from('data/harry_potter.txt')

generated_text = model.generate_text(max_len=100, seed1='you', seed2='knew')
print(generated_text)  # "you knew what was coming to a fistful of tubers had hit him"

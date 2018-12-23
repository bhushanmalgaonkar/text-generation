#!/usr/bin/env python3

from models.probabilistic import Probabilistic


with open('data/comments.txt', encoding='utf8', errors='ignore') as f:
    text = ' endofline '.join(line.strip() for line in f.readlines() if len(line) > 0)

model = Probabilistic()
model.fit(text)

MAX_LEN = 100
for i in range(20):
    text = ['endofline']
    for i in range(MAX_LEN):
        if len(text) > 1:
            next = model.next(text[-1], text[-2])
        else:
            next = model.next(text[-1])
        if next == 'endofline':
            print(' '.join(text[1:]))
            break
        text.append(next)
    else:
        print('out of for', ' '.join(text[1:]))

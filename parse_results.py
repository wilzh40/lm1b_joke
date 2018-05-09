import json
from collections import Counter
import os

cats = []
jsons = []
i = 0
for filename in os.listdir('./humor_results/'):
    with open('./humor_results/' + filename) as result:
        for line in result:
            if i % 2 == 0:
                cats.append(line)
            else:
                jsons.append(line)
            i+=1


tuples = []
cat1 = []
cat2 = []
for c,l in enumerate(cats):
    parts = l.split('\t')
    score_json = jsons[c].strip()
    joke = parts[1].strip()
    try:
        scale = json.loads(score_json)['scale']
    except:
        scale = 0
    ## cat1 , cat2 , or cat3, which category it belongs to
    if 'cat1' in parts[0]:
        tuples.append((1, parts[1].strip(), int(scale)))
        cat1.append((joke, int(scale)))
    else:
        tuples.append((2, parts[1].strip(), int(scale)))
        cat2.append((joke, int(scale)))

counts1 = Counter(x[1] for x in cat1)
counts2 = Counter(x[1] for x in cat2)
print(counts1)
print(counts2)

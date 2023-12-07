import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Utility import *

base = read("./check_in.json", [0,0.6])
block = read("./check_in.json", [0.6,0.7])

# p, connection = propose_p(base, block)

# connection = collections.Counter(connection)

# print(connection)

# Draw a figure

U,I = number_subGraph(block)
graph_map = collections.Counter()

for i in U:
	graph_map[i] += 1
for i in I:
	graph_map[i] += 1

data = list(graph_map.items())
data = sorted(data, key=lambda x : x[0])
x,y = list(zip(*data))
y = collections.Counter(y[1:])

y = list(y.items())
xx,yy = list(zip(*y))

print(xx)
print(yy)

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(aspect="equal"))

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)


wedges, texts = ax.pie(yy, textprops=dict(color="w"))

ax.legend(wedges, xx,
          title=" Number of nodes in a subgraph ",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

ax.set_title(" ")

plt.show()

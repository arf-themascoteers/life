import math
from itertools import accumulate


def get_rs(num_agents):
    r_values = [1 / math.pow(1 + 1, 2) for i in range(num_agents)]
    s = sum(r_values)
    r_values = [r / s for r in r_values]
    r_values = list(accumulate(r_values))
    return r_values

for i in range(2,31):
    rs = get_rs(i)
    print(rs)
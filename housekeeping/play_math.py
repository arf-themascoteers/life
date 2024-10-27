import torch
import math
from itertools import accumulate


def inverse_sigmoid_torch(x):
    return -torch.log(1.0 / x - 1.0)


def rs(num_agents):
    m = num_agents/4
    r_values = [math.exp(-i/m) for i in range(num_agents)]
    s = sum(r_values)
    r_values = [r/s for r in r_values]
    r_values = list(accumulate(r_values))
    return r_values


def det():
    original_size = 200
    target = 2
    num_agents = 5
    slot_size = 1 / (target + 1)
    print(slot_size)
    ticks = [i * slot_size for i in range(target+2)]
    print(ticks)
    ticks = ticks[1:-1]
    print(ticks)

    band_unit = 1/original_size

    v = []
    for i in range(target):
        v.append(i*band_unit)

    start_vector = torch.tensor(v, dtype=torch.float)
    print(start_vector)

    v = []
    for i in range(target):
        v.append(1-i*band_unit)
    v.reverse()
    end_vector = torch.tensor(v, dtype=torch.float)

    distance = torch.dist(start_vector, end_vector, p=2)
    print(distance)

    rs = [math.sqrt((i+1)/num_agents)*distance for i in range(num_agents)]
    print(rs)

    raw_params = [inverse_sigmoid_torch(t) for t in torch.tensor(ticks, dtype=torch.float32)]
    print(raw_params)


if __name__ == '__main__':
    print(inverse_sigmoid_torch(torch.tensor(0.99)))
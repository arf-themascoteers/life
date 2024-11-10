def n(N):
    return int(round(5 + (N - 5) * (10 - 5) / (30 - 5),0))

for i in range(5,31):
    print(n(i))
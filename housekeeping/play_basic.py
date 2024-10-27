def x():
    return 1,2

y, z = zip(*[x() for i in range(3)])
print(z)
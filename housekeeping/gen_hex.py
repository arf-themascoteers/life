import random

hex_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
print(hex_color)

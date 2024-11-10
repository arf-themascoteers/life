def get_band_number(wl):
    a = wl-400
    b = a/0.5
    return b


print(get_band_number(400))
print(get_band_number(499.5))
print(get_band_number(2450))
print(get_band_number(2499.5))


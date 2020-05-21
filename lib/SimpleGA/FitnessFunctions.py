def multimodal(chromosome):
    x = chromosome[0]
    y = chromosome[1]

    modes = x**4 - 5 * x**2 + y**4 - 5 * y**2
    tilt = 0.5 * x * y + 0.3 * x + 15
    stretch = 0.1

    return stretch * (modes + tilt)
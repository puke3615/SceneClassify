
def build_generator(train, v):
    def wrap(value):
        return float(train) and value
    print(wrap(v))
build_generator(True, 0.5)
build_generator(False, 0.5)
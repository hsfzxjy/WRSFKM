class Params:

    def __init__(self, dct):

        self.__dct = dct

    def __getattr__(self, name):
        return self.__dct.get(name, None)

    def __str__(self):
        return str(self.__dct)

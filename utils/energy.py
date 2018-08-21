class Energy:

    def __init__(self, tol=3):

        self.lst = []
        self.tol = tol

    def converged(self, next_E=None):

        if next_E is not None:
            lst = self.lst + [next_E]
        else:
            lst = self.lst

        if len(lst) < self.tol:
            return False

        for i in range(-self.tol, -1):
            if abs(lst[i] - lst[i + 1]) > 1:
                return False

        return True

    def add(self, E):

        self.lst.append(E)

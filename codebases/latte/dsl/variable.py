import compiler, inspect
import compiler.ast

class Variable(object):
    """ Variable class for shared variable analysis """
    def __init__(self, name):
        super(Variable, self).__init__()
        self.name = name

class IndexVariable(Variable):
    """docstring for IndexVariable"""
    def __init__(self, name, start, stop, step=1):
        super(IndexVariable, self).__init__(name)
        self.start = start
        self.stop = stop
        self.step = step

class ArrayVariable(Variable):
    """docstring for ArrayVariable"""
    def __init__(self, name):
        super(ArrayVariable, self).__init__(name)

'''
    Latte Semantic Analyzer
'''
import ast

neuron_code_generators = { }

class NeuronCodeGenerator(object):
    """class for neuron specific code generation"""
    def __init__(self, neuron_ast):
        super(NeuronCodeGenerator, self).__init__()
        # field variables
        self.neuron_ast = neuron_ast
        self.fields = { }
        # AST processing
        for function_ast in self.extract_functions():
            self.process_init(function_ast)

    def extract_functions(self):
        """
        extract all functions defined in a neuron class
        """
        for node in ast.walk(self.neuron_ast):
            if isinstance(node, ast.FunctionDef):
                yield node
        return

    def process_init(self, function_ast):
        if function_ast.name != "__init__":
            return
        for node in ast.walk(function_ast):
            # only look at the Assign nodes
            if isinstance(node, ast.Assign):
                # check if the target is an attribute of self
                if isinstance(node.targets[0], ast.Attribute):
                    if node.targets[0].value.id == "self":
                        # we need to record this
                        self.add_field(node.targets[0].attr, node.value)

    def add_field(self, field_name, field_type):
        if isinstance(field_type, ast.Num):
            self.fields[field_name] = "float"
            return
        elif isinstance(field_type, ast.List):
            # print field_name, field_type.elts
            if field_type.elts == []:
                self.fields[field_name] = "vector<float>"
                return
            if isinstance(field_type.elts[0], ast.List):
                self.fields[field_name] = "vector<vector<float>>"
                return
        print "ignore field", field_name


def extract_neuron_classes(filename):
    """
    read in a file and find out all
    classes that is either Neuron,
    or its subtype
    """
    source = open(filename, "r")
    AST = ast.parse(source=source.read())
    source.close()
    for node in ast.walk(AST):
        if isinstance(node, ast.ClassDef):
            # if the class is Neuron itself
            if node.name == "Neuron":
                yield node 
            # if the class is a subclass of Neuron
            for base in node.bases:
                if base.id == "Neuron":
                    yield node
    return

def process_lib(filename):
    """
    read in a library file parse all neuron types,
    and their associated forward/backward functions
    """
    for neuron_ast in extract_neuron_classes(filename):
        neuron_code_generators[neuron_ast.name] = NeuronCodeGenerator(neuron_ast)

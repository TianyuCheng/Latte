'''
    Latte Semantic Analyzer
'''
import ast

neuron_analyzers = { }

class NeuronAnalyzer(object):
    """class for neuron specific code generation"""
    def __init__(self, neuron_ast):
        super(NeuronAnalyzer, self).__init__()
        # field variables
        self.name = neuron_ast.name
        self.neuron_ast = neuron_ast
        self.fields = { }
        # AST processing
        for function_ast in self.extract_functions():
            self.process_init(function_ast)

    def analyze(self, enm_name):
        """ pass in enm_name to generate SoA code"""
        self.enm = enm_name
        self.forward_codes = [ ]
        self.backward_codes = [ ]

        # find base class and incorporates its fields
        for base in self.neuron_ast.bases:
            if base.id in neuron_analyzers:
                for field, field_type in neuron_analyzers[base.id].fields.iteritems():
                    self.fields[field] = field_type
        
        # AST processing
        for function_ast in self.extract_functions():
            self.process_forward(function_ast)

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
                        # we need to record this field
                        self.add_field(node)

    def process_forward(self, function_ast):
        if function_ast.name != "forward":
            return
        for node in ast.walk(function_ast):
            if isinstance(node, ast.Assign):
                var_name = self.parse_var_name(node.targets[0])
                var_value = self.parse_expr(node.value)
                # ignore data copying (naming convention, ends with 'inputs')
                if var_name.split('[')[0].endswith("inputs"): continue
                print "Assignment: %s = %s" % (var_name, var_value)

    def add_field(self, node):
        var_name = self.parse_var_name(node.targets[0])
        var_type = self.parse_var_type(node)
        if var_type is None:
            print "ignore %s.%s" % (self.name, var_name)
        else:
            self.fields[var_name] = var_type

    def parse_expr(self, node):
        if isinstance(node, ast.Call):
            func = self.parse_var_name(node.func)
            args = map(self.parse_var_name, node.args)
            # TODO: try mapping to MLK operations here
            return func + "(" + ', '.join(args) + ")"
        if isinstance(node, ast.BinOp):
            print ast.dump(node)
        return self.parse_var_name(node)
    
    def parse_var_name(self, node):
        # simply Name node
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            var_name = node.attr
            # translate array of data into SoA in Cpp
            if var_name in self.fields and self.fields[var_name].startswith("vector"):
                var_name = "%s_enm_%s" % (self.enm, node.attr)
            return var_name
        if isinstance(node, ast.Index):
            return self.parse_var_name(node.value)
        if isinstance(node, ast.Subscript):
            return self.parse_var_name(node.value) + "[%s]" % self.parse_var_name(node.slice)

    def parse_var_type(self, node):
        field_type = node.value
        if isinstance(field_type, ast.Num):
            return "float*"
        elif isinstance(field_type, ast.List):
            # print field_name, field_type.elts
            if field_type.elts == []:
                return "vector<float*>"
            if isinstance(field_type.elts[0], ast.List):
                return "vector<vector<float*>>"

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
        neuron_analyzers[neuron_ast.name] = NeuronAnalyzer(neuron_ast)
    
    # # test run
    # for name, neuron_analyzer in neuron_analyzers.iteritems():
    #     # neuron_analyzer.analyze("ip1")
    #     print neuron_analyzer.fields
    return neuron_analyzers

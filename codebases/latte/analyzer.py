'''
    Latte Semantic Analyzer
'''
import ast
from ast_matcher import *
from templates import *

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

    def init_fields(self):
        """ pass in enm_name to generate SoA code"""
        # find base class and incorporates its fields
        for base in self.neuron_ast.bases:
            if base.id in neuron_analyzers:
                for field, field_type in neuron_analyzers[base.id].fields.iteritems():
                    self.fields[field] = field_type

    def analyze(self, enm_info):
        self.enm_name = enm_info[:1]
        self.fp_codes = []
        self.bp_codes = []
        for function in self.extract_functions():
            self.process_forward(function)
            self.process_backward(function)
        return '\n'.join(self.fp_codes), '\n'.join(self.bp_codes)

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
        # for node in ast.walk(function_ast):
        #     if isinstance(node, ast.Assign):
        #         var_name = self.parse_var_name(node.targets[0])
        #         var_value = self.parse_expr(node.value)
        #         # ignore data copying (naming convention, ends with 'inputs')
        #         if var_name.split('[')[0].endswith("inputs"): continue
        #         print "Assignment: %s = %s" % (var_name, var_value)
        print "-----------------------------"
        for stmt in stmt_walk(function_ast):
            stmt_code = self.process_stmt(stmt)
            self.fp_codes.append(stmt_code)
        self.fp_codes = filter(lambda x: x is not None, self.fp_codes)

    def process_backward(self, function_ast):
        if function_ast.name != "backward":
            return

    def process_stmt(self, stmt):
        if isinstance(stmt, ast.Assert):
            return
        if isinstance(stmt, ast.Assign):
            var_name = self.parse_var_name(stmt.targets[0])
            var_value = self.parse_expr(stmt.value)
            return "%s = %s;" % (var_name, var_value)
        if isinstance(stmt, ast.For):
            for_stmt = "for (int {i} = {start}; {i} < {stop}; ++{i}) {code}"
            # try to match for loop by template
            match_result = None
            tmpl = template_for("range")
            print ast.dump(stmt)
            if tmpl.match(stmt):
                match_result = tmpl.wildcard
            # assert match_result is not None
            if match_result is None:
                return None
            # print "==============>", match_result
            for_index = self.parse_var_name(match_result["i"])
            for_stop = self.parse_expr(match_result["N"])
            body = self.process_stmt(match_result["body"])
            return for_stmt.format(i=for_index, start=0, stop=for_stop, code=body)
        print "=====> PROCESS STMT: (NO MATCH)", ast.dump(stmt)

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
            # print ast.dump(node)
            if isinstance(node.op, ast.Add):
                op = " + "
                return "(" + self.parse_expr(node.left) + op + self.parse_expr(node.right) + ")"
            elif isinstance(node.op, ast.Sub):
                op = " - "
                return "(" + self.parse_expr(node.left) + op + self.parse_expr(node.right) + ")"
            elif isinstance(node.op, ast.Mult):
                op = " * "
                return "(" + self.parse_expr(node.left) + op + self.parse_expr(node.right) + ")"
            elif isinstance(node.op, ast.Div):
                op = " / "
                return "(" + self.parse_expr(node.left) + op + self.parse_expr(node.right) + ")"
            elif isinstance(node.op, ast.Pow):
                return "pow(" + self.parse_expr(node.left) + ", " + self.parse_expr(node.right) + ")"
        return self.parse_var_name(node)
    
    def parse_var_name(self, node):
        # simply Name node
        if isinstance(node, str):
            return node
        if isinstance(node, ast.Num):
            return str(node.n)
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

def process_lib(filename, ensemble_info):
    """
    read in a library file parse all neuron types,
    and their associated forward/backward functions
    """
    for neuron_ast in extract_neuron_classes(filename):
        neuron_analyzers[neuron_ast.name] = NeuronAnalyzer(neuron_ast)
    
    # NOTE: we extend Neuron to base class, no need to second pass
    # for name, neuron_analyzer in neuron_analyzers.iteritems():
    #    neuron_analyzer.init_fields()
    
    forward_codes = { }
    backward_codes = { }
    for ensemble in ensemble_info:
        _name, _type, _prev, _dim_x, _dim_y, _neuron_type = ensemble[:6]
        # print neuron_analyzers
        analyzer = neuron_analyzers[_neuron_type]
        fp_code, bp_code = analyzer.analyze(ensemble)
        forward_codes[_name] = fp_code
        backward_codes[_name] = bp_code
    return neuron_analyzers, forward_codes, backward_codes

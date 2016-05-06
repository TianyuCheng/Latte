'''
    Latte Semantic Analyzer
'''
import ast
from ast_matcher import *
from templates import *
from copy import deepcopy
from translator import *
from lib import *

neuron_analyzers = { }

field_blacklist = set([\
    "prev_dim_x",
    "prev_dim_y",
    "forward_adj",          # not sure we want to allocate
    "backward_adj",         # memory for forward and backward adj
])

class NeuronAnalyzer(object):
    """class for neuron specific code generation
    Each neuron type has its own analyzer"""
    def __init__(self, neuron_ast, options):
        super(NeuronAnalyzer, self).__init__()
        self.MKL_FLAG = options.MKL_FLAG
        self.DP_FLAG = options.DP_FLAG
        # field variables
        self.name = neuron_ast.name
        self.neuron_ast = neuron_ast
        self.fields = { }
        self.name2enm = None
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

    def delete_unused_fields(self):
        # do not delete any field in the base class
        # except for inputs and anything ends with inputs
        if self.name == "Neuron":
            to_delete = [ ]
            for field in self.fields:
                if field.endswith("inputs"):
                    to_delete.append(field)
            for delete in to_delete:
                del self.fields[delete]
            return

        used_variables = set()
        for function in self.extract_functions():
            if function.name == "forward" or function.name == "backward" or function.name == "__claim__":
                for node in ast.walk(function):
                    if isinstance(node, ast.Attribute):
                        used_variables.add(node.attr)
        unused_variables = set(self.fields.keys()) - used_variables
        # delete all unused variables from fields
        for field_name in unused_variables:
            if field_name in self.fields:
                del self.fields[field_name]
        print "UNUSED FIELDS:", unused_variables

    def analyze(self, enm_info, name2enm, conn_type, share_weights):
        self.name2enm = name2enm
        self.conn_type = conn_type
        self.share_weights = share_weights
        self.enm, _, self.enm_prev, _dim_x, _dim_y, _, self.aux  = enm_info[:7]
        self.fp_codes = []
        self.bp_codes = []
        for function in self.extract_functions():
            self.process_forward(function, enm_info)
        for function in self.extract_functions():
            self.process_backward(function, enm_info)
        print self.enm, self.fp_codes
        return None if len(self.fp_codes) == 0 else self.fp_codes[0], \
               None if len(self.bp_codes) == 0 else self.bp_codes[0]

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

    def curr_enm_dim(self):
        curr_enm_info = self.name2enm[self.enm]
        return tuple(curr_enm_info[3:5])

    def curr_enm_type(self):
        curr_enm_info = self.name2enm[self.enm]
        return curr_enm_info[2]

    def prev_enm_dim(self):
        if self.enm_prev not in self.name2enm:
            return (-1, -1)     # some dummy value, likely not to be used
        prev_enm_info = self.name2enm[self.enm_prev]
        return tuple(prev_enm_info[3:5])

    def prev_enm_type(self):
        prev_enm_info = self.name2enm[self.enm_prev]
        return prev_enm_info[2]

    '''
       TODO: add pattern match for statment here
    '''
    def process_forward(self, function_ast, enm_info):
        if function_ast.name != "forward": return
        #######################################################
        curr_enm, prev_enm = enm_info[0], enm_info[2]
        curr_dim = self.curr_enm_dim()
        for_node_x = ForNode(ConstantNode("x"), ConstantNode(0), ConstantNode(curr_dim[0]), ConstantNode(1))
        for_node_y = ForNode(ConstantNode("y"), ConstantNode(0), ConstantNode(curr_dim[1]), ConstantNode(1))

        self.fp_codes.append(for_node_x)
        for_node_x.add_child(for_node_y)

        prev_analyzer = None
        if self.enm_prev is not None and self.enm_prev[2] in neuron_analyzers:
            prev_analyzer = neuron_analyzers[self.enm_prev[2]]
        trans = Translator(self, prev_analyzer, curr_enm, prev_enm, \
                 self.conn_type, self.share_weights, self.MKL_FLAG, self.DP_FLAG)
        for stmt in stmt_walk(function_ast):
            for_node_y.add_child(trans.process_stmt(stmt))

        # remove the double nested forloop if nothing is inside
        if len(for_node_y.children) == 0:
            self.fp_codes = []
        #######################################################

    def process_backward(self, function_ast, enm_info):
        if function_ast.name != "backward": return
        #######################################################
        curr_enm, prev_enm = enm_info[0], enm_info[2]
        curr_dim = self.curr_enm_dim()
        for_node_x = ForNode(ConstantNode("x"), ConstantNode(0), ConstantNode(curr_dim[0]), ConstantNode(1))
        for_node_y = ForNode(ConstantNode("y"), ConstantNode(0), ConstantNode(curr_dim[1]), ConstantNode(1))

        self.bp_codes.append(for_node_x)
        for_node_x.add_child(for_node_y)

        prev_analyzer = None
        if self.enm_prev is not None:
            prev_type = self.name2enm[self.enm_prev][5]
            if prev_type in neuron_analyzers:
                prev_analyzer = neuron_analyzers[prev_type]
        trans = Translator(self, prev_analyzer, curr_enm, prev_enm,\
                self.conn_type, self.share_weights, self.MKL_FLAG, self.DP_FLAG)
        for stmt in stmt_walk(function_ast):
            for_node_y.add_child(trans.process_stmt(stmt))

        # remove the double nested forloop if nothing is inside
        if len(for_node_y.children) == 0:
            self.bp_codes = []
        #######################################################

    def add_field(self, node):
        var_name = node.targets[0].attr
        var_type = self.parse_var_type(node)
        if var_type is None or var_name in field_blacklist:
            print "ignore %s.%s" % (self.name, var_name)
        else:
            self.fields[var_name] = var_type

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

    def get_field_type(self, field):
        if field in self.fields:
            return self.fields[field]
        return None

def extract_neuron_classes(filename):
    """
    read in a file and find out all
    classes that is either Neuron,
    or its subtype
    """
    source = open(filename, "r")
    AST = ast.parse(source.read())
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

def extract_functions(filename):
    """
    read in a file and find out all
    layer function definitions
    """
    source = open(filename, "r")
    AST = ast.parse(source.read())
    source.close()
    for node in ast.walk(AST):
        if isinstance(node, ast.FunctionDef):
            yield node 
    return

def process_add_connection_helper(all_functions, function_ast):
    if function_ast.name.endswith("Layer"):
        args = list(map(lambda x: x.id, function_ast.args.args))
        layer_name = function_ast.name
        tmpl = template_add_connection()
        if tmpl.matchall(function_ast):
            # right now we assume there should be only one match
            # i.e. only one add_connection call in the layer function
            mapping = tmpl.matches[0]['mappings']
            return layer_name, args, mapping
        else:
            # not finding add_connection call, it must be somewhere
            # called by some other functions
            for node in ast.walk(function_ast):
                if isinstance(node, ast.Call):
                    # we only process top level function calls
                    # no attributes is allowed, e.g. self.Layer()
                    if isinstance(node.func, ast.Name):
                        if node.func.id in all_functions:
                            call_args = list(map(lambda x: x.id, node.args))
                            callee_function = all_functions[node.func.id]
                            sublayer, callee_args, mappings = process_add_connection_helper(\
                                    all_functions, callee_function)
                            if sublayer is not None:
                                # rename arguments
                                mappings = deepcopy(mappings)
                                for new_name, old_name in zip(call_args, callee_args):
                                    mappings = RewriteName(old_name, new_name).visit(mappings)
                                # return layer_name, args, mapping
                                return layer_name, args, mappings
            print layer_name, "NO MATCH FOR ADD_CONNECTION"
    return None, None, None

def ast2lambda(mapping, args, ensemble, name2enm):
    # generate module wrapper
    mapping = ast.Module(\
        body = [\
            ast.Assign(\
                targets = [ast.Name(id="func", ctx=ast.Store())],\
                value = mapping
            )
        ]
    )
    # some fields that will be used in the lambda
    dim_x, dim_y = ensemble['dim_x'], ensemble['dim_y']
    if "prev" in ensemble:
        prev_dim_x, prev_dim_y = name2enm[ensemble['prev']][3:5]
        mapping = SubstituteAttributeToNum('self', 'dim_x', dim_x).visit(mapping)
        mapping = SubstituteAttributeToNum('self', 'dim_y', dim_y).visit(mapping)
        mapping = SubstituteAttributeToNum('prev', 'dim_x', prev_dim_x).visit(mapping)
        mapping = SubstituteAttributeToNum('prev', 'dim_y', prev_dim_y).visit(mapping)
    # substitue args
    for arg in args:
        if arg in ensemble:
            mapping = SubstituteNameToNum(arg, ensemble[arg]).visit(mapping)
    # compile the ast
    mapping = ast.fix_missing_locations(mapping)
    # print ast.dump(mapping)
    codeobj = compile(mapping, '<string>', 'exec')
    exec(codeobj)
    return func

def check_uniform_dependency(args, mapping, ensemble_info, name2enm):
    dim_x, dim_y = ensemble_info['dim_x'], ensemble_info['dim_y']
    mapping = ast2lambda(mapping, args, ensemble_info, name2enm)
    # number of dimension
    dim = 2
 
    mapped_indices = sorted(mapping(*tuple([ 0 for i in range(dim) ])))
    # print mapped_indices
    for d in range(dim):
        for i in range(dim_x):
            for j in range(dim_y):
                # double dimension
                if sorted(mapping(i, j)) != mapped_indices:
                    return False
    return True

def check_one_to_one(args, mapping, ensemble_info, name2enm):
    dim_x, dim_y = ensemble_info['dim_x'], ensemble_info['dim_y']
    mapping = ast2lambda(mapping, args, ensemble_info, name2enm)
    # number of dimension
    dim = 2
 
    mapped_indices = sorted(mapping(*tuple([ 0 for i in range(dim) ])))
    # print mapped_indices
    for i in range(dim_x):
        for j in range(dim_y):
            # double dimension
            if len(mapping(i, j)) != 1:
                return False
    return True

def process_ensemble_share_weight(all_functions, function_ast):
    # find directly in the function
    for tmpl in new_ensemble_templates:
        if tmpl.matchall(function_ast):
            # only extract the first mapping
            match = tmpl.matches[0]
            if "share_weights" in match:
                return bool(match["share_weights"])
    # find in the called functions
    for node in ast.walk(function_ast):
        if isinstance(node, ast.Call):
            # we only process top level function calls
            # no attributes is allowed, e.g. self.Layer()
            if isinstance(node.func, ast.Name):
                if node.func.id in all_functions:
                    if process_ensemble_share_weight(all_functions,\
                            all_functions[node.func.id]):
                        return True
    return False


def process_add_connection(filename, name2enm):
    """TODO: Docstring for process_add_connection.
    :returns: TODO

    """
    conn_types = { }

    print "ADD CONNECTION ----------------------------"
    all_functions = dict(map(lambda x: (x.name, x), extract_functions(filename)))
    for function_ast in all_functions.itervalues():
        layer_name, args, mapping = \
                process_add_connection_helper(all_functions, function_ast)
        if layer_name is not None:
            share_weights = process_ensemble_share_weight(all_functions, function_ast)
            conn_types[layer_name] = (args, mapping, share_weights)

    # review the mapping functions for layer connections
    for layer, (args, mappings, share_weights) in conn_types.iteritems():
        print ""
        print "layer:", layer
        print "args: ", args
        print "share weights: ", share_weights
        print "conn: ", ast_dump(mappings)
    print "------------------------------------------"
    return conn_types

def process_lib(filename, ensemble_info, name2enm, conn_types, options):
    """
    read in a library file parse all neuron types,
    and their associated forward/backward functions
    """
    # create neuron analyzer for each neuron subtype
    for neuron_ast in extract_neuron_classes(filename):
        neuron_analyzers[neuron_ast.name] = NeuronAnalyzer(neuron_ast, options)
    
    # process the fields of the neuron base types
    neuron_analyzers['Neuron'].delete_unused_fields()
    for name, neuron_analyzer in neuron_analyzers.iteritems():
       neuron_analyzer.init_fields()

    # delete unused variables
    print "+++++++++++++++++++++++++++++++++++++++++++"
    for name, neuron_analyzer in neuron_analyzers.iteritems():
       neuron_analyzer.delete_unused_fields()
    print "+++++++++++++++++++++++++++++++++++++++++++"

    # DOING SECOND TIME TO GET THE BASE's VARIABLES
    # process the fields of the neuron base types
    for name, neuron_analyzer in neuron_analyzers.iteritems():
       neuron_analyzer.init_fields()
       # print "-------------->", neuron_analyzer.name, neuron_analyzer.fields
    
    print "###########################################"
    forward_codes = { }
    backward_codes = { }
    fp_codes = [ ]
    bp_codes = [ ]

    for ensemble in ensemble_info:
        _name, _type, _prev, _dim_x, _dim_y, _neuron_type = ensemble[:6]
        # print neuron_analyzers
        analyzer = neuron_analyzers[_neuron_type]
        conn_type = None
        share_weights = False
        if _type in conn_types:
            _, conn_type, share_weights = conn_types[_type]
        fp_code_node, bp_code_node = analyzer.analyze(ensemble, name2enm, conn_type, share_weights)
        forward_codes[_name] = fp_code_node
        backward_codes[_name] = bp_code_node
        fp_codes.append(fp_code_node)
        bp_codes.append(bp_code_node)

    return neuron_analyzers, forward_codes, backward_codes, fp_codes, bp_codes

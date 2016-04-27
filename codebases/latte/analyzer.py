'''
    Latte Semantic Analyzer
'''
import ast
from ast_matcher import *
from templates import *
from copy import deepcopy
from translator import *

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
    def __init__(self, neuron_ast, pattern_match = True):
        super(NeuronAnalyzer, self).__init__()
        self.enable_pattern_match = pattern_match
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
        used_variables = set()
        for function in self.extract_functions():
            if function.name == "forward" or function.name == "backward":
                for node in ast.walk(function):
                    if isinstance(node, ast.Attribute):
                        used_variables.add(node.attr)
        unused_variables = set(self.fields.keys()) - used_variables
        # delete all unused variables from fields
        for field_name in unused_variables:
            if field_name in self.fields:
                del self.fields[field_name]
        print "UNUSED FIELDS:", unused_variables

    def analyze(self, enm_info, name2enm):
        self.name2enm = name2enm
        self.enm, _, self.enm_prev, _dim_x, _dim_y  = enm_info[:5]
        self.fp_codes = []
        self.bp_codes = []
        for function in self.extract_functions():
            self.process_forward(function, enm_info)
        for function in self.extract_functions():
            self.process_backward(function, enm_info)
        return ';\n'.join(map(str, self.fp_codes)) + "\n", \
               ';\n'.join(map(str, self.bp_codes)) + "\n"

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
        # self.enm, _, self.enm_prev, _dim_x, _dim_y  = enm_info[:5]
        # statements = list(stmt_walk(function_ast))
        # for sid in range(len(statements)):
        #     self.fp_codes.append(self.process_stmt(statements[sid], statements[sid:sid+2]))
        # self.fp_codes = filter(lambda x: x is not None, self.fp_codes)
        # if len(self.fp_codes) == 0: return
        # else: 
        #     # TODO: change the universal for-loop to use shared variable analysis
        #     self.fp_codes = [ "\tfor (int y = 0; y < %d; y++) {" % _dim_y ] + self.fp_codes
        #     self.fp_codes = [ "for (int x = 0; x < %d; x++) {" % _dim_x ] + self.fp_codes
        #     self.fp_codes = [ "// Forward Propagation for " + self.enm ] + self.fp_codes
        #     self.fp_codes.append("\t}\n}")

        #######################################################
        curr_enm, prev_enm = enm_info[0], enm_info[2]
        curr_dim = self.curr_enm_dim()
        for_node_x = ForNode(ConstantNode("x"), ConstantNode(0), ConstantNode(curr_dim[0]), ConstantNode(1))
        for_node_y = ForNode(ConstantNode("y"), ConstantNode(0), ConstantNode(curr_dim[1]), ConstantNode(1))

        self.fp_codes.append(for_node_x)
        for_node_x.add_child(for_node_y)

        trans = Translator(self, curr_enm, prev_enm, self.enable_pattern_match)
        for stmt in stmt_walk(function_ast):
            for_node_y.add_child(trans.process_stmt(stmt))

        # remove the double nested forloop if nothing is inside
        if len(for_node_y.children) == 0:
            self.fp_codes = []
        #######################################################

    def process_backward(self, function_ast, enm_info):
        if function_ast.name != "backward": return
        # self.enm, _, self.enm_prev, _dim_x, _dim_y  = enm_info[:5]
        # statements = list(stmt_walk(function_ast))
        # for sid in range(len(statements)):
        #     self.bp_codes.append(self.process_stmt(statements[sid], statements[sid:sid+2]))
        # self.bp_codes = filter(lambda x: x is not None, self.bp_codes)
        # if len(self.bp_codes) == 0: return
        # else: 
        #     self.bp_codes = [ \
        #             "// Backward Propagation for " + self.enm, \
        #             "for (int x = 0; x < %d; x ++) {" % _dim_x, \
        #             "\tfor (int y = 0; y < %d; y ++) {" % _dim_y ] \
        #             + self.bp_codes
        #     self.bp_codes.append("\t}\n}")
        #######################################################
        curr_enm, prev_enm = enm_info[0], enm_info[2]
        curr_dim = self.curr_enm_dim()
        for_node_x = ForNode(ConstantNode("x"), ConstantNode(0), ConstantNode(curr_dim[0]), ConstantNode(1))
        for_node_y = ForNode(ConstantNode("y"), ConstantNode(0), ConstantNode(curr_dim[1]), ConstantNode(1))

        self.bp_codes.append(for_node_x)
        for_node_x.add_child(for_node_y)

        trans = Translator(self, curr_enm, prev_enm, self.enable_pattern_match)
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

    # def process_stmt(self, stmt, statements=[]):
    #     if isinstance(stmt, ast.Pass): return
    #     if isinstance(stmt, ast.Assert): return
    #     if isinstance(stmt, ast.Assign):
    #         tmpl = template_asgn("output")
    #         if tmpl.prefix_of(stmt):
    #             expr = self.parse_expr(tmpl.wildcard['exp'])
    #             return "\t"*2 + "%s[x][y] = %s;" % (self.enm+"_output", expr)
    #
    #         tmpl = template_asgn("grad_activation")
    #         if tmpl.prefix_of(stmt):
    #             expr = self.parse_expr(tmpl.wildcard['exp'])
    #             return "\t"*2 + "%s[x][y] = %s;" % (self.enm+"_grad_activation", expr)
    #
    #         tmpl = template_asgn("grad_output")
    #         if tmpl.prefix_of(stmt):
    #             expr = self.parse_expr(tmpl.wildcard['exp'])
    #             return "\t"*2 + "*(%s+x*%s+y) = %s;" % \
    #                     (self.enm+"_grad_output", self.name2enm[self.enm][4], expr)
    #
    #         var_name = self.parse_var_name(stmt.targets[0])
    #         var_value = self.parse_expr(stmt.value)
    #         return "\t"*2+ "%s = %s;" % (var_name, var_value)
    #
    #     if isinstance(stmt, ast.For):
    #         # pattern match found
    #         if not self.enable_pattern_match:
    #             set_ast_match(False)
    #         tmpl = template_dot_product("range")
    #         matched = tmpl.match(stmt) 
    #         if matched:
    #             A, B, i, j, dim_x, dim_y, C = map(self.parse_var_name, tmpl.wildcard.values())
    #             prev = self.name2enm[self.enm][2]
    #             prev_dim_x, prev_dim_y = self.name2enm[prev][3:5]
    #             pm_str = "\t"*2+"sgemm_dp(&%s, %s[x][y], %s, %s*%s);" % \
    #                     (C, A, B, prev_dim_x, prev_dim_y)
    #             return pm_str
    #
    #         tmpl = template_bp_scalar_prod()
    #         matched = tmpl.match(stmt)
    #         if matched:
    #             B, _,  _, dim_x, dim_y, scalar = map(self.parse_var_name, tmpl.wildcard.values())
    #             if "DataLayer" in self.name2enm[self.name2enm[self.enm][2]][1]: return ""
    #             C = self.name2enm[self.enm][2] + "_grad_output"
    #             prev = self.name2enm[self.enm][2]
    #             prev_dim_x, prev_dim_y = self.name2enm[prev][3:5]
    #             pm_str = "\t"*2+"sgemm_axpy(%s, %s, %s[x][y], %s*%s);" % \
    #                     (C, scalar, B, prev_dim_x, prev_dim_y)
    #             #print "========>", pm_str
    #             return pm_str
    #
    #         tmpl = template_bp_axpy()
    #         matched = tmpl.match(stmt)
    #         # print ast.dump(stmt)
    #         if matched:
    #             print map(self.parse_var_name, tmpl.wildcard.values())
    #             C, B, di, dj, scalar, dim_x, dim_y = map(self.parse_var_name, tmpl.wildcard.values())
    #             prev = self.name2enm[self.enm][2]
    #             prev_dim_x, prev_dim_y = self.name2enm[prev][3:5]
    #             pm_str = "\t"*2+"sgemm_axpy(%s[x][y], %s, %s, %s*%s);" % \
    #                     (C, scalar, B, prev_dim_x, prev_dim_y)
    #             #print "========>", pm_str
    #             return pm_str
    #         set_ast_match(True)
    #
    #         # pattern match not found
    #         for_stmt = "for (int {i} = {start}; {i} < {stop}; ++{i}) {{\n{code}\n}}"
    #         # try to match for loop by template
    #         match_result = None
    #         tmpl = template_for("range")
    #         if tmpl.match(stmt):
    #             match_result = tmpl.wildcard
    #         # assert match_result is not None
    #         if match_result is None:
    #             return None
    #         # print "==============>", match_result
    #         for_index = self.parse_var_name(match_result["i"])
    #         for_stop = self.parse_expr(match_result["N"])
    #         body = self.process_stmt(match_result["body"])
    #         return for_stmt.format(i=for_index, start=0, stop=for_stop, code=body)
    #     print "=====> PROCESS STMT: (NO MATCH)", ast.dump(stmt)
    #
    # def parse_expr(self, node):
    #     if isinstance(node, ast.Call):
    #         func = self.parse_var_name(node.func)
    #         args = map(self.parse_var_name, node.args)
    #         # TODO: try mapping to MKL operations here
    #         return func + "(" + ', '.join(args) + ")"
    #     if isinstance(node, ast.BinOp):
    #         # print ast.dump(node)
    #         if isinstance(node.op, ast.Add):
    #             op = " + "
    #             return "(" + self.parse_expr(node.left) + op + self.parse_expr(node.right) + ")"
    #         elif isinstance(node.op, ast.Sub):
    #             op = " - "
    #             return "(" + self.parse_expr(node.left) + op + self.parse_expr(node.right) + ")"
    #         elif isinstance(node.op, ast.Mult):
    #             op = " * "
    #             return "((" + self.parse_expr(node.left) + ")"+ op + "(" + self.parse_expr(node.right) + "))"
    #         elif isinstance(node.op, ast.Div):
    #             op = " / "
    #             return "(" + self.parse_expr(node.left) + op + self.parse_expr(node.right) + ")"
    #         elif isinstance(node.op, ast.Pow):
    #             return "pow(" + self.parse_expr(node.left) + ", " + self.parse_expr(node.right) + ")"
    #     return self.parse_var_name(node)
    #
    # def parse_var_name(self, node):
    #     # simply Name node
    #     if isinstance(node, str):
    #         return node
    #     if isinstance(node, ast.Num):
    #         return str(node.n)
    #     if isinstance(node, ast.Name):
    #         return node.id
    #     if isinstance(node, ast.Attribute):
    #         var_name = node.attr
    #         # translate array of data into SoA in Cpp
    #         # if var_name in self.fields: and self.fields[var_name].startswith("vector"):
    #         if self.name2enm is not None:
    #             # prev_dim_x and prev_dim_y are built-in variables,
    #             # so we hard code it
    #             if var_name == "prev_dim_x":
    #                 return self.name2enm[self.enm_prev][3]
    #             if var_name == "prev_dim_y":
    #                 return self.name2enm[self.enm_prev][4]
    #             if var_name == "dim_x":
    #                 return self.name2enm[self.enm][3]
    #             if var_name == "dim_y":
    #                 return self.name2enm[self.enm][4]
    #             if var_name == "grad_output":
    #                 return "*(%s_grad_output+x*%s+y)" % (self.enm,self.name2enm[self.enm][4])
    #             if var_name == "grad_activation":
    #                 return "*(%s_grad_activation+x*%s+y)" % (self.enm,self.name2enm[self.enm][4])
    #             if var_name == "output":
    #                 return "*(%s_output+x*%s+y)" % (self.enm,self.name2enm[self.enm][4])
    #             if var_name == "label":
    #                 return "cur_label[x][y]" 
    #             # inputs does not exists in our c++ code, we need to map
    #             # inputs to previous ensemble's output
    #             if var_name.endswith("inputs"):
    #                 var_name = "%s_output" % self.enm_prev
    #                 return var_name
    #             # transform the AoS to SoA structure
    #             if var_name in self.fields and node.value.id == "self":
    #                 var_name = "%s_%s" % (self.enm, node.attr)
    #         return var_name
    #     if isinstance(node, ast.Index):
    #         return self.parse_var_name(node.value)
    #     if isinstance(node, ast.Subscript):
    #         # find node name and determine whether the variable is a field
    #         var_name = self.parse_var_name(node.value)
    #         field_name = var_name.strip(self.enm + "_")
    #         field_type = "float*"
    #         if field_name.find('[') >= 0:
    #             field_name = field_name[:field_name.find('[')]
    #         if field_name in self.fields:
    #             field_type = self.fields[field_name]
    #
    #         # check it is 1d or 2d array index
    #         if not isinstance(node.value, ast.Subscript):
    #             return self.parse_var_name(node.value)
    #         else:
    #             index_j = self.parse_var_name(node.slice)
    #             index_i = self.parse_var_name(node.value.slice)
    #             # double dimension array index
    #             if field_type == "vector<vector<float*>>":
    #                return "*(%s[x][y]+%s*%s+%s)" % (var_name, index_i,\
    #                        self.name2enm[self.enm_prev][4], index_j)
    #             else:
    #                 return "*(%s+%s*%s+%s)" % (var_name, index_i, self.name2enm[self.enm_prev][4], index_j)
    #

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

def ast2lambda(mapping, args, ensemble):
    mapping = ast.Module(\
        body = [\
            ast.Assign(\
                targets = [ast.Name(id="func", ctx=ast.Store())],\
                value = mapping
            )
        ]
    )
    for arg in args:
        if arg in ensemble:
            mapping = SubstituteNameToNum(arg, ensemble[arg]).visit(mapping)
    mapping = ast.fix_missing_locations(mapping)
    # print ast.dump(mapping)
    codeobj = compile(mapping, '<string>', 'exec')
    exec(codeobj)
    return func

def check_uniform_dependency(args, mapping, ensemble_info):
    mapping = ast2lambda(mapping, args, ensemble_info)
    dim_x, dim_y = ensemble_info['dim_x'], ensemble_info['dim_y']
    dim = 1 if dim_x == 1 else 2
    mapped_indices = sorted(mapping(tuple([ 0 for i in range(dim) ])))
    # print mapped_indices
    for d in range(dim):
        for i in range(dim_x):
            for j in range(dim_y):
                if dim_x == 1:
                    # single dimension
                    if sorted(mapping(j)) != mapped_indices:
                        return False
                else:
                    # double dimension
                    if sorted(mapping(i, j)) != mapped_indices:
                        return False
    return True

def process_add_connection(filename):
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
            conn_types[layer_name] = (args, mapping)

    # review the mapping functions for layer connections
    for layer, (args, mappings) in conn_types.iteritems():
        print ""
        print "layer:", layer
        print "args: ", args
        print "conn: ", ast_dump(mappings)
    print "------------------------------------------"
    return conn_types

def process_lib(filename, ensemble_info, name2enm, PM_FLAG=True):
    """
    read in a library file parse all neuron types,
    and their associated forward/backward functions
    """
    # create neuron analyzer for each neuron subtype
    for neuron_ast in extract_neuron_classes(filename):
        neuron_analyzers[neuron_ast.name] = NeuronAnalyzer(neuron_ast, PM_FLAG)
    
    # process the fields of the neuron base types
    for name, neuron_analyzer in neuron_analyzers.iteritems():
       neuron_analyzer.init_fields()

    # delete unused variables
    print "+++++++++++++++++++++++++++++++++++++++++++"
    for name, neuron_analyzer in neuron_analyzers.iteritems():
       neuron_analyzer.delete_unused_fields()
    print "+++++++++++++++++++++++++++++++++++++++++++"
    
    print "###########################################"
    forward_codes = { }
    backward_codes = { }
    fp_codes = [ ]
    bp_codes = [ ]

    for ensemble in ensemble_info:
        _name, _type, _prev, _dim_x, _dim_y, _neuron_type = ensemble[:6]
        # print neuron_analyzers
        analyzer = neuron_analyzers[_neuron_type]
        fp_code, bp_code = analyzer.analyze(ensemble, name2enm)
        forward_codes[_name] = fp_code
        backward_codes[_name] = bp_code
        fp_codes.append(fp_code)
        bp_codes.append(bp_code)

    return neuron_analyzers, forward_codes, backward_codes, fp_codes, bp_codes

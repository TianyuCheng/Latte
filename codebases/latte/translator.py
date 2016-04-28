import ast
from ast_matcher import *
from templates import *
from copy import deepcopy
from term import *
from structures import *

function_args = {
    "sgemm_dp": [ 3, 1, 1, 1 ],
    "sgemm_axpy": [ 3, 1, 1, 1 ],
    "sgemm_copy": [ 3, 1, 1 ],
    "sgemm_zeros": [ 3, 1 ]
}

def match_forloop(stmt):
    tmpls = [ template_for("range"), template_for("xrange") ]
    for tmpl in tmpls:
        if tmpl.match(stmt):
            return tmpl.wildcard
    return None

class Translator(object):
    """
    Translator translates from Python AST to
    self-defined structure, since Python AST
    is complicated to manipulate
    """
    def __init__(self, neuron_analyzer, curr_enm, prev_enm, conn_type, share_weights, pattern_match=True):
        super(Translator, self).__init__()
        self.neuron_analyzer = neuron_analyzer
        self.curr_enm = curr_enm
        self.prev_enm = prev_enm
        self.statements = []
        # getting the dimension from analyzer for constant replacement
        self.curr_enm_dim = neuron_analyzer.curr_enm_dim()
        self.prev_enm_dim = neuron_analyzer.prev_enm_dim()
        # connections
        self.connection = conn_type
        self.share_weights = share_weights
        # set pattern match flag
        self.pattern_match = pattern_match

    def process_stmt(self, stmt):
        # ignore the stmts for syntax and debug
        if isinstance(stmt, ast.Pass): return
        if isinstance(stmt, ast.Assert): return
        # process stmt of each type
        if isinstance(stmt, ast.Assign):
            return self.process_assign(stmt)
        if isinstance(stmt, ast.For):
            return self.process_for(stmt)
        # We cannot process this statement, so we print it for debug
        term.dump("PROCESS STMT(NO MATCH): %s" % ast.dump(stmt), term.WARNING)

    def process_assign(self, node):
        # pattern match
        if self.pattern_match:
            # try pattern match a bunch of different patterns
            tmpl = template_asgn("output")
            if tmpl.prefix_of(node):
                expr = self.process_node(tmpl.wildcard['exp'])
                return AssignmentNode(ArrayNode(\
                        ConstantNode(self.curr_enm+"_output"), ['x', 'y']), \
                        expr)

            tmpl = template_asgn("grad_activation")
            if tmpl.prefix_of(node):
                expr = self.process_node(tmpl.wildcard['exp'])
                return AssignmentNode(ArrayNode(\
                        ConstantNode(self.curr_enm+"_grad_activation"), ['x', 'y']), \
                        expr)

            tmpl = template_asgn("grad_output")
            if tmpl.prefix_of(node):
                expr = self.process_node(tmpl.wildcard['exp'])
                return AssignmentNode(ArrayNode(\
                        ConstantNode(self.curr_enm+"_grad_output"), ['x', 'y']), \
                        expr)

        # assign node
        var_name = self.process_node(node.targets[0])
        var_value = self.process_node(node.value)
        return AssignmentNode(var_name, var_value)

    def process_for(self, node):
        # pattern match
        if self.pattern_match:
            tmpl = template_dot_product()
            matched = tmpl.match(node) 
            print matched, "==================="
            if matched:
                for x in map(self.process_node, tmpl.wildcard.values()):
                    print x
                A, C, i, B, _, j = map(self.process_node, tmpl.wildcard.values())
                call = CallNode("sgemm_dp")
                call.add_arg(C, 1, 1)
                call.add_arg(A, 1, 0)
                call.add_arg(B, 1, 0)
                call.add_arg(ConstantNode(self.prev_enm_dim[0] * self.prev_enm_dim[1]), 1, 0)
                return call

            tmpl = template_bp_scalar_prod()
            matched = tmpl.match(node) 
            if matched:
                B, _,  _, dim_x, dim_y, scalar = map(self.process_node, tmpl.wildcard.values())
                prev_type = self.neuron_analyzer.prev_enm_type()
                if prev_type is None or prev_type.endswith("DataLayer"):
                    return None
                C = ConstantNode(self.prev_enm + "_grad_output")
                call = CallNode("sgemm_axpy")
                call.add_arg(C, 1, 1)
                call.add_arg(DereferenceNode(scalar), 1, 0)
                call.add_arg(B, 1, 0)
                call.add_arg(ConstantNode(self.prev_enm_dim[0] * self.prev_enm_dim[1]), 1, 0)
                return call

            tmpl = template_bp_axpy()
            matched = tmpl.match(node)
            # print ast.dump(node)
            if matched:
                C, B, di, dj, scalar, dim_x, dim_y = map(self.process_node, tmpl.wildcard.values())
                call = CallNode("sgemm_axpy")
                call.add_arg(C, 1, 1)
                call.add_arg(DereferenceNode(scalar), 1, 0)
                call.add_arg(B, 1, 0)
                call.add_arg(ConstantNode(self.prev_enm_dim[0] * self.prev_enm_dim[1]), 1, 0)
                return call

        # ---------------------------------------------------------------------

        # match for-backward-adj loop
        tmpl = template_for_backward_adj()
        if tmpl.match(node):
            result = tmpl.wildcard
            elt, indices = self.process_adjacency([])
            index = indices[elt[0].get_constant()]
            for_i = ForNode(ConstantNode("i"), ConstantNode(index[0]),\
                            ConstantNode(index[1]), ConstantNode(index[2]))
            index = indices[elt[1].get_constant()]
            for_j = ForNode(ConstantNode("j"), ConstantNode(index[0]),\
                            ConstantNode(index[1]), ConstantNode(index[2]))
            for_i.add_child(for_j)
            for stmt in result['body']:
                for_j.add_child(self.process_stmt(stmt))
            return for_i

        # ---------------------------------------------------------------------

        # match for-range loop
        result = match_forloop(node)
        if result is None:
            term.dump("PROCESS FOR STMT(ERROR): %s" % ast.dump(node), term.FAIL)
            return
        # extract information from forloop
        initial_name = self.process_node(result['i'])
        initial = ConstantNode(0)
        loop_bound = self.process_node(result['N'])
        increment = ConstantNode(1)
        # print loop_bound
        # print result['N']
        for_node = ForNode(initial_name, initial, loop_bound, increment)
        # append the inner codes to the loop
        codes = result['body']
        if not isinstance(codes, list):
            codes = [ codes ]
        for stmt in codes:
            for_node.add_child(self.process_stmt(stmt))
        return for_node

    def process_node(self, node):
        # parse the variable name, and create structure
        if isinstance(node, str) or isinstance(node, int) or \
           isinstance(node, float):
            return ConstantNode(node)
        if isinstance(node, ast.Num):
            return ConstantNode(node.n)
        if isinstance(node, ast.Name):
            return ConstantNode(node.id)
        if isinstance(node, ast.BinOp):
            l = self.process_node(node.left)
            r = self.process_node(node.right)
            op = self.process_op(node.op)
            return ExpressionNode(l, r, op)
        if isinstance(node, ast.Subscript):
            return self.process_subscript(node)
        if isinstance(node, ast.Index):
            return self.process_node(node.value)
        if isinstance(node, ast.Attribute):
            return self.process_attribute(node)
        if isinstance(node, ast.Call):
            return self.process_call(node)
        term.dump("PROCESS NODE(NO MATCH): %s" % ast.dump(node), term.WARNING)

    def process_call(self, node):
        func_name = self.process_node(node.func)
        func_args = map(self.process_node, node.args)
        node = CallNode(func_name)
        # default all arguments are read only
        arg_flags = [ 1 ] * len(func_args)
        # check if the function falls in our special functions: sgemm
        if func_name in function_args:
            arg_flags = function_args[func_name]
            assert len(arg_flags) == len(func_args)
        # fill the CallNode with arguments
        for arg, flag in zip(func_args, arg_flags):
            node.add_arg(arg, flag&0x1, flag&0x2)
        return node

    def process_subscript(self, node):
        array_name = self._find_array_name(node)
        array_idx = self._find_array_index(node)
        # return IndexNode(array_name, array_idx, self.prev_enm_dim[1])
        return IndexNode(array_name, array_idx, self.curr_enm_dim[1])

    def _find_array_name(self, node):
        if isinstance(node, ast.Name):
            return ConstantNode(node.id)
        if isinstance(node, ast.Attribute):
            return self.process_attribute(node)
        assert isinstance(node, ast.Subscript)
        return self._find_array_name(node.value)

    def _find_array_index(self, node):
        if isinstance(node, ast.Name):
            return self.process_node(node)
        if isinstance(node, ast.Index):
            return self._find_array_name(node.value)
        assert isinstance(node, ast.Subscript)
        if isinstance(node.value, ast.Subscript):
            return self._find_array_index(node.value) + [ self.process_node(node.slice) ]
        return [ self._find_array_index(node.slice) ]

    def process_op(self, op):
        if isinstance(op, ast.Add):
            return "+"
        elif isinstance(op, ast.Sub):
            return "-"
        elif isinstance(op, ast.Mult):
            return "*"
        elif isinstance(op, ast.Div):
            return " / "
        elif isinstance(op, ast.Pow):
            return "pow"
        term.dump("PROCESS OP(INVALID OP): %s" % op, term.FAIL)

    def process_attribute(self, node):
        owner = self.process_node(node.value)
        attr = node.attr
        if str(owner) == "self":
            # built-in dimension analysis
            if attr == "prev_dim_x":
                return ConstantNode(self.prev_enm_dim[0])
            if attr == "prev_dim_y":
                return ConstantNode(self.prev_enm_dim[1])
            if attr == "dim_x":
                return ConstantNode(self.curr_enm_dim[0])
            if attr == "dim_y":
                return ConstantNode(self.curr_enm_dim[1])

            #############################################
            # replace inputs with outputs from last layer
            # this is done to fit the current design
            # we might want to follow the paper and do
            # input copy from last layer if the inputs
            # are not shared
            if attr.endswith("inputs"):
                var_name = "%s_output" % self.prev_enm
                return ConstantNode(var_name)
            #############################################

            if attr.endswith("label"):
                return ArrayNode(ConstantNode("cur_label"), ['x', 'y'])

            # analyze field type
            field_type = self.neuron_analyzer.get_field_type(attr)
            if field_type is None:
                return ConstantNode(self.curr_enm + "_" + attr)
            elif field_type == "vector<vector<float*>>":
                # 2D fields
                var_name = "%s_%s" % (self.curr_enm, attr)
                return ArrayNode(\
                        ConstantNode(var_name), ['x', 'y'])
            else:
                # 1D fields
                # transform to SoA form
                var_name = "%s_%s" % (self.curr_enm, attr)
                return IndexNode(\
                        ConstantNode(var_name), ['x', 'y'], \
                        self.curr_enm_dim[1])
        elif str(owner) == "prev":
            if attr == "pos_x":
                return ConstantNode("i")
            elif attr == "pos_y":
                return ConstantNode("j")
            else:
                return ConstantNode(self.prev_enm + "_" + node.attr)
        else:
            # calls like np.tanh, suffice to only return the attr
            return ConstantNode(node.attr)

    def process_adjacency(self, stmts):
        if self.connection is None: return
        assert isinstance(self.connection, ast.Lambda)
        assert isinstance(self.connection.body, ast.ListComp)
        args = self.connection.args
        args = map(self.process_node, args.args)
        body = self.connection.body
        elt = map(self.process_node, body.elt.elts)
        gen = body.generators
        assert len(gen) == 2
        assert isinstance(gen[0].target, ast.Name)
        assert isinstance(gen[1].target, ast.Name)
        assert isinstance(gen[0].iter, ast.Call)
        assert isinstance(gen[1].iter, ast.Call)
        assert gen[0].iter.func.id == "range"
        assert gen[1].iter.func.id == "range"
        indices = {
            gen[0].target.id: map(lambda x: x.n, gen[0].iter.args),
            gen[1].target.id: map(lambda x: x.n, gen[1].iter.args)
        }
        # transform the loop bound into range format
        for key in indices.iterkeys():
            if len(indices[key]) == 1:
                indices[key] = [0] + indices[key] + [1]
        return elt, indices

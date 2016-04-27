import ast
from ast_matcher import *
from templates import *
from copy import deepcopy
from term import *
from structures import *

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
    def __init__(self, neuron_analyzer, curr_enm, prev_enm):
        super(Translator, self).__init__()
        self.neuron_analyzer = neuron_analyzer
        self.curr_enm = curr_enm
        self.prev_enm = prev_enm
        self.statements = []
        # getting the dimension from analyzer for constant replacement
        self.curr_enm_dim = neuron_analyzer.curr_enm_dim()
        self.prev_enm_dim = neuron_analyzer.prev_enm_dim()

    def process_stmt(self, stmt):
        # ignore the stmts for syntax and debug
        if isinstance(stmt, ast.Pass): return
        if isinstance(stmt, ast.Assert): return
        # process stmt of each type
        if isinstance(stmt, ast.Assign):
            # assign node
            var_name = self.process_node(stmt.targets[0])
            var_value = self.process_node(stmt.value)
            return AssignmentNode(var_name, var_value)
        if isinstance(stmt, ast.For):
            # for node
            result = match_forloop(stmt)
            if result is None:
                term.dump("PROCESS FOR STMT(ERROR): %s" % ast.dump(stmt), term.FAIL)
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
        # We cannot process this statement, so we print it for debug
        term.dump("PROCESS STMT(NO MATCH): %s" % ast.dump(stmt), term.WARNING)

    def process_node(self, node):
        # parse the variable name, and create structure
        if isinstance(node, str) or isinstance(node, int) or \
           isinstance(node, float):
            return node
        if isinstance(node, ast.Num):
            return ConstantNode(node.n)
        if isinstance(node, ast.Name):
            return node.id
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
        # node = CallNode()
        return "call"

    def process_subscript(self, node):
        array_name = self._find_array_name(node)
        array_idx = self._find_array_index(node)
        return "subscript"

    def _find_array_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
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
        if owner == "self":
            # built-in dimension analysis
            if attr == "prev_dim_x":
                return ConstantNode(self.prev_enm_dim[0])
            if attr == "prev_dim_y":
                return ConstantNode(self.prev_enm_dim[1])
            if attr == "dim_x":
                return ConstantNode(self.curr_enm_dim[0])
            if attr == "dim_y":
                return ConstantNode(self.curr_enm_dim[1])

            # replace inputs with outputs from last layer
            # this is done to fit the current design
            # we might want to follow the paper and do
            # input copy from last layer if the inputs
            # are not shared
            if attr.endswith("inputs"):
                var_name = "%s_output" % self.prev_enm
                return var_name

            # transform to SoA form
            var_name = "%s_%s" % (self.curr_enm, attr)
            return var_name
        else:
            # calls like np.tanh, suffice to only return the attr
            return node.attr

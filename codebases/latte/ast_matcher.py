from copy import deepcopy
import ast, inspect

ENABLE_AST_MATCH = True

def set_ast_match(ast_match_flag):
    global ENABLE_AST_MATCH
    ENABLE_AST_MATCH = ast_match_flag

class BinOpCounter(ast.NodeVisitor):
    def __init__(self):
        super(BinOpCounter, self).__init__()
        self.count = 0

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Add) or isinstance(node.op, ast.Mult):
            self.count += 1

class ReorderByList(ast.NodeTransformer):
    def __init__(self, reorderList):
        super(ReorderByList, self).__init__()
        self.count = 0
        self.reorderList = reorderList

    def visit_BinOp(self, node):
        self.generic_visit(node)
        # change the order or add/mul since they are commutative
        if isinstance(node.op, ast.Add) or isinstance(node.op, ast.Mult):
            if self.reorderList[self.count]:
                node.left, node.right = node.right, node.left
            self.count += 1
        return node

class ReorderBinOp(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        # change the order or add/mul since they are commutative
        if isinstance(node.op, ast.Add) or isinstance(node.op, ast.Mult):
            # put the number operand on the left
            if isinstance(node.right, ast.Num) and not isinstance(node.left, ast.Num):
                node.left, node.right = node.right, node.left
        return node

class ExpandAugAssign(ast.NodeTransformer):
    def visit_AugAssign(self, node):
        self.generic_visit(node)
        assign_node = ast.Assign( \
            targets = [ node.target ], \
            value = ast.BinOp( \
                left = node.target,   \
                op = node.op,         \
                right = node.value    \
            )
        )
        return assign_node

class RewriteName(ast.NodeTransformer):
    """change name of a node"""
    def __init__(self, old_name, new_name):
        super(RewriteName, self).__init__()
        self.old_name = old_name
        self.new_name = new_name

    def visit_Name(self, node):
        self.generic_visit(node)
        if node.id == self.old_name:
            node.id = self.new_name
        return node

def ast_parse_file(filename):
    f = open(filename, "r")
    AST = ast.parse(f.read())
    AST = ExpandAugAssign().visit(AST)
    AST = ReorderBinOp().visit(AST)
    f.close()
    return AST

def ast_parse_source(src):
    AST = ast.parse(src)
    AST = ExpandAugAssign().visit(AST)
    AST = ReorderBinOp().visit(AST)
    return AST

def ast_dump(AST):
    if isinstance(AST, str):
        return AST
    if isinstance(AST, list):
        return '\n'.join(AST)
    return ast.dump(AST)

def stmt_walk(node):
    """generator function: grab statements 1 by 1"""
    # if node is a module or a for loop
    if isinstance(node, ast.Module) or \
       isinstance(node, ast.For) or \
       isinstance(node, ast.FunctionDef) or \
       isinstance(node, ast.If):
        for stmt in node.body:
            yield stmt

def template(tmpl_func):
    """wrapper around templates"""
    def template_wrapper(*args):
        source = inspect.getsource(tmpl_func)
        return ASTTemplate(source, *args)

    return template_wrapper

class ASTTemplate(object):
    """ASTTemplate"""
    def __init__(self, source, *args):
        super(ASTTemplate, self).__init__()

        self.ast = ast_parse_source(source)
        self.ast = self.ast.body[0]                     # find wrapper function
        self.num_stmts = len(self.ast.body)
        
        # preprocessing: replace arguments
        assert len(args) == len(self.ast.args.args)

        for i, arg_node in enumerate(self.ast.args.args):
            old_name = arg_node.id
            new_name = str(args[i])
            self.ast = RewriteName(old_name, new_name).visit(self.ast)
        
        self.fname = self.ast.name                       # find template name

        self.asts = []
        # get number of add/mul operations
        # compute all the permutations
        visitor = BinOpCounter()
        visitor.visit(self.ast)
        self._gen_all_asts([], visitor.count)

        # # try printing all permuted asts
        # for tpl_ast in self.asts:
        #     for tpl_stmt in tpl_ast:
        #         print ast.dump(tpl_stmt)
        #     print "------------"

    def _gen_all_asts(self, reorderList, remain):
        if remain == 0:
            self.asts.append(ReorderByList(reorderList).visit(deepcopy(self.ast)).body)
            return
        # recursively generate all asts
        self._gen_all_asts(reorderList + [ False ], remain - 1)
        self._gen_all_asts(reorderList + [ True  ], remain - 1)

    def __str__(self):
        return self.fname.strip("template_")

    def matchall(self, tgt):
        self.matches = []

        # first try matching tgt as a whole
        if self.match(tgt):
            self.matches.append(self.wildcard)

        # then try matching each stmt in tgt
        for stmt in stmt_walk(tgt):
            if self.match(stmt):
                self.matches.append(self.wildcard)

        return len(self.matches) > 0

    def prefix_of(self, tgt):
        tgt_num_stmts = 1
        if isinstance(tgt, list):
            tgt_num_stmts = len(tgt)
        elif "body" in tgt._fields:
            tgt_num_stmts = len(tgt.body)
            tgt = tgt.body
        else:
            tgt_num_stmts = 1
            tgt = [ tgt ]
        if tgt_num_stmts < self.num_stmts:
            return False
        return self.match(tgt[:self.num_stmts])

    def match(self, tgt):
        if not ENABLE_AST_MATCH:
            return False
        # explicitly expand target from AugAssign to Assign
        # for semantic matching
        if isinstance(tgt, list):
            for i, t in enumerate(tgt):
                tgt[i] = ExpandAugAssign().visit(t)
        else:
            tgt = ExpandAugAssign().visit(tgt)

        """ match ast with template """
        for tpl_ast in self.asts:
            # see if it matches with any of the possible combinations of
            # asts
            self.wildcard = dict()
            if self._match(tpl_ast, tgt):
                return True
        return False

    def _match(self, tpl, tgt):
        """ match helper function """
        if self._set_wildcard(tpl, tgt):
            return True

        if isinstance(tpl, str) and isinstance(tgt, str):
            # direct string comparison
            return tpl == tgt

        if isinstance(tpl, bool) and isinstance(tgt, bool):
            # direct boolean comparison
            return tpl == tgt

        if isinstance(tpl, int) and isinstance(tgt, int):
            # direct int comparison
            return tpl == tgt

        if isinstance(tpl, float) and isinstance(tgt, float):
            # direct float comparison
            return tpl == tgt

        # deal with wrappers
        if isinstance(tgt, ast.Module):
            tgt = tgt.body

        if not isinstance(tgt, list):
            tgt = [ tgt ]
            
        if not isinstance(tpl, list):
            tpl = [ tpl ]
        
        # check number of stmts
        if len(tpl) != len(tgt):
            # print "1"
            return False
        
        for i in xrange(len(tpl)):
            tpl_node = tpl[i]
            tgt_node = tgt[i]
            if self._set_wildcard(tpl_node, tgt_node):
                continue
            else:
                # check node match types
                if type(tpl_node) != type(tgt_node):
                    # print "2"
                    # print tpl_node
                    # print tgt_node
                    return False

                # avoid comparing None
                if tpl_node is None:
                    continue

                # compare fields
                tpl_kids = list(ast.iter_fields(tpl_node))
                tgt_kids = list(ast.iter_fields(tgt_node))
                if len(tpl_kids) != len(tgt_kids):
                    # print "3"
                    return False
                for i in xrange(len(tpl_kids)):
                    _, tpl_value = tpl_kids[i]
                    _, tgt_value = tgt_kids[i]
                    if not self._match(tpl_value, tgt_value):
                        # print "4"
                        return False
        return True

    def _set_wildcard(self, tpl, tgt):
        # using one wildcard to match a list of argument
        if isinstance(tpl, ast.Call) and isinstance(tgt, ast.Call) and \
            tpl.func.id == tgt.func.id and len(tpl.args) != len(tgt.args) and \
            len(tpl.args) == 1 and isinstance(tpl.args[0], ast.Name):
            return self._set_wildcard(tpl.args[0], tgt.args)

        # match dangling expression
        if isinstance(tpl, ast.Expr) and isinstance(tpl.value, ast.Name):
            return self._set_wildcard(tpl.value, tgt)

        # match wildcard variable
        if isinstance(tpl, ast.Name) and tpl.id.startswith('_') and \
                len(tpl.id) > 1:
            value = None
            # wildcard matching
            if isinstance(tgt, ast.Name):
                value = tgt.id
            elif isinstance(tgt, ast.Num):
                value = tgt.n
            elif isinstance(tgt, ast.Str):
                value = tgt.s
            else:
                value = tgt
            # check if this wildcard has already been set
            if tpl.id[1:] in self.wildcard:
                # if ast_dump(self.wildcard[tpl.id[1:]]) != ast_dump(value):
                #     print "-------------------------------------->"
                #     print ast_dump(self.wildcard[tpl.id[1:]])
                #     print ast_dump(value)
                #     print "-------------------------------------->"
                return ast_dump(self.wildcard[tpl.id[1:]]) == ast_dump(value)
            else:
                self.wildcard[tpl.id[1:]] = value
                return True
            # self.wildcard[tpl.id[1:]] = value
            # return True

        return False

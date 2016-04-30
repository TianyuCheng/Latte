"""Tree representation of for loops for analysis/tiling/fusion purposes"""
class Node(object):
    def __init__(self):
        # should be extended from
        self.parent = None
        self.child_number = -1
        self.children = []

    def add_child(self, child):
        """Add a child to this node"""
        if child is None:
            return

        child_number = len(self.children)
        self.children.append(child)

        # set number and parent
        child.set_child_number(child_number)
        child.set_parent(self)
    
    def set_parent(self, p):
        self.parent = p

    def get_parent(self):
        return self.parent

    def set_child_number(self, num):
        self.child_number = num

    def get_child_number(self):
        return self.child_number

    def replace_node(self, other_node):
        """Given another node, take the place of that node in the tree
        by becoming the child of whatever its parent was and making it
        a child of ourself"""
        other_child_number = other_node.get_child_number()
        other_parent = other_node.get_parent()

        if not other_parent == None:
            # replace the other child with self
            other_parent.substitute_child(other_child_number, self)

        # make the other node a child of this one
        self.add_child(other_node)


    def substitute_child(self, child_number, new_child):
        """Overwrite the child at child_number with a new child"""
        # get old child
        old_child = self.children[child_number]

        # make it a child of no one
        old_child.set_child_number(-1)
        old_child.set_parent(None)

        # overwrite
        self.children[child_number] = new_child
        # set child number and parent of new child
        new_child.set_child_number(child_number)
        new_child.set_parent(self)

    def get_children(self):
        """return list of children"""
        return self.children


class ForNode(Node):
    """Holds information for a for loop"""
    def __init__(self, initial_name, initial, loop_bound, increment):
        super(ForNode, self).__init__()

        # save initial variables
        self.initial_name = initial_name
        self.initial = initial
        self.loop_bound = loop_bound
        self.increment = increment

    def set_initial(self, i):
        self.initial = i

    def set_initial_name(self, i):
        self.initial_name = i

    def set_loop_bound(self, i):
        self.loop_bound = i

    def set_increment(self, i):
        self.increment = i

    def get_initial(self):
        return self.initial

    def get_initial_name(self):
        return self.initial_name

    def get_loop_bound(self):
        return self.loop_bound

    def get_increment(self):
        return self.increment

    def deep_copy(self):
        """returns a node that is a copy of this node"""
        my_copy = ForNode(self.initial_name, self.initial, self.loop_bound,
                          self.increment)

        for child in self.children:
            child_copy = child.deep_copy()
            my_copy.add_child(child_copy)

        return my_copy

    def __str__(self):
        """Prints the ENTIRE loop including its children"""
        for_fmt = "for (int {i} = {initial}; {i} < {bound}; {i} += {increment}) {{\n{code}\n}}"
        return for_fmt.format(i=self.initial_name, \
                              initial=str(self.initial), \
                              bound=str(self.loop_bound), \
                              increment=str(self.increment), \
                              code='\n'.join(map(str, self.children)))

class ConstantNode(Node):
    """Holds a constant, whether it be a number or a variable name"""
    def __init__(self, constant):
        super(ConstantNode, self).__init__()

        # holds the constant
        self.constant = constant

    def get_constant(self):
        return self.constant

    def deep_copy(self):
        my_copy = ConstantNode(self.constant)

        # shouldn't have children, but whatever
        for child in self.children:
            child_copy = child.deep_copy()
            my_copy.add_child(child_copy)

        return my_copy


    #def is_used(self, use, use_list):
    #    """Adds our constant to a use list if the use is equal to what is
    #    being asked about"""
    #    if use == self.constant:
    #        use_list.append(use)

    def __str__(self):
        return str(self.constant)


class AssignmentNode(Node):
    """Top level: holds an assignment statement: needs a left and a right"""
    def __init__(self, left, right):
        super(AssignmentNode, self).__init__()

        # left and right are nodes
        self.left = left
        self.right = right

        if isinstance(self.left, IndexNode):
            self.left = DereferenceNode(self.left)
        if isinstance(self.right, IndexNode):
            self.right = DereferenceNode(self.right)

    def deep_copy(self):
        my_copy = AssignmentNode(self.left.deep_copy(), self.right.deep_copy())

        # shouldn't have children
        for child in self.children:
            child_copy = child.deep_copy()
            my_copy.add_child(child_copy)

        return my_copy

    def __str__(self):
        return "%s = %s;" % (str(self.left), str(self.right))


class ExpressionNode(Node):
    """Holds an expression (i.e. binary op): a left expression, and operator, then the right
    expression"""
    def __init__(self, left, right, operator):
        super(ExpressionNode, self).__init__()

        # a node
        self.left = left
        # NOTE operator should be a + or a *
        self.operator = operator
        # a node
        self.right = right

        if isinstance(self.left, IndexNode):
            self.left = DereferenceNode(self.left)
        if isinstance(self.right, IndexNode):
            self.right = DereferenceNode(self.right)

    def deep_copy(self):
        my_copy = ExpressionNode(self.left.deep_copy(), self.right.deep_copy(), 
                                 self.operator)

        # shouldn't have children
        for child in self.children:
            child_copy = child.deep_copy()
            my_copy.add_child(child_copy)

        return my_copy

    def __str__(self):
        if self.operator == "pow":
            return "%s(%s, %s)" % (self.operator, str(self.left), str(self.right))
        return "(%s %s %s)" % (str(self.left), self.operator, str(self.right))


class ArrayNode(Node):
    def __init__(self, base_addr, indices):
        super(ArrayNode, self).__init__()
        self.base_addr = base_addr
        # indices could be multiple dimensions
        # they are stored in a list in the natural order,
        # e.g. [ i, j ]
        if isinstance(indices, list):
            self.indices = indices
        else:
            self.indices = [ indices ]

    def deep_copy(self):
        my_copy = ArrayNode(self.base_addr, self.indices[:])

        for child in self.children:
            child_copy = child.deep_copy()
            my_copy.add_child(child_copy)

        return my_copy

    def __str__(self):
        indices = ''.join(map(lambda x: "[%s]" % str(x), self.indices))
        return "%s%s" % (self.base_addr, indices)


class IndexNode(Node):
    """Purpose of this is to store pointer arithmetic expressions, i.e. i*10 + j.
    Works for 2D pointer arithmetic at most"""
    def __init__(self, base_addr, indices, stride=1):
        super(IndexNode, self).__init__()
        # indices could be multiple dimensions
        # they are stored in a list in the natural order,
        # e.g. [ i, j ]
        self.base_addr = base_addr
        self.stride = stride

        if isinstance(indices, list):
            self.indices = indices
        else:
            self.indices = [ indices ]

        assert len(self.indices) <= 2

    def deep_copy(self):
        my_copy = IndexNode(self.base_addr, self.indices[:], self.stride)

        for child in self.children:
            child_copy = child.deep_copy()
            my_copy.add_child(child_copy)

        return my_copy

    def __str__(self):
        if len(self.indices) == 1:
            # single dimension pointer arithmetic
            return "(%s+%s)" % (self.base_addr, str(self.indices[0]))
        else:
            return "(%s+%s*%d+%s)" % (self.base_addr, \
                    str(self.indices[0]), self.stride, str(self.indices[1]))


class DereferenceNode(Node):
    def __init__(self, node):
        super(DereferenceNode, self).__init__()
        self.add_child(node)

    def deep_copy(self):
        my_copy = DereferenceNode(self.children[0].deep_copy())

        return my_copy

    def __str__(self):
        return "(*%s)" % str(self.children[0])

class GetPointerNode(Node):
    def __init__(self, node):
        super(GetPointerNode, self).__init__()
        self.add_child(node)

    def deep_copy(self):
        my_copy = GetPointerNode(self.children[0].deep_copy())

        return my_copy

    def __str__(self):
        return "(&%s)" % str(self.children[0])
        

class CallNode(Node):
    """Represents calls to functions. May have multiple arguments. Should also
    specify which arguments are read from/written to"""
    def __init__(self, func):
        super(CallNode, self).__init__()
        # call attributes
        self.func = func
        self.args_rw = []

    def add_arg(self, arg, read, write):
        if not self.func.get_constant().startswith("sgemm"):
            if isinstance(arg, IndexNode):
                arg = DereferenceNode(arg)
        self.add_child(arg)
        # set flag for arg read/write
        rw = 0
        if read:  rw |= 0x1
        if write: rw |= 0x2
        self.args_rw.append(rw)

    def deep_copy(self):
        my_copy = CallNode(self.func)

        my_copy.args_rw = self.args_rw[:]

        for child in self.children:
            child_copy = child.deep_copy()
            my_copy.add_child(child_copy)

        return my_copy

    def __str__(self):
        args = ', '.join(map(str, self.children))
        if isinstance(self.parent, ForNode):
            return "%s(%s);" % (self.func, args)
        return "%s(%s)" % (self.func, args)

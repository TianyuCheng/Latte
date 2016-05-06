import copy

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

        # make sure it's a Node
        if not isinstance(child, Node):
            raise Exception("Trying to add non-node as child")

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

    def copy(self, to_copy):
        """If to_copy is a node, do a deep copy, else just return it"""
        if isinstance(to_copy, Node):
            return to_copy.deep_copy()
        else:
            return to_copy

class ListNode(Node):
    """Type of node that needs access to the list find and replace
    method"""
    def __init__(self):
        super(ListNode, self).__init__()

    def list_find_and_replace(self, the_list, to_find, replacement):
        to_return = []

        for i in range(len(the_list)):
            element = the_list[i]

            if isinstance(element, list):
                # recursive call on the list
                new_list = list_find_and_replace(element, to_find, replacement)

                to_return.append(new_list)
            else:
                # single element, not a list: check and replace
                if element == to_find:
                    to_return.append(replacement)
                else:
                    to_return.append(element)

        return to_return 


class ForNode(Node):
    """Holds information for a for loop"""
    def __init__(self, initial_name, initial, loop_bound, increment):
        super(ForNode, self).__init__()

        if not (isinstance(initial_name, Node) and
                isinstance(initial, Node) and
                isinstance(loop_bound, Node) and
                isinstance(increment, Node)):
            raise Exception("Everything pass into ForNode must be a node")

        # save initial variables
        self.initial_name = initial_name
        self.initial = initial
        self.loop_bound = loop_bound
        self.increment = increment

    def set_initial(self, i):
        self.initial.set_constant(i)

    def set_initial_name(self, i):
        self.initial_name.set_constant(i)

    def set_loop_bound(self, i):
        self.loop_bound.set_constant(i)

    def set_increment(self, i):
        self.increment.set_constant(i)

    def get_initial(self):
        return self.initial.get_constant()

    def get_initial_name(self):
        return self.initial_name.get_constant()

    def get_loop_bound(self):
        return self.loop_bound.get_constant()

    def get_increment(self):
        return self.increment.get_constant()

    def deep_copy(self):
        """returns a node that is a copy of this node"""
        my_copy = ForNode(self.initial_name.deep_copy(), 
                          self.initial.deep_copy(), 
                          self.loop_bound.deep_copy(),
                          self.increment.deep_copy())

        for child in self.children:
            child_copy = child.deep_copy()
            my_copy.add_child(child_copy)

        return my_copy

    def find_and_replace(self, to_find, replacement):
        """look through self and all children to replace something"""
        # check through our stuff to see if anything needs to be changed
        self.initial_name.find_and_replace(to_find, replacement)
        self.initial.find_and_replace(to_find, replacement)
        self.loop_bound.find_and_replace(to_find, replacement)
        self.increment.find_and_replace(to_find, replacement)

        # check through children
        for child in self.children:
            child.find_and_replace(to_find, replacement)

    def get_writes(self):
        variable_names = []
        array_accesses = []

        for child in self.children:
            v, a = child.get_writes()

            variable_names = variable_names + v
            array_accesses = array_accesses + a

        return variable_names, array_accesses

    def get_reads(self):
        variable_names = []
        array_accesses = []

        for child in self.children:
            v, a = child.get_reads()

            variable_names = variable_names + v
            array_accesses = array_accesses + a

        return variable_names, array_accesses

          
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

    def set_constant(self, new_value):
        self.constant = new_value

    def find_and_replace(self, to_find, replacement):
        if self.constant == to_find:
            self.constant = replacement

    def is_var(self):
        """returns if the constant is a var; i.e. a string"""
        return isinstance(self.constant, str)

    def deep_copy(self):
        my_copy = ConstantNode(self.constant)

        # shouldn't have children, but whatever
        for child in self.children:
            child_copy = child.deep_copy()
            my_copy.add_child(child_copy)

        return my_copy

    def get_use(self):
        """assumes constant nodes have no children"""
        if self.is_var():
            # only matters if we represent a variable
            return [self.constant], []
        else:
            return [], []

    def get_writes(self):
        return self.get_use()

    def get_reads(self):
        return self.get_use()

    def __str__(self):
        return str(self.constant)

class AssignmentNode(Node):
    """Top level: holds an assignment statement: needs a left and a right"""
    def __init__(self, left, right):
        super(AssignmentNode, self).__init__()

        # left right better be nodes
        if not (isinstance(left, Node) and
                isinstance(right, Node)):
            raise Exception("Everything passed into Assignment must be a node")

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

    def find_and_replace(self, to_find, replacement):
        self.left.find_and_replace(to_find, replacement)
        self.right.find_and_replace(to_find, replacement)

        for child in self.children:
            child.find_and_replace(to_find, replacement)

    def get_writes(self):
        """assumes no children"""
        # only left hand side is written to
        variable_names, array_accesses = self.left.get_writes()

        return variable_names, array_accesses

    def get_reads(self):
        """assumes no children"""
        # only right hand side is read
        variable_names, array_accesses = self.right.get_reads()

        return variable_names, array_accesses

    def __str__(self):
        return "%s = %s;" % (str(self.left), str(self.right))


class ExpressionNode(Node):
    """Holds an expression (i.e. binary op): a left expression, and operator, then the right
    expression"""
    def __init__(self, left, right, operator):
        super(ExpressionNode, self).__init__()

        # left right better be nodes
        if not (isinstance(left, Node) and
                isinstance(right, Node)):
            raise Exception("left right in Expression must be nodes")

        # a node
        self.left = left
        self.right = right

        self.operator = operator

        if isinstance(self.left, IndexNode):
            self.left = DereferenceNode(self.left)
        if isinstance(self.right, IndexNode):
            self.right = DereferenceNode(self.right)

    def deep_copy(self):
        my_copy = ExpressionNode(self.left.deep_copy(), self.right.deep_copy(), 
                                 self.copy(self.operator))

        # shouldn't have children, but whatever
        for child in self.children:
            child_copy = child.deep_copy()
            my_copy.add_child(child_copy)

        return my_copy

    def find_and_replace(self, to_find, replacement):
        self.left.find_and_replace(to_find, replacement)
        self.right.find_and_replace(to_find, replacement)

        for child in self.children:
            child.find_and_replace(to_find, replacement)

    def get_writes(self):
        """expressions may have function calls as 1 of the operands; if
        so, call get writes on the function call"""
        variable_names = []
        array_accesses = []

        # if operand is a call, get the writes from the call
        if isinstance(self.left, CallNode):
            v, a = self.left.get_writes()
            variable_names = variable_names + v
            array_accesses = array_accesses + a

        if isinstance(self.right, CallNode):
            v, a = self.right.get_writes()
            variable_names = variable_names + v
            array_accesses = array_accesses + a
        # otherwise we don't want to call get writes since usually
        # operands are only read from
        
        return variable_names, array_accesses

    def get_reads(self):
        # an expression reads its left and right operands: call get reads on
        # both of them
        variable_names = []
        array_accesses = []

        v, a = self.left.get_reads()
        variable_names = variable_names + v
        array_accesses = array_accesses + a

        v, a = self.right.get_reads()
        variable_names = variable_names + v
        array_accesses = array_accesses + a

        return variable_names, array_accesses


    def __str__(self):
        if self.operator == "pow":
            return "%s(%s, %s)" % (self.operator, str(self.left), str(self.right))
        return "(%s %s %s)" % (str(self.left), self.operator, str(self.right))


class ArrayNode(ListNode):
    def __init__(self, base_addr, indices):
        super(ArrayNode, self).__init__()

        self.base_addr = base_addr

        #TODO amke sure everything in indices is a list? 
        # (right now it's assumed)

        # indices could be multiple dimensions
        # they are stored in a list in the natural order,
        # e.g. [ i, j ]
        if isinstance(indices, list):
            self.indices = indices
        else:
            self.indices = [ indices ]

    def deep_copy(self):
        my_copy = ArrayNode(self.copy(self.base_addr), copy.deepcopy(self.indices))

        for child in self.children:
            child_copy = child.deep_copy()
            my_copy.add_child(child_copy)

        return my_copy


    def find_and_replace(self, to_find, replacement):
        #TODO check?
        if isinstance(self.base_addr, Node):
            self.base_addr.find_and_replace(to_find, replacement)
        else:
            if self.base_addr == to_find:
                self.base_addr = replacement
        
        # deal with indices
        self.indices = self.list_find_and_replace(self.indices, 
                            to_find, replacement)

        for child in self.children:
            child.find_and_replace(to_find, replacement)

    def get_base_addr(self):
        return self.base_addr

    def get_indices(self):
        return self.indices

    def get_use(self):
        base_string = self.base_addr
        indices = []

        # it's possible that the base address could be another array node;
        # I'm told this only goes at most 1 level deep
        if isinstance(self.base_addr, ArrayNode):
            base_string = self.base_addr.get_base_addr()
            indices = copy.deepcopy(self.base_addr.get_indices())

        indices = indices + copy.deepcopy(self.indices)

        while isinstance(base_string, ConstantNode):
            base_string = base_string.get_constant()

        # returns nothing for a variable name, but returns a tuple
        # with the array name first then a list of the indices
        return [], [(base_string, indices)]

    def get_writes(self):
        return self.get_use()

    def get_reads(self):
        return self.get_use()

    def __str__(self):
        indices = ''.join(map(lambda x: "[%s]" % str(x), self.indices))
        return "%s%s" % (self.base_addr, indices)


class IndexNode(ListNode):
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

        # at this point only supports 2D arithmetic
        assert len(self.indices) <= 2

    def deep_copy(self):
        my_copy = IndexNode(self.copy(self.base_addr), 
                            copy.deepcopy(self.indices), 
                            self.copy(self.stride))

        for child in self.children:
            child_copy = child.deep_copy()
            my_copy.add_child(child_copy)

        return my_copy

    def find_and_replace(self, to_find, replacement):
        if isinstance(self.base_addr, Node):
            self.base_addr.find_and_replace(to_find, replacement)
        else:
            if self.base_addr == to_find:
                self.base_addr = replacement

        if isinstance(self.stride, Node):
            self.stride.find_and_replace(to_find, replacement)
        else:
            if self.stride == to_find:
                self.stride = replacement
        
        # deal with indices
        self.indices = self.list_find_and_replace(self.indices, 
                            to_find, replacement)

        # shouldn't have children, but just in case
        for child in self.children:
            child.find_and_replace(to_find, replacement)

    def get_use(self):
        array_name = self.base_addr
        while isinstance(array_name, ConstantNode):
            array_name = array_name.get_constant()

        if len(self.indices) == 1: 
            return [], [(array_name, [self.indices[0]])]
        else:
            return [], [(array_name, [self.indices[0], self.indices[1]])]

    def get_writes(self):
        return self.get_use()

    def get_reads(self):
        return self.get_use()

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

    def find_and_replace(self, to_find, replacement):
        child = self.children[0]
        child.find_and_replace(to_find, replacement)

    def get_writes(self):
        return self.children[0].get_writes()

    def get_reads(self):
        return self.children[0].get_reads()

    def __str__(self):
        return "(*%s)" % str(self.children[0])

class GetPointerNode(Node):
    def __init__(self, node):
        super(GetPointerNode, self).__init__()
        self.add_child(node)

    def deep_copy(self):
        my_copy = GetPointerNode(self.children[0].deep_copy())

        return my_copy

    def find_and_replace(self, to_find, replacement):
        child = self.children[0]
        child.find_and_replace(to_find, replacement)

    def get_writes(self):
        return self.children[0].get_writes()

    def get_reads(self):
        return self.children[0].get_reads()

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

        # unify all func to be constant node
        if isinstance(self.func, str):
            self.func = ConstantNode(self.func)

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
        my_copy = CallNode(self.copy(self.func))

        my_copy.args_rw = copy.deepcopy(self.args_rw)

        for child in self.children:
            child_copy = child.deep_copy()
            my_copy.add_child(child_copy)

        return my_copy

    def find_and_replace(self, to_find, replacement):
        if isinstance(self.func, Node):
            self.func.find_and_replace(to_find, replacement)
        else:
            if self.func == to_find:
                self.func = replacement

        # check children
        for child in self.children:
            child.find_and_replace(to_find, replacement)

    def get_writes(self):
        variable_names = []
        array_accesses = []

        # get the number of arguments
        num_args = len(self.args_rw)

        for i in range(num_args):
            rw_bit = self.args_rw[i]

            # if an argument is potentially written to...
            if 0x2 & rw_bit:
                child = self.children[i]

                # get var names and or array accesses for that child
                a, b = child.get_writes();
                
                # add the corresponding things to the right arrays
                variable_names = variable_names + a
                array_accesses = array_accesses + b
            else:
                continue

        return variable_names, array_accesses

    def get_reads(self):
        variable_names = []
        array_accesses = []

        # get the number of arguments
        num_args = len(self.args_rw)

        for i in range(num_args):
            rw_bit = self.args_rw[i]

            # if an argument is potentially read from...
            if 0x1 & rw_bit:
                child = self.children[i]

                # get var names and or array accesses for that child
                a, b = child.get_reads();
                
                # add the corresponding things to the right arrays
                variable_names = variable_names + a
                array_accesses = array_accesses + b
            else:
                continue

        return variable_names, array_accesses

    def __str__(self):
        args = ', '.join(map(str, self.children))

        if isinstance(self.parent, ForNode):
            return "%s(%s);" % (self.func, args)

        return "%s(%s)" % (self.func, args)

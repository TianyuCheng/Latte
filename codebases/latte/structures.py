"""Tree representation of for loops for analysis/tiling/fusion purposes"""

class Node(object):
    def __init__(self):
        # should be extended from
        self.parent = None
        self.child_number = -1
        self.children = []

    def add_child(self, child):
        """Add a child to this node"""
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

        # replace the other child with self
        other_parent.substitute_child(other_child_number, other_parent)

        # mnake the other node a child of this one
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
    

class ForNode(Node):
    def __init__(self, initial, initial_name, loop_bound, increment):
        # the super call doesn't work for some reason, so I'm reconstruting
        # the Node constructor (not very large anyways)
        super(ForNode, self).__init__()
        self.parent = None
        self.child_number = -1
        self.children = []

        # save initial variables
        self.initial = initial
        self.initial_name = initial_name
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

    def __str__(self):
        """Prints the ENTIRE loop including its children"""
        to_return = "for (int " + self.initial_name + " = "
        to_return = to_return + str(self.initial) + "; "
        to_return = to_return + self.initial_name + " < "
        to_return = to_return + str(self.loop_bound) + "; "
        to_return = to_return + self.initial_name + " = "
        to_return = to_return + self.initial_name + " + "
        to_return = to_return + str(self.increment) + ") {\n"

        for node in self.children:
            to_return = to_return + node.__str__()

        to_return = to_return + "}\n"

        return to_return
        
class StatementNode:
    def __init__(self):
        super(StatementNode, self).__init__()
        pass

    pass

class ExpressionNode:
    def __init__(self):
        super(ExpressionNode, self).__init__()
        pass

    pass


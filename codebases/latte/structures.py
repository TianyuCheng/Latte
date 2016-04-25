"""Tree representation of for loops for analysis/tiling/fusion purposes"""

class Node:
    def __init__(self):
        # should be extended from
        self.parent = -1
        self.child_number = -1
        self.children = []

    def add_child(self, child):
        child_number = len(self.children)
        self.children.append(child)

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

    def insert_self(self, other_node):
        """Given another node, take the place of that node in the tree
        by becoming the child of whatever its parent was and making it
        a child of ourself"""
        other_child_number = other_node.get_child_number()
        other_parent = other_node.get_parent()

    def substitute_child(self, child_number, new_child)
        """Overwrite the child at child_number with a new child"""
        # get old child
        old_child = self.children[child_number]

        # make it a child of no one
        old_child.set_child_number(-1)
        old_child.set_parent(-1)

        # overwrite
        self.children[child_number] = new_child
        # set child number and parent
        new_child.set_child_number(child_number)
        new_child.set_parent(self)

class ForNode:

class StatementNode:

class ExpressionNode:

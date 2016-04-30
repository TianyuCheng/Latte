from structures import *

class Optimizer(object):
    def __init__(self, dict_of_trees, ensemble_order):
        self.dict_of_trees = dict_of_trees
        self.ensemble_order = ensemble_order


class TilingOptimizer(Optimizer):
    def __init__(self, dict_of_trees, ensemble_order):
        super(TilingOptimizer, self).__init__(dict_of_trees, ensemble_order)

        # change this to change tile size
        self.tile_size = 3;

        self.num_left = 0;
    
    def optimize(self):
        """Run tiling optimization. Look at each node's loop bounds, see how 
        much it can be tiled by, and do the tiling."""
        new_ensemble_order = []

        # loop through every tree structure
        for e_name in self.ensemble_order:
            new_ensemble_order.append(e_name)

            # grab the node: will be a for loop node
            current_node = self.dict_of_trees[e_name]

            # may not have a ndoe
            if current_node == None:
                continue

            for_nodes = [ current_node ]

            # get all for nodes under this node to tile if necessary
            for child in current_node.get_children():
                if isinstance(child, ForNode):
                    for_nodes.append(child)

            # loop through every for node in order of appearance
            for for_node in for_nodes:
                # run tiling on the loop; it may add new for loops
                # we need to handle
                new_es = self.tile_loop(for_node)

                # append any new for nodes to our handling
                node_parent = for_node.get_parent()

                if not node_parent == None:
                    parent_parent = node_parent.get_parent()
                    if not parent_parent == None:
                        # must add to parent of parent as parent
                        # is the tile loop; needs to be on same level
                        for i in new_es:
                            # note this should be safe since we handle
                            # all parents before their children in
                            # tiling
                            node_to_add = self.dict_of_trees[i]
                            parent_parent.add_child(node_to_add)
                    else:
                        # tile loop is at the top: add to new_en_order
                        for i in new_es:
                            new_ensemble_order.append(i)
                else:
                    # no parent, is root: append to new ensemble
                    # order
                    for i in new_es:
                        new_ensemble_order.append(i)

            # if the current node now has a parent i.e. it was tiled,
            # move up the tree until we get to the root
            new_root = current_node;
            while not new_root.get_parent() == None:
                new_root = current_node.get_parent()

            # replace with new root
            self.dict_of_trees[e_name] = new_root;
        
        self.ensemble_order = new_ensemble_order

        # return the new order
        return new_ensemble_order


    def tile_loop(self, for_node):
        ensembles_to_return = []

        loop_bound = for_node.get_loop_bound()
        loop_var_name = for_node.get_initial_name()

        remainder = loop_bound % self.tile_size

        altered_loop_bound = int(loop_bound) - remainder

        # if loop bound is less than our tile size, ignore it
        if loop_bound < self.tile_size:
            return ensembles_to_return

        tile_var_name = "_tile_" + loop_var_name

        # add an outer for loop and change "this" for loop to deal
        # with the new tile
        tile_node = ForNode(ConstantNode(tile_var_name), 
                            ConstantNode(for_node.get_initial()), 
                            ConstantNode(altered_loop_bound), 
                            ConstantNode(self.tile_size))

        # replace the new node in whatever place the for node we are
        # looking at was
        tile_node.replace_node(for_node)

        # change the initial for loop to work with our new tiling of it
        for_node.set_initial(tile_var_name)
        for_node.set_loop_bound(tile_var_name + " + " + str(self.tile_size))


        # if no remaider, no clean up code required, else we do need cleanup
        # code
        if not remainder == 0:
            leftover_loop_var = "_remain_" + for_node.get_initial_name()
            new_start = int(loop_bound) - remainder
            # create a new for node to handle the leftovers
            # for r = prev upper bound, r < remainder, increment 1
            leftover_for = ForNode(ConstantNode(leftover_loop_var), 
                                   ConstantNode(new_start),
                                   ConstantNode(remainder), 
                                   ConstantNode(1))

            # clone children, add to our new for node
            original_children = for_node.get_children()

            for child in original_children:
                clone_child = child.deep_copy()
                leftover_for.add_child(clone_child)

            # find and replace all occurances of the original for 
            # variable with the leftover loop var

            leftover_children = leftover_for.get_children()

            for child in leftover_children:
                child.find_and_replace(for_node.get_initial_name(),
                                       leftover_loop_var)

            # create a new name for this
            e_name = "_tile_left_" + str(self.num_left)
            self.num_left = self.num_left + 1

            self.dict_of_trees[e_name] = leftover_for

            ensembles_to_return.append(e_name)

        # returns "ensemble" names that should be appended
        return ensembles_to_return

class FusionOptimizer(Optimizer):
    def __init__(self, dict_of_trees, ensemble_order):
        super(FusionOptimizer, self).__init__(dict_of_trees, ensemble_order)

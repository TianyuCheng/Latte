import structures

class Optimizer(object):
    def __init__(self, dict_of_trees, ensemble_order):
        self.dict_of_trees = dict_of_trees
        self.ensemble_order = ensemble_order


class TilingOptimizer(Optimizer):
    def __init__(self, dict_of_trees, ensemble_order):
        super(TilingOptimizer, self).__init__(dict_of_trees, ensemble_order)

        # change this to change tile size
        self.tile_size = 5;
    
    def optimize(self):
        """Run tiling optimization. Look at each node's loop bounds, see how 
        much it can be tiled by, and do the tiling."""
        # loop through every tree structure
        for e_name in self.ensemble_order:
            # grab the node: will be a for loop node
            current_node = self.dict_of_trees[e_name]

            for_nodes = [ current_node ]

            # get all for nodes under this node
            for child in current_node.get_children():
                if isinstance(child, structures.ForNode):
                    for_nodes.append(child)

            # loop through every for node in order of appearance
            for for_node in for_nodes:
                # run tiling on the loop
                self.tile_loop(for_node)

            # if the current node now has a parent i.e. it was tiled,
            # move up the tree until we get to the root
            new_root = current_node;
            while not new_root.get_parent() == None:
                new_root = current_node.get_parent()

            # replace with new root
            self.dict_of_trees[e_name] = new_root;


    def tile_loop(self, for_node):
        loop_bound = for_node.get_loop_bound()
        loop_var_name = for_node.get_initial_name()

        # if loop bound is less than our tile size, ignore it
        if loop_bound < self.tile_size:
            return

        tile_var_name = "_tile_" + loop_var_name

        # add an outer for loop and change "this" for loop to deal
        # with the new tile
        tile_node = structures.ForNode(tile_var_name, for_node.get_initial(), 
                                       loop_bound, self.tile_size)

        # replace the new node in whatever place the for node we are
        # looking at was
        tile_node.replace_node(for_node)

        # change the initial for loop to work with our new tiling of it
        for_node.set_initial(tile_var_name)
        for_node.set_loop_bound(tile_var_name + " + " + str(self.tile_size))

        remainder = loop_bound % self.tile_size

        # if no remaider, no clean up code required, else we do need cleanup
        # code
        if not remainder == 0:
            # TODO
            #TODO
            pass

class FusionOptimizer(Optimizer):
    def __init__(self, dict_of_tress, ensemble_order):
        super(FusionOptimizer, self).__init__(dict_of_trees, ensemble_order)

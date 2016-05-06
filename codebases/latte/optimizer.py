from structures import *

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
        much it can be tiled by, and do the tiling.
        ONLY WORKS ON TOP LEVEL LOOPS AT THE MOMENT (since most processing
        occurs there, it should be fine, Loc would think)"""
        new_ensemble_order = []

        # loop through every tree structure
        for e_name in self.ensemble_order:
            new_ensemble_order.append(e_name)

            # grab the node: will/MUST be a for loop node
            current_node = self.dict_of_trees[e_name]

            #TODO add an assert to test for being a for loop?

            # may not have a node; in that case, ignore it
            if current_node == None:
                continue

            for_nodes = []

            # grab for nodes that are children of the current for
            # node: ONLY grab them if they are the FIRST child
            current_loop_node = current_node

            # get nested for loops under the first for loop if they
            # do not have anything between them
            while True:
                children = current_loop_node.get_children()
                
                if len(children) >= 1:
                    first_child = children[0]

                    if isinstance(first_child, ForNode):
                        for_nodes.insert(0, current_loop_node)
                        current_loop_node = first_child
                        continue
                    else:
                        for_nodes.insert(0, current_loop_node)
                        break
                else:
                    for_nodes.insert(0, current_loop_node)
                    break

            # the current top level
            current_top_level = [current_node]
            new_es = None

            # loop through every for node in order of DEEPEST
            # for node to the outer most one
            for for_node in for_nodes:
                # run tiling on the loop
                current_top_level, new_es = self.tile_loop(for_node, current_top_level)

                # tile loop will add the tiled for loop to the correct spot as well
                # as return the current top_level which may include a remainder node

            if not new_es == []:
                # at top level, but a remainder for loop was added: add name to
                # new ensemble order
                assert len(new_es) == 1
                for i in new_es:
                    new_ensemble_order.append(i)

            # if the current for node now has a parent i.e. it was tiled,
            # move up the tree until we get to the root
            new_root = current_node;
            while not new_root.get_parent() == None:
                new_root = new_root.get_parent()

            # replace with new root in the ensemble
            self.dict_of_trees[e_name] = new_root;
        
        self.ensemble_order = new_ensemble_order

        # return the new order
        return new_ensemble_order


    def tile_loop(self, for_node, current_top_level):
        ensembles_to_return = []

        loop_bound = for_node.get_loop_bound()
        loop_var_name = for_node.get_initial_name()
        used_tile_size = self.tile_size

        # if loop bound is less than our tile size, use it as the "tile size"
        if loop_bound < self.tile_size:
            used_tile_size = loop_bound


        remainder = loop_bound % used_tile_size

        # the current implementation will NOT tile bounds that cause
        # a remainder to appear; do nothing and return the current top level 
        if not remainder == 0:
            return current_top_level, ensembles_to_return


        altered_loop_bound = int(loop_bound) - remainder

        tile_var_name = "_tile_" + loop_var_name

        # create outer for tile loop and change "this" for loop to deal
        # with the new tile
        tile_node = ForNode(ConstantNode(tile_var_name), 
                            ConstantNode(for_node.get_initial()), 
                            ConstantNode(altered_loop_bound), 
                            ConstantNode(used_tile_size))

        top_level = current_top_level[0]

        original_top_level = []

        for i in current_top_level:
            original_top_level.append(i.deep_copy())

        # replace the new node in whatever place the top level node is
        tile_node.replace_node(top_level)

        # if current top level has a remainder node in the second entry, add
        # it as a child as well
        if len(current_top_level) >= 2:
            assert len(current_top_level) == 2
            tile_node.add_child(current_top_level[1])

        # change the for loop we are tiling to work with our new tiling of it
        for_node.set_initial(tile_var_name)
        for_node.set_loop_bound(tile_var_name + " + " + str(used_tile_size))

        # prepare new top level array
        new_top_level = [tile_node]

        # if remainder, need to create remainder node
        # CAUTION: THIS IS BROKEN DO NOT USE; plus if not remainder == 0 
        # we would never even reach this place in the current implementation
        # since I made it so we ignore loops with remainders when tiling
        if not remainder == 0:
            leftover_loop_var = "_remain_" + for_node.get_initial_name()
            new_start = int(loop_bound) - remainder

            # create a new for node to handle the leftovers
            # for r = prev upper bound, r < remainder, increment 1
            leftover_for = ForNode(ConstantNode(leftover_loop_var), 
                                   ConstantNode(new_start),
                                   ConstantNode(int(loop_bound)), 
                                   ConstantNode(1))

            for i in original_top_level:
                i.find_and_replace(i.get_initial_name(),
                                   leftover_loop_var)
                leftover_for.add_child(i)

            ## clone children, add to our new for node
            #original_children = original_top_level.get_children()

            #for child in original_children:
            #    clone_child = child.deep_copy()
            #    leftover_for.add_child(clone_child)

            # find and replace all occurances of the original for 
            # variable with the leftover loop var
            #leftover_children = leftover_for.get_children()

            #for child in leftover_children:
            #    child.find_and_replace(for_node.get_initial_name(),
            #                           leftover_loop_var)

            # create a new name for this
            e_name = "_tile_left_" + str(self.num_left)
            self.num_left = self.num_left + 1

            self.dict_of_trees[e_name] = leftover_for

            new_top_level.append(leftover_for)

            ensembles_to_return.append(e_name)

        # returns the new top level and any new "ensemble" names that should 
        # be appended, specifically only the leftover tile node name
        return new_top_level, ensembles_to_return


class FusionOptimizer(Optimizer):
    def __init__(self, dict_of_trees, ensemble_order):
        super(FusionOptimizer, self).__init__(dict_of_trees, ensemble_order)

    def optimize(self):
        # copy the order: this copy will represent the nodes that haven't
        # been fused over yet
        ensemble_order_copy = self.ensemble_order[:]

        #print ensemble_order_copy
        for current_ensemble in self.ensemble_order:
            #print current_ensemble
            # if the ensemble we are looking at doesn't exist anymore
            # ignore it
            if current_ensemble not in ensemble_order_copy:
                continue

            my_for_node = self.dict_of_trees[current_ensemble]

            # ignore if it has no for node
            if my_for_node == None:
                continue


            # loop over ensemble that comes after this 1
            self_location = ensemble_order_copy.index(current_ensemble)

            to_loop_over = None

            if self_location + 1 >= len(ensemble_order_copy):
                to_loop_over = []
            else:
                to_loop_over = [ensemble_order_copy[self_location + 1]]
                

            for other_ensemble in to_loop_over:
                # shouldn't be ourselves in the new list we are looping over
                assert not other_ensemble == current_ensemble

                # get the top level for node we are considering fusing
                other_for_node = self.dict_of_trees[other_ensemble]

                if other_for_node == None:
                    # no for loop, ignore it
                    continue

                # deep copy it so we can mess with it
                other_for_node = other_for_node.deep_copy()

                current1 = my_for_node
                current2 = other_for_node

                inner_for = None
                other_body = []

                loops_good = True

                while True:
                    if (not isinstance(current1, ForNode) and\
                          isinstance(current2, ForNode)) or\
                         (isinstance(current1, ForNode) and\
                          not isinstance(current2, ForNode)):
                        # one is a for node, 1 isn't: not matching
                        loops_good = False
                        break
                    else:
                        # both for nodes: match the for params
                        initial1 = current1.get_initial()
                        initial2 = current2.get_initial()

                        loop_bound1 = current1.get_loop_bound()
                        loop_bound2 = current2.get_loop_bound()

                        increment1 = current1.get_increment()
                        increment2 = current2.get_increment()

                        # all things must match
                        if not initial1 == initial2:
                            loops_good = False
                            break
                        if not loop_bound1 == loop_bound2:
                            loops_good = False
                            break
                        if not increment1 == increment2:
                            loops_good = False
                            break

                        # everything matches: go ahead and begin replacing for loop
                        # names with our own stuff just in case...
                        other_for_node.find_and_replace(other_for_node.get_initial_name(),
                                                my_for_node.get_initial_name())

                        forchild1 = False
                        forchild2 = False

                        # check next children for another possible for node
                        children1 = current1.get_children()
                        prev_for = current1

                        if len(children1) >= 1:
                            current1 = children1[0]

                            if (isinstance(current1, ForNode)):
                                forchild1 = True
                            else:
                                forchild1 = False

                        children2 = current2.get_children()

                        if len(children2) >= 1:
                            current2 = children2[0]

                            if (isinstance(current2, ForNode)):
                                forchild2 = True
                            else:
                                forchild2 = False

                        # if it's the case that 1 has a for child but the
                        # other doesn't, the loop structure doesn't match
                        if (forchild1 and not forchild2) or\
                           (not forchild1 and forchild2):
                            loops_good = False
                            break
                        # if both do not have for children anymore, structure
                        # must match
                        elif not forchild1 and not forchild2:
                            # save the inner most for loop so we know where to add
                            # during fusion
                            inner_for = prev_for

                            # grab the remaining body of the second for loop (i.e.
                            # the children) and save it for later: we might 
                            # fuse it
                            other_body = children2
                            # get out
                            break
                        # otherwise it will continue into the next iteration

                # if loops_good is not true, we cannot fuse; continue to the
                # next loop 
                if not loops_good:
                    print "loop bounds/structure for", current_ensemble, other_ensemble,\
                           "do not match"
                    #print "\n\nfail\n\n"
                    print current_ensemble, other_ensemble
                    print my_for_node, other_for_node
                    continue

                # otherwise we move onto the variable dependency checks,
                # the non-trivial part of the fusion check

                # do checks from current loop to the loop we want to fuse first

                w_variable_names, w_array_accesses = my_for_node.get_writes()
                r_variable_names, r_array_accesses = other_for_node.get_reads()

                # if there is a write to a variable name in the first loop, it
                # can't be the case that the second loop reads it as fusion may
                # result in it reading a value that isn't good yet
                fusion_good = True

                #print "Starting analysis"
                #print my_for_node
                #print other_for_node

                print "going to check", current_ensemble, other_ensemble,\
                      "dependencies"

                for var_name in w_variable_names:
                    if var_name in r_variable_names:
                        print var_name, "is written in for loop 1 but read in 2nd loop"
                        fusion_good = False
                        break

                    # it's also possible that the var in question is actually
                    # an array that is being completely overwritten by a 
                    # function call or something of the sort: check to see
                    # if there is a read of it in the r_array_accesses
                    # as well

                    r_accesses = set()

                    for r in r_array_accesses:
                        r_accesses.add(r[0])

                    if var_name in r_accesses:
                        print var_name, "is written in for loop 1 but read in 2nd loop"
                        print my_for_node, other_for_node
                        fusion_good = False
                        break

                if not fusion_good:
                    # if fusion isn't good, continue to the next loop
                    continue

                for array_access in w_array_accesses:
                    array_name = array_access[0]
                    array_indices = array_access[1]

                    good = True

                    #print "sanity"
                    #print r_array_accesses
                    #print array_name
                    #print "check"

                    for indices in [x for (a, x) in r_array_accesses if a == array_name]:
                        # right now, assume that indices can only be x, y type 
                        # deals as in no x + 1 or y - 1 or that kind of stuff

                        # if we are accessing something in the second, it must be that we are 
                        # accessing the exact same things, i.e array_indices must match
                        if not array_indices == indices:
                            print "accesses to", array_name, "in loop 1 and 2",\
                                  "do not match", array_indices, indices
                                
                            fusion_good = False
                            good = False
                            break

                    if not good:
                        break

                    # also make sure that the array isn't being read as a whole
                    # i.e as a var name
                    if array_name in r_variable_names:
                        print "the second loop is reading an array 1st loop",\
                              "writes to"
                        fusion_good = False
                        break

                if not fusion_good:
                    # if fusion isn't good, continue to the next loop
                    continue
                
                # now checks from loop we want to fuse to current loop
                w_variable_names, w_array_accesses = other_for_node.get_writes()
                r_variable_names, r_array_accesses = my_for_node.get_reads()

                print "now analyzing loop 2 to loop 1 dependencies"
                print w_variable_names, w_array_accesses
                print r_variable_names, r_array_accesses

                # the only problems that could exist are if the second for loop
                # writes to an index that the first 1 is going to read in a later
                # itereation or if the second for loop writes a variable that
                # is used the by the first
                # e.g.
                # first reads array[x], second writes array[x+1]; 
                # fusion is illegal
                # first reads some var x, but second writes to it every 
                # iteration; fusion is illegal

                for var_name in w_variable_names:
                    if var_name in r_variable_names:
                        print var_name, "is written in for loop 2 but read in",\
                              "for loop 1, which could be bad"
                        fusion_good = False
                        break

                    # it's also possible that the var in question is actually
                    # an array that is being completely overwritten by a 
                    # function call or something of the sort: check to see
                    # if there is a read of it in the r_array_accesses
                    # as well

                    r_accesses = set()

                    for r in r_array_accesses:
                        r_accesses.add(r[0])

                    if var_name in r_accesses:
                        print var_name, "is written in for loop 2 but read in",\
                              "for loop 1, which could be bad"
                        fusion_good = False
                        break

                for array_access in w_array_accesses:
                    array_name = array_access[0]
                    array_indices = array_access[1]

                    good = True

                    for indices in [x for (a, x) in r_array_accesses if a == array_name]:
                        # right now, assume that indices can only be x, y type 
                        # deals as in no x + 1 or y - 1 or that kind of stuff

                        # since we are assuming that, we can't really check for
                        # x+1 or things like that, meaning we'll just be conservative
                        # and force the acccesses to be the same

                        # if we are accessing something in the second, it must be that we are 
                        # accessing the exact same things, i.e array_indices must match
                        if not array_indices == indices:
                            print "accesses to", array_name, "in loop 1 and 2",\
                                  "do not match", array_indices, indices,\
                                  "(2 to 1)"
                                
                            fusion_good = False
                            good = False
                            break

                    if not good:
                        break

                    # also make sure that the array isn't being read as a whole
                    # i.e as a var name
                    if array_name in r_variable_names:
                        print "the first loop is reading an array 2nd loop",\
                              "writes to"
                        fusion_good = False
                        break

                if not fusion_good:
                    # if fusion isn't good, continue to the next loop
                    continue

                # if fusion is still good at this point, do the fusion
                # get the entire other loop body and add it as a child to the
                # inner most for loop
                print "commencing fusion of", current_ensemble, other_ensemble

                for child in other_body:
                    inner_for.add_child(child)

                # remove from ensembles we need to check
                ensemble_order_copy.remove(other_ensemble)
        return ensemble_order_copy

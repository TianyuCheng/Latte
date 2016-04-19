#!/usr/bin/env python2

import inspect, compiler
from compiler.ast import *

def template(tmpl_func):
    def template_wrapper():
        source = inspect.getsource(tmpl_func)
        return ASTTemplate(source)
    return template_wrapper

class ASTTemplate(object):
    """ASTTemplate"""
    def __init__(self, source):
        super(ASTTemplate, self).__init__()
        ast = compiler.parse(source)
        self.ast = ast_remove_template(ast)
        self.types = [ type(None), int, float, str ]

    def match(self, tgt):
        """ match ast with template """
        self.wildcard = dict()
        return self._match(self.ast, tgt)

    def _match(self, tpl, tgt):
        """ match helper function """

        if isinstance(tpl, Discard):
            # ignore Discard
            tpl_kid = tpl.getChildren()[0]
            if tpl_kid.name.startswith('_'):
                self.wildcard[tpl_kid.name] = tgt
                return True
            else:
                return self._match(tpl_kid, tgt)

        # check basic type equality
        for instance_type in self.types:
            if isinstance(tpl, instance_type) and isinstance(tgt, instance_type):
                return tpl == tgt
            if not isinstance(tpl, instance_type) and not isinstance(tgt, instance_type):
                continue        # not applicable for basic type checking
            else:
                return False    # type not equal

        # now at least tpl and tgt are iterable and safe to call getChildren()
        tpl_kids = tpl.getChildren()
        tgt_kids = tgt.getChildren()
        if len(tpl_kids) != len(tgt_kids):
            return False

        kids = zip(tpl_kids, tgt_kids)
        for tpl_kid, tgt_kid in kids:
            isa_match = False
            if isinstance(tpl_kid, Name) or isinstance(tpl_kid, AssName):
                if tpl_kid.name.startswith('_'):
                    self.wildcard[tpl_kid.name] = tgt_kid
                    isa_match = True

            # if not performing matching operations, then try verbatim comparison
            if not isa_match:
                # if anything does not match, return False
                if not self._match(tpl_kid, tgt_kid):
                    return False
        return True

def ast_remove_module(ast):
    """ remove the Module(None, Stmt(...)) wrapper """
    assert isinstance(ast, Module)
    ast = ast.getChildren()[1]      # remove Module
    ast = ast.getChildren()[0]      # find body
    return ast

def ast_remove_template(ast):
    """ remove the Module(None, Stmt(...)) wrapper """
    assert isinstance(ast, Module)
    ast = ast.getChildren()[1]      # remove Module
    ast = ast.getChildren()[0]      # remove Stmt
    ast = ast.getChildren()[5]      # remove Function
    ast = ast.getChildren()[0]      # find body
    return ast

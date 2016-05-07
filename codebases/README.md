PyLatte Project
=============

Authors
-------

- Student I
    + Name: Tianyu Cheng
    + EID: tc26752
    + Email: tianyu.cheng@utexas.edu

- Student II
    + Name: Loc Hoang
    + EID: ldh967
    + Email: loc@cs.utexas.edu

- Student III
    + Name: Xin Lin
    + EID: xl5224
    + Email: jimmylin@utexas.edu

Status
----------

- [DONE] User Description of Network
    
    - [DONE] PyLatte Programming Model: mlp.py,  and more ..

    - [DONE] Neuron Specification (fields, forward, backward): FCNeuron, and more ..
    
- [DONE] Internal Representation

    - [DONE] Implicit adjacency lists: lambda representation of functions

    - [DONE] Parse Lambda Calculus

- [DONE] Share Variable Analysis

    - [DONE] Interpret Lambda Calculus

    - [DONE] Determine uniform dependency of dimension

- [DONE] Synthesis

    - [DONE] Data Flow: traversing dataflow graph by ensemble partitioning

    - [DONE] Compute: convert AoS to SoA form

- [] Optimization

    - [DONE] Library Kernel Pattern Matching: FC, Conv, Pooling

    - [DONE] Model Parallelism: Loop Tiling pass 

    - [KIND OF DONE] Loop Fusion pass

    - [DONE] Data Parallelism: Parallel in Batch Items
    
    - [] Memory-efficient Computation After Fusion: 


- [DONE] Code Generation


Performance Evaluation
-----------------------

| Optimization \ Network        | Data(28,28)+FC(50,50)+Softmax(1,10) | 
|-------------------------------|-------------------------------------|
| None                          | 6.94                                | 
| MKL(-m)                       | 5.63                                | 
| Tiling(-t)                    | 6.14                                | 
| MKL+Tiling (-m -t)            | 4.90                                | 
|                               |                                     | 
| DP (-b -w 4)                  | 4.70                                | 
| DP-MKL (-b -w 4 -m)           | 3.13                                | 
| DP-Tiling (-b -w 4 -t)        | 4.83                                | 
| DP-MKL-Tiling (-b -w 4 -t -m) | 3.23                                | 
|                               |                                     | 

Installation (Stampede)
-----------------------

To run this program, one must have intel MKL and boost installed. On TACC Stampede nodes, 
one needs to load intel icc and boost using the following commands:

```
module load intel/15.0.2
module load boost/1.55.0
```

After loading intel icc and boost, type:

```
make mlp
```


Usage
-----

To generate code for given Latte script, please run

    python generator.py [latte_script(.py)] [out_cpp_file(.cpp)]

Please put the generated cpp file "[out\_cpp\_file(.cpp)]" together with "Latte.h". 

To compile the generated code, enter the root of codebases (where you should 
place "Latte.h" and generated "[out\_cpp\_file(.cpp)]"), and run the following
command:

    make 
    
Please make sure that you have installed the MKL library correctly and that you 
compile the generated cpp code using the flags provided in the MKL compiling 
advisor. 

System Details
-----

Latte is DSL/compiler system that attempt to make it easy for users to write
efficient neural network code using their defined DSL. The user must define
neurons, ensembles, and networks using the DSL, and the Latte compiler will
be able to create efficient neural network code.

Latte was implemented in Python in this system. 

# System Flow

* The user first has to write Python code in the Latte DSL format. The user
should define the neurons, layers, ensembles, etc. they wish to use and
specify how they are connected in the network that he or she wants to create.

* The system will then parse this code and create a Python AST from it.

* This Python AST is pattern-matched to some template, and from the pattern
matching the system generates C++ code. Some optimizations are done 
during this generation.

* The Python AST is also transformed into an AST that we defined ourselves.
This AST makes analysis and changes for later optimization easier.

* The generated C++ code is analyzed 1 more time in order to produce 
a loop-tiled and loop-fused version of it.

* At this point, the system has created neural network code that can be
compiled using a regular C++ compiler.


# Chosen File Summaries

## lib.py

This file comprises the core of the Python-Latte DSL. It defines classes such as
Neuron, Ensemble, and Network that are used by Latte to represent neural networks.

By using this file, one is able to write Python code using the Latte DSL model.

What follows are summaries of selected classes.

### Neuron

Represents a single neuron in a network. One must define forward and backward 
functions that compute forward and backward propogation for the neuron.

Note that a Neuron needs to be subtyped: it should not be used on its own.

### Ensemble

An ensemble is a (2-D) group of neurons. Neurons in an ensemble must all be of the
same type.

One connects ensembles together to create different kinds of layers in a network, 
such as Fully Connected Layers or Softmax Loss Layers.

### Network

A network represents a neural network, and it holds a set of ensembles that comprise
the neural network a user wishes to create. Note that the ensembles in question 
should already be connected to each other as the class does not hold connection 
information.

The class also tracks data sets and training data in order to run 
training/solving on a network.

## templates.py

This file defines a set of templates ASTs that will be used to match to
ASTs that are extracted from a Latte-Python description of a neural network.

Each template for has "wildcards" that are eventually replaced by data from a
the AST that each template is matching against.

By doing so ths system is able to have a some kind of uniformity when examining ASTs
since all ASTs will be matched to some template if they are valid.

## ast\_matcher.py

The system's AST matching implementation is in this file. 

Given an AST from some Latte-Python file, the system can use the match functions
defined in the file in order to attempt to match the AST to some
template defined in `templates.py`. The system also set the wildcards in the templates
by using data from the AST extracted from the Latte-Python file if a AST matches
on some template.

There are also definitions of AST visitor functions that do things such as count
binary operations, reorder binary operations, or rename nodes.

## analyzer.py

Neuron analyzer code is contained in this file.

A neuron analyzer is created for every type of neuron defined in the library:
in this case, `lib.py` is parsed in order to create an analyzer for all of
the neuron types contained in that file. 

An analyzer takes ensemble information and creates C++ code based on the
passed in ensemble information and the type of analyzer that the ensemble
information was passed into. Since forward/backward propogation information
is extracted from the library, the analyzer is able to create the forward
and backward propogation code for the ensemble. This code is then later
used to generate the neural network code.

## translator.py

This takes a Python AST and transforms it into an AST that we defined ourselves
to make it easier to work with.

## structures.py

This defines nodes for our custom AST that we use to generate code and represent
the program.

Notable things about it include allowing us to generate C++ code by simply
calling print on a node, being able to determine writes and reads in a construct
easily, and basically storing most of the information needed to do analysis
for tiling and fusion.

## optimizer.py

The file holds the 2 optimizers in the system: tiling and fusion.

Tiling works by creating tile loop bounds from the bottom up of a nested for
loop structure. It does not produce cleanup code, and will just do nothing
on loops that require cleanup code.

Fusion first examines the tile loop structures (it is being assumed that
tiling always runs with fusion, but the code might still work even if you
do not tile as it has been (attempted to have been/not tested) coded to
handle non-tiled loops as well. The tile loop structures 
must match for fusion to work. It then examines read/write dependencies
between the 2 loops to see if fusion can occur. Our fusion analyzer
is conservative in that it will not fuse if there is a possibility
of a dependence issue between 2 nodes.

Notably, fusion will likely not work on most of our cases as our
analyzer may be too conservative. We did get it "working" for 1 
case: iris-fc-relu, where we used a special one to one layer to show
that allows the analyzer to have more information when doing fusion. More
specifically, a layer being 1 to 1 means that if an array is
used without specifying an index, we implictly asssume that the use
in question will only read/write a single location x,y. This layer
allows us to show you that fusion is possible given the right circumstances
and if we had a more fine-grained analyzer.

## generator.py

This is the main program that in the end generates code in the system. It defines many
functions that create C++ code for the system, and it is responsible for
putting the components of the system together.

As a review of what it does (and by extension a review of the system as well),
an AST is first parsed from a Latte-Python file. This AST is then pattern-matched
using the AST matcher capabilities of the system. From this the system
derives ensemble information, which is saved. The ensemble information is passed
into the neuron analyzers to create the forward and backward propogation nodes
(from which you can create code).

These nodes are then analyzed by optimizer.py if the options are specified so
that tiling and/or fusion can be done.

The code itself is then generated by creating code for all of the ensembles,
layers, solvers, and other things.


Datasets
----

Ask Jimmy or Tianyu about datasets used if necessary.
If Loc is correct, they chose data from a big data set to be the training
set + the test set among other possible things.


Assumptions/Design Decisions/Notes
-----

There may be other assumptions that are not listed here.

- We assume tiling is always legal.

- Fusion does not currently work for most of our tests. It works only for
the 1 to 1 layer used in iris-fc-relu.

- Tiling does not work if the tile size does not divide a loop bound.

- Assume only one data layer and loss layer in each network

- Assume the function name for a data layer contains "DataLayer"

- Assume the function name for a loss layer contains "LossLayer"

- The function you use for add\_connection must be defined as a lambda function.

- One should follow templates when creating new neurons or ensembles. However, 
one should be free to reorder additions and multiplications as it should
be possible to match reordered additions (we have a template for all 
permutations).

- The user won't define variables that begin with \_tile (since that is
how our tile variables are defined).

- No different ensemble types: fused ensembles with layers. (i.e. may not be
exactly Latte)

- Our project is not a 1 to 1 implementation of Latte.

References
--------
We primarily referred to the Latte Paper provided in the course CS380C Compilers.


- [MKL Library] Documentation of Matrix Multiplication:

    https://software.intel.com/en-us/node/468480

- [MKL Library] Performance Comparison of Matrix Multiplication of MKL Library:

    http://www.ics.uci.edu/~paolo/FastMM/FMM-Reference/reference.html

- [MKL Library] Intel® Math Kernel Library Link Line Advisor:

    https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor

- [Python] API of Python AST Library (Abstract Syntax Tree): 

	https://docs.python.org/2/library/ast.html
	
- [Python] API of Python Parser Library:
	
	https://docs.python.org/2/library/parser.html

- Library of Libsvm Datasets:

   https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html

- [Other] Markdown

   https://guides.github.com/features/mastering-markdown/

- [Other] Markdown Table Generator

   http://www.tablesgenerator.com/markdown\_tables

- [Other] Loop Fusion (and other Loop xforms)

   https://www.cs.utexas.edu/users/lin/cs380c/handout27.pdf

- [Other] Class notes


Errors
------

It seems our accuracy has been reduced (this is an observation on Loc's part, so
maybe Jimmy or Tianyu have a better explanation than me/Loc). Plus, he only
tested for a few of the tests, so it may not be a universal problem.


Acknowledgements
-------
We would like to give many thanks to Prof. Pingali and our Teaching Assistant 
Mr. Roshan for their teaching and instructions. 

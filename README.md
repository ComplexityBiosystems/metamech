This repository contains `python3` and `C` code for the design of mechanical metamaterials as explained in 

S Bonfanti, R Guerra, F Font-Clos, D Rayneau-Kirkhope, and S Zapperi. 2020.  
*Automatic Design of Mechanical Metamaterial Actuators*  
preprint version [arXiv:2002.03032](http://arxiv.org/abs/2002.03032.)  
journal version [ add link to journal  ]

## Requirements
To build the metamech library you need the following external packages:

+ keras
+ matplotlib
+ numpy
+ ordered_set
+ pandas
+ pytest
+ scipy
+ tqdm

See `requirements.txt` for exact pinned versions (but usually earlier versions work as well).

## Installation
We recommend that you **install all requirements** in an isolated environment via conda, pyenv, pipenv or a similar environment manager. If you are a conda user

```bash
conda create --name metamech pip
```
will create and activate an isolated python environment called metamech. Then you can activate the environment and install all dependencies automatically without interfering with your standard python installation
```bash
conda activate metamech
pip install -r requirements.txt
```
 
**Compile the `C` part of the code**, which runs the FIRE algorithm, into
a shared library. Depending on your operating system, do as follows:

On a macOS with Clang:
```bash
clang -std=c11 -Wall -Wextra -pedantic -c -fPIC metamech/minimize.c -o metamech/minimize.o
clang -shared metamech/minimize.o -o metamech/minimize.dylib
```

On a Linux system with GCC you can use:
```bash
gcc -Wall -Wextra -pedantic -c -fPIC metamech/minimize.c -o metamech/minimize.o
gcc -shared metamech/minimize.o -o metamech/minimize.sso
```
> **WARNING** do not change `.sso` to the more standard `.so`  extension. 


**Install the python part of the code** in development mode.
```bash
pip install -e .
```

**Run unit tests**, to make sure everything went well
```bash
pytest -v
```

## Usage
A set of handy pre-configured loader functions for the structures used in the manuscript are available at the accompanying repository [metamech_datasets](https://github.com/ComplexityBiosystems/metamech_datasets). Here we show how to design a small mechanical metamaterial from scratch, to illustrate how the library is structured.

```python
import numpy as np

from metamech.lattice import Lattice
from metamech.actuator import Actuator

# triangular regular lattice of side length 8
nodes_positions = np.array([
    [0.     , 0.86603], [0.5    , 0.     ], [1.     , 0.86603], [1.5    , 0.     ],
    [2.     , 0.86603], [2.5    , 0.     ], [3.     , 0.86603], [3.5    , 0.     ],
    [4.     , 0.86603], [4.5    , 0.     ], [5.     , 0.86603], [5.5    , 0.     ],
    [6.     , 0.86603], [6.5    , 0.     ], [7.     , 0.86603], [7.5    , 0.     ],
    [8.     , 0.86603], [0.     , 2.59808], [0.5    , 1.73205], [1.     , 2.59808],
    [1.5    , 1.73205], [2.     , 2.59808], [2.5    , 1.73205], [3.     , 2.59808],
    [3.5    , 1.73205], [4.     , 2.59808], [4.5    , 1.73205], [5.     , 2.59808],
    [5.5    , 1.73205], [6.     , 2.59808], [6.5    , 1.73205], [7.     , 2.59808],
    [7.5    , 1.73205], [8.     , 2.59808], [0.     , 4.33013], [0.5    , 3.4641 ],
    [1.     , 4.33013], [1.5    , 3.4641 ], [2.     , 4.33013], [2.5    , 3.4641 ],
    [3.     , 4.33013], [3.5    , 3.4641 ], [4.     , 4.33013], [4.5    , 3.4641 ],
    [5.     , 4.33013], [5.5    , 3.4641 ], [6.     , 4.33013], [6.5    , 3.4641 ],
    [7.     , 4.33013], [7.5    , 3.4641 ], [8.     , 4.33013], [0.     , 6.06218],
    [0.5    , 5.19615], [1.     , 6.06218], [1.5    , 5.19615], [2.     , 6.06218],
    [2.5    , 5.19615], [3.     , 6.06218], [3.5    , 5.19615], [4.     , 6.06218],
    [4.5    , 5.19615], [5.     , 6.06218], [5.5    , 5.19615], [6.     , 6.06218],
    [6.5    , 5.19615], [7.     , 6.06218], [7.5    , 5.19615], [8.     , 6.06218],
    [0.     , 7.79423], [0.5    , 6.9282 ], [1.     , 7.79423], [1.5    , 6.9282 ],
    [2.     , 7.79423], [2.5    , 6.9282 ], [3.     , 7.79423], [3.5    , 6.9282 ],
    [4.     , 7.79423], [4.5    , 6.9282 ], [5.     , 7.79423], [5.5    , 6.9282 ],
    [6.     , 7.79423], [6.5    , 6.9282 ], [7.     , 7.79423], [7.5    , 6.9282 ],
    [8.     , 7.79423]])

edges_indices = np.array([
       [ 0,  1],[ 0,  2],[ 0, 18],[ 1,  2],[ 1,  3],[ 2, 18],
       [ 2,  3],[ 2,  4],[ 2, 20],[ 3,  5],[ 3,  4],[ 4,  5],
       [ 4, 22],[ 4, 20],[ 4,  6],[ 5,  7],[ 5,  6],[ 6, 22],
       [ 6,  7],[ 6, 24],[ 6,  8],[ 7,  9],[ 7,  8],[ 8, 10],
       [ 8, 24],[ 8,  9],[ 8, 26],[ 9, 10],[ 9, 11],[10, 26],
       [10, 11],[10, 12],[10, 28],[11, 12],[11, 13],[12, 28],
       [12, 13],[12, 14],[12, 30],[13, 15],[13, 14],[14, 15],
       [14, 32],[14, 30],[14, 16],[15, 16],[16, 32],[17, 19],
       [17, 18],[17, 35],[18, 19],[18, 20],[19, 35],[19, 21],
       [19, 20],[19, 37],[20, 22],[20, 21],[21, 22],[21, 39],
       [21, 37],[21, 23],[22, 24],[22, 23],[23, 24],[23, 39],
       [23, 41],[23, 25],[24, 26],[24, 25],[25, 27],[25, 26],
       [25, 41],[25, 43],[26, 27],[26, 28],[27, 43],[27, 29],
       [27, 28],[27, 45],[28, 29],[28, 30],[29, 45],[29, 31],
       [29, 30],[29, 47],[30, 32],[30, 31],[31, 32],[31, 49],
       [31, 47],[31, 33],[32, 33],[33, 49],[34, 35],[34, 36],
       [34, 52],[35, 36],[35, 37],[36, 52],[36, 37],[36, 38],
       [36, 54],[37, 39],[37, 38],[38, 39],[38, 56],[38, 54],
       [38, 40],[39, 41],[39, 40],[40, 56],[40, 41],[40, 58],
       [40, 42],[41, 43],[41, 42],[42, 44],[42, 58],[42, 43],
       [42, 60],[43, 44],[43, 45],[44, 60],[44, 45],[44, 46],
       [44, 62],[45, 46],[45, 47],[46, 62],[46, 47],[46, 48],
       [46, 64],[47, 49],[47, 48],[48, 49],[48, 66],[48, 64],
       [48, 50],[49, 50],[50, 66],[51, 53],[51, 52],[51, 69],
       [52, 53],[52, 54],[53, 69],[53, 55],[53, 54],[53, 71],
       [54, 56],[54, 55],[55, 56],[55, 73],[55, 71],[55, 57],
       [56, 58],[56, 57],[57, 58],[57, 73],[57, 75],[57, 59],
       [58, 60],[58, 59],[59, 61],[59, 60],[59, 75],[59, 77],
       [60, 61],[60, 62],[61, 77],[61, 63],[61, 62],[61, 79],
       [62, 63],[62, 64],[63, 79],[63, 65],[63, 64],[63, 81],
       [64, 66],[64, 65],[65, 66],[65, 83],[65, 81],[65, 67],
       [66, 67],[67, 83],[68, 69],[68, 70],[69, 70],[69, 71],
       [70, 71],[70, 72],[71, 73],[71, 72],[72, 73],[72, 74],
       [73, 75],[73, 74],[74, 75],[74, 76],[75, 77],[75, 76],
       [76, 78],[76, 77],[77, 78],[77, 79],[78, 79],[78, 80],
       [79, 80],[79, 81],[80, 81],[80, 82],[81, 83],[81, 82],
       [82, 83],[82, 84],[83, 84],
       ])

# construct the lattice
lattice = Lattice(
    nodes_positions=nodes_positions,
    edges_indices=edges_indices,
    linear_stiffness=10,
    angular_stiffness=0.2
)

# we want to start off with a configuration
# will all possible edges present
for edge in lattice._possible_edges:
    lattice.flip_edge(edge)

# input displacement is 0.1 units in -y direction
# applied to top-right nodes
input_nodes = [81, 82, 83, 84]
input_vectors = np.array([
    [ 0.       , -0.1],
    [ 0.       , -0.1],
    [ 0.       , -0.1],
    [ 0.       , -0.1]
])

# measure output at top-left nodes along -x direction
output_nodes = [68, 69, 70, 71]
output_vectors = np.array([
    [-1, 0],
    [-1, 0],
    [-1, 0],
    [-1, 0],
])

# freeze the bottomn layer
frozen_nodes = [1, 3, 5, 7, 9, 11, 13, 15]

# construct the actuator
actuator = Actuator(
    lattice=lattice,
    input_nodes=input_nodes,
    input_vectors=input_vectors,
    output_nodes=output_nodes,
    output_vectors=output_vectors,
    frozen_nodes=frozen_nodes
)
```
If you're on a jupyter notebook or ipython, you can now visualize the actuator.
```python
# creates a matplotlib figure
# of displaced configuration
from metamech.viz import show_actuator
show_actuator(actuator)
```

Initially, the the structure is a full lattice which does not perform the desired
mechanical action, so the efficiency is close to 0:
```python
print("initial efficiency", actuator.efficiency)
# initial efficiency -0.08673312423435053
```

To find a more efficienct configuration, we need to run the Monte Carlo part of the
algorithm which adds/removes bonds as explained in the manuscript.
We do this with a Metropolis object.
```python
from metamech.metropolis import Metropolis
metropolis = Metropolis(
    actuator=actuator
)
metropolis.run(
    initial_temperature=0.1,
    final_temperature=0,
    num_steps=1000
)
```
While the Monte Carlo simulation runs, you can see a progress bar which shows some key
variables such as temperature, efficiency and acceptance rate.
When the simulation finishes, we check that the final efficiency is indeed much higher:

```python
print("final efficiency", actuator.efficiency)
# final efficiency 2.8265941145286715
```

The `metroplis.history` attirbute stores the evolution of key parameters in a dictionary of lists,
which we can easily turn into a pandas dataframe, to then export to our preferred format

```python
import pandas as pd
history_df = pd.DataFrame(metropolis.history)
history_df.to_xlsx("/path/to/MC_history.xlsx")
history_df.to_csv("/path/to/MC_history.csv")
```

We can store the final configuration in LAMMPS format, to visualize it with other software such as Ovito.

```python
actuator.to_lammps("/path/to/output.lammps")
actuator._get_displaced_actuator().to_lammps("/path/to/output_displaced.lammps")
```

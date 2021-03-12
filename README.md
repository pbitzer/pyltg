# README #

This package is primarily for lightning scientists. There are classes to manipulate data from common lightning instrumentation.

However, there are a number of utilities that others may find useful.

The "core" classes correspond to the various lightning instruments/data sources:
- LMA
- Vaisala (NLDN and GLD360)
- Earth Networks
- Lightning Imaging Sensor (LIS)
- Geostationary Lightning Mapper (GLM)
- Huntsville Alabama Marx Meter Array (HAMMA) and similar (Level 2 data) 

### Installing ###

* ~~Clone the repository [https://bitbucket.org/uah_ltg/pyltg]~~ *The repository has been moved to Github!*
* Clone the repository [https://github.com/pbitzer/pyltg/]
* Go to the `pyltg` directory created and install it with pip:
```
pip install -e .
```
(It is highly recommended that you use the `-e` switch!)

* Standard science packages are required (e.g., `numpy, pandas, matplotlib`).
See the `environment.yml` file for a full list of dependencies.

* Alternatively, you can create a conda environment. For example, for a conda
environment named `ltg` (after going to the `pyltg` directory cloned):
```
conda env create -f environment.yml
conda activate ltg
```

* For full use of Geostationary Lightning Mapper (GLM) code, you should also 
install `glmtools`  [https://github.com/deeplycloudy/glmtools]. While you
can follow the installation discussed there, you can also simply clone the 
repository and install it with `pip install -e .`  (from within
the `glmtools` directory created after cloning.)

### Basic Use ###

Import the package:
```
import pyltg
```
On package import, the "core" classes are also imported.

Say you have an LMA file. Read in the file at initialization:

Initialize a instance of the base class:
```
f = "nalma_lylout_20170427_07_3600.dat.gz"
lma = LMA(f)
```

Limit the lats and lons to a particular range: 
```
lat_bound = [35, 36]
lon_bound = [-88, -87]
cnt = lma.limit(lats=lat_bound, lons=lon_bound)
```

Any attribute of the data can be passed into limit. 

Attributes can be accessed directly:
```
my_time = lma.time
```
See the example notebook(s) [https://www.nsstc.uah.edu/users/phillip.bitzer/python_doc/pyltg/examples.html] for more information.

### Caveats, Gotchas, etc ###

* Under the hood, the data is maintained in a Pandas Dataframe, but for the most part you won't need to interact at that level.
* There is a `baseclass` class that is inherited by all the others. Most users have no need to work with this directly. Instead, go straight to the class corresponding to the lightning data you're using.

### Issues ###
If you run into a bug, please report it [https://github.com/pbitzer/pyltg/issues]

### Contribution guidelines ###

* Incoming

### Who do I talk to? ###

* Phillip Bitzer (bitzerp@uah.edu)

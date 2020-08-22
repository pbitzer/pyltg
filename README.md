# README #

This package is primarily for lightning scientists. There are classes to manipulate data from common lightning instrumentation.

However, there are a number of utilities that others may find useful.

The "core" classes correspond to the various lightning instrumentats/data sources:
- LMA
- Vaisala (NLDN and GLD360)
- Earth Networks
- GLM (in progress)
- HAMMA (in progress)

### Installing ###

* Clone the repository [https://bitbucket.org/uah_ltg/pyltg]
* Go to the `pyltg` directory created and install it with pip:
```
pip install -e .
```
(It is highly recommended that you use the `-e` switch!)

* Standard science packages are required (e.g., `numpy, pandas, matplotlib`).
See the `environment.yml` file for a full list of dependencies 

* Alternatively, you create a conda environment (after going to the `pyltg` 
directory cloned):
```
conda env create environment.yml
```
* For full use of Geostationary Lightning Mapper (GLM) code, you should also 
install `glmtools`  [https://github.com/deeplycloudy/glmtools]. While you
can follow the installation discussed there, you can also simply clone this 
repository and install it with `pip install -e .` as well (from within
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
pare_lma = lma.limit(lats=lat_bound, lons=lon_bound)
```
Right now, a DataFrame is returned, not a object. This will be changed "soon."

Any attribute of the data can be passed into limit. 

Attributes can be accessed directly:
```
my_time = lma.time
```


### Caveats, Gotchas, etc ###

* Under the hood, the data is maintained in a pandas Dataframe, but for the most part you won't need to interact at that level.
* There is a `baseclass` class that is inherited by all the others. Most users have no need for this. Instead, go straight to the class you need.

### Issues ###
If you run into a bug, please report it [https://bitbucket.org/uah_ltg/pyltg/issues]

### Contribution guidelines ###

* Incoming

### Who do I talk to? ###

* Phillip Bitzer (bitzerp@uah.edu)

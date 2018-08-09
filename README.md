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

* Clone this repository somewhere on your Python path. 
* Right now, the only dependicies are the standard ones installed with Anaconda (Python 3.6)

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

### Contribution guidelines ###

* Incoming

### Who do I talk to? ###

* Phillip Bitzer (bitzerp@uah.edu)

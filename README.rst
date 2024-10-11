============
rompy-xbeach
============


.. image:: https://img.shields.io/pypi/v/rompy_xbeach.svg
    :target: https://pypi.python.org/pypi/rompy_xbeach

.. image:: https://img.shields.io/travis/rom-py/rompy_xbeach.svg
    :target: https://travis-ci.com/rom-py/rompy_xbeach

.. image:: https://readthedocs.org/projects/rompy-xbeach/badge/?version=latest
    :target: https://rompy-xbeach.readthedocs.io/en/latest/?version=latest
    :alt: Documentation Status




Rompy Xbeach Config package.



Features
--------

Grid
~~~~
* CRS aware grid object
* Ability to specify origin in different coordinate systems from the grid (e.g., lat/lon)
* Plotting supporting projections and high resolution coastlines
* Option to display the grid mesh
* Option to display the offshore boundary and origin
* Method to expand the grid boundaries
* Property methods to hold the cartopy stereographic projection and the transform

Data
~~~~
* CRS aware
* Interpolator interface
* Modified source objects to support crs (rioxarray based)
* New custom Geotiff specific source
* Seaward extension framework
* Lateral extension capability

Xarray Accessor
~~~~~~~~~~~~~~~
* Class method to create a xbeach dataset from model files
* Method to plot the bathymetry grid, profiles and slopes together


Questions
---------
* Support interpolate_na In the data objects?
    * Existing options in rioxarray: {'linear', 'nearest', 'cubic'}
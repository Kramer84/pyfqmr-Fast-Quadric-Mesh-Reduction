pyfqmr : Python Fast Quadric Mesh Reduction
===========================================

Cython wrapper around `sp4acerat's quadrics mesh reduction
algorithm <https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification>`__.

Requirements:
~~~~~~~~~~~~~

-  *Numpy*
-  *Cython* (only for compilation, but not needed if installed from PyPI)

Installation :
~~~~~~~~~~~~~~
pyfqmr can be installed via  `pip <https://pypi.org/project/pyfqmr/0.1.1/>`_ :


.. code:: bash

    pip install pyfqmr


Compilation :
~~~~~~~~~~~~~

Run:

.. code:: bash

    pip install .

Usage:
~~~~~~

.. code:: python

    >>> # We assume you have a numpy based mesh processing software
    >>> # Where you can get the vertices and faces of the mesh as numpy arrays.
    >>> # For example Trimesh or meshio
    >>> import pyfqmr
    >>> import trimesh as tr
    >>> bunny = tr.load_mesh('example/Stanford_Bunny_sample.stl')
    >>> # Simplify object
    >>> mesh_simplifier = pyfqmr.Simplify()
    >>> mesh_simplifier.setMesh(bunny.vertices, bunny.faces)
    >>> mesh_simplifier.simplify_mesh(target_count = 1000, aggressiveness=7, preserve_border=True, verbose=True)
    >>> vertices, faces, normals = mesh_simplifier.getMesh()
    >>>
    >>> # To make verbose visible, use logging module :
    >>> import logging
    >>>
    >>> # Configure the logger to show debug messages
    >>> logging.basicConfig(level=logging.DEBUG)
    >>> logger = logging.getLogger("pyfqmr")
    >>>
    >>> # Optionally, log to a file:
    >>> # logging.basicConfig(filename='mesh_simplification.log', level=logging.DEBUG)
    >>>
    >>> # Now, when `simplify_mesh(verbose=True)` is called,
    >>> # messages will appear in the console or the log file



Controlling the reduction algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters of the **`simplify\_mesh`** method that can be tuned.

-  **target\_count**
    Target number of triangles.
-  **update\_rate**
    Number of iterations between each update.
-  **max\_iterations**
    Maximal number of iterations
-  **aggressiveness**
    Parameter controlling the growth rate of the threshold at each iteration when lossless is False.
-  **preserve\_border**
    Flag for preserving the vertices situated on open borders. Applies the method described in `this issue <https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification/issues/14>`__.
-  **alpha**
    Parameter for controlling the threshold growth. Exact implication described below.
-  **K**
    Parameter for controlling the threshold growth. Exact implication described below.
-  **lossless**
    Flag for using the lossless simplification method. Sets the update rate to 1 .
-  **threshold\_lossless**
    Maximal error after which a vertex is not deleted, only when the lossless flag is set to True.
-  **verbose**
    Falg controlling verbosity

Controlling the lossless reduction algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters of the **`simplify\_mesh\_lossless`** method that can be tuned.

-  **verbose**
    Falg controlling verbosity
-  **epsilon**
    Maximal error after which a vertex is not deleted.
-  **max\_iterations**
    Maximum number of iterations.
-  **preserve\_border**
    Flag for preserving the vertices situated on open borders. Applies the method described in `this issue <https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification/issues/14>`__.

Note
~~~~

- The **`simplify\_mesh\_lossless`** method is different from the **`simplify\_mesh`** method with the lossless flag enabled, and should be prefered when quality is the aim and not a precise number of target triangles.
- Tests have shown that the **threshold\_lossless** argument has little to no influence on the reduction of the meshes.


Implications of the parameters for the threshold growth rate :
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
(``simplify_mesh()`` method when not in lossless mode)

$$threshold = alpha \* (iteration + K)^{agressiveness}$$


More information is to be found on Sp4cerat's repository :
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
`Fast-Quadric-Mesh-Simplification <https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification>`__

Huge thanks to Sp4cerat for making his code available!

# pyfqmr : Python Fast Quadric Mesh Reduction 

Cython wrapper around [sp4acerat's quadrics mesh reduction algorithm](https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification). 

### Requirements: 
- *Cython*

### Compilation and installation :
Run:
```bash
python setup.py install
```
### Usage:
```python
>>> #We assume you have a numpy based mesh processing software
>>> #Where you can get the vertices and faces of the mesh as numpy arrays.
>>> #For example Trimesh or meshio
>>> import pyfqmr
>>> import trimesh as tr
>>> bunny = tr.load_mesh('Stanford_Bunny_sample.stl')
>>> #Simplify object
>>> mesh_simplifier = pyfqmr.Simplify()
>>> mesh_simplifier.setMesh(bunny.vertices, bunny.faces)
>>> mesh_simplifier.simplify_mesh(target_count = 1000, aggressiveness=7, preserve_border=True, verbose=10)
iteration 0 - triangles 112402 threshold 2.187e-06
iteration 5 - triangles 62674 threshold 0.00209715
iteration 10 - triangles 21518 threshold 0.0627485
iteration 15 - triangles 9086 threshold 0.61222
iteration 20 - triangles 4692 threshold 3.40483
iteration 25 - triangles 2796 threshold 13.4929
iteration 30 - triangles 1812 threshold 42.6184
iteration 35 - triangles 1262 threshold 114.416
simplified mesh in 0.2518 seconds 
>>> vertices, faces, normals = mesh_simplifier.getMesh()

```

### Controlling the reduction algorithm

Parameters of the '''simplify_mesh''' method that can be tuned.

* **target_count**  
	Target number of triangles.
* **update_rate**  
	Number of iterations between each update.
* **max_iterations**  
	Maximal number of iterations 
* **aggressiveness**  
	Parameter controlling the growth rate of the threshold at each iteration when lossless is False.
* **preserve_border**  
	Flag for preserving the vertices situated on open borders. Applies the method descriped in [this issue](https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification/issues/14).
* **alpha**  
	Parameter for controlling the threshold growth. Exact implication described below.
* **K**  
	Parameter for controlling the thresold growth. Exact implication described below.
* **lossless**  
	Flag for using the lossless simplification method. Sets the update rate to 1 .
* **threshold_lossless**  
	Maximal error after which a vertex is not deleted, only when the lossless flag is set to True.
* **verbose**  
	Controls verbosity

##### Implications of the parameters of the threshold growth rate
This is only true when not in lossless mode. 
- **threshold = alpha * (iteration + K)\*\*agressiveness** 

More information is to be found on Sp4cerat's repo : [Fast-Quadric-Mesh-Simplification](https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification)

Huge thanks to Sp4cerat for making his code available! 
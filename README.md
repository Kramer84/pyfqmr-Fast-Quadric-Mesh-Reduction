# Fast-Quadric-Mesh-Simplification - Python 

Cython wrapper around [sp4acerat's quadrics mesh reduction algorithm](https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification). 

### Requirements: 
- *Numpy*
- *Trimesh*
- *Cython*

### Compilation :
Run inside of the pySimplify folder 
``` 
python setup.py build_ext --inplace
```
### Usage:
```
>>> from pySimplify import pySimplify
>>> import trimesh as tr
>>> bunny = tr.load_mesh('Stanford_Bunny_sample.stl)
>>> bunny
<trimesh.Trimesh(vertices.shape=(56203, 3), faces.shape=(112402, 3))>
>>> simplify = pySimplifyy()
>>> simplify.setMesh(bunny)
>>> simplify.simplify_mesh(target_count = 1000, aggressiveness=7, preserve_border=True, verbose=10)
iteration 0 - triangles 112402 threshold 2.187e-06
iteration 5 - triangles 62674 threshold 0.00209715
iteration 10 - triangles 21518 threshold 0.0627485
iteration 15 - triangles 9086 threshold 0.61222
iteration 20 - triangles 4692 threshold 3.40483
iteration 25 - triangles 2796 threshold 13.4929
iteration 30 - triangles 1812 threshold 42.6184
iteration 35 - triangles 1262 threshold 114.416
simplified mesh in 0.2254 seconds from 112402 to 1000 triangles
>>> smallBunny = simplify.getMesh()
>>> smallBunny
<trimesh.Trimesh(vertices.shape=(502, 3), faces.shape=(1000, 3))>
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
- **threshold = alpha*pow( iteration + K, agressiveness )**

### Sp4acerat's comments :

Mesh triangle reduction using quadrics - for Windows, OSX and Linux (thx Chris Rorden)

![img](https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification/blob/master/screenshot.png?raw=true)

**Summary** Since I couldn't find any code that's fast, memory efficient, free and for high quality, I developed my version of the quadric based edge collapse mesh simplification method. It uses a threshold to determine which triangles to delete, which avoids sorting but might lead to lesser quality. It is about 4x as fast as Meshlab and can simplify 2M -> 30k triangles in 3.5 seconds.

**Usage** The functionality is contained in Simplify.h. The function to call is *simplify_mesh(target_count)*. The code is kept pretty slim, so the main method has just around 400 lines of code. 

**Obj File Limitations** The Obj file may only have one group or object. Its a very simple reader/writer, so don't try to use multiple objects in one file

**Windows, OSX and Linux Command Line Tool added**

Thanks to [Chris Rorden](https://github.com/neurolabusc) for creating a command line version and providing binaries for OSX and Linux.

**Pascal Version**

[Chris Rorden](https://github.com/neurolabusc) further created a pascal version that you can find here

https://github.com/neurolabusc/Fast-Quadric-Mesh-Simplification-Pascal-

License : MIT

Please don't forget to cite this page if you use the code!

## Projects Using this Method

**[Surf-Ice](http://www.mccauslandcenter.sc.edu/crnl/)**

Surf Ice is a tool for surface rendering the cortex with overlays to illustrate tractography, network connections, anatomical atlases and statistical maps. While there are many alternatives, Surf Ice is easy to use and uses advances shaders to generate stunning images. It supports many popular mesh formats [3ds, ac3d, BrainVoyager (srf), ctm, Collada (dae), dfs, dxf, FreeSurfer (Asc, Srf, Curv, gcs, Pial, W), GIfTI (gii), gts, lwo, ms3d, mz3, nv, obj, off, ply, stl, vtk], connectome formats (edge/node) and tractography formats [bfloat, pdb, tck, trk, vtk].

![img](https://www.nitrc.org/plugins/mwiki/images/thumb/1/17/Surfice%3ASimplify.jpg/180px-Surfice%3ASimplify.jpg)
![img](https://www.nitrc.org/plugins/mwiki/images/thumb/8/8e/Surfice%3AAmbientOcclusion.jpg/180px-Surfice%3AAmbientOcclusion.jpg)

**[THREE.JS Sample using the Method](https://cdn.rawgit.com/timknip/mesh-decimate/afe5339/examples/three.js/index.html)**

![img](https://i.imgur.com/qhHFxq4.png)

**[Live Web Version by 
Tiger Yuhao Huang](https://myminifactory.github.io/Fast-Quadric-Mesh-Simplification/)**

![img](https://i.imgur.com/N5e2U9u.png)

**[ Unity Mesh Decimator by Mattias Edlund / Whinarn](https://github.com/Whinarn/UnityMeshSimplifier)**

**[ .NET Mesh Decimator by Mattias Edlund / Whinarn](https://github.com/Whinarn/MeshDecimator)**

**[ Javascript Mesh Decimator by Andrew Taber / ataber](https://github.com/ataber/mesh-simplify)**

**[ Javascript Mesh Decimator by Joshua Koo / zz85](https://gist.github.com/zz85/a317597912d68cf046558006d7647381)**

**[ Java Mesh Decimator by Jayfella](https://hub.jmonkeyengine.org/t/isosurface-mesh-simplifier/41046)**


# distutils: language = c++

cimport cython

from libcpp.vector cimport vector
from libcpp cimport bool

from time import time as _time

cimport numpy as np
import numpy as np

class _hidden_ref(object):
    """Hidden Python object to keep a reference to our numpy arrays
    """
    def __init__(self):
        self.faces = None
        self.verts = None

_REF = _hidden_ref() 

cdef extern from "Simplify.h" namespace "Simplify" :
    void simplify_mesh( int target_count, int update_rate, double aggressiveness, 
                        bool verbose, int max_iterations,double alpha, int K, 
                        bool lossless, double threshold_lossless, bool preserve_border)
    void setMeshFromExt(vector[vector[double]] vertices, vector[vector[int]] faces)
    vector[vector[int]] getFaces()
    vector[vector[double]] getVertices()
    vector[vector[double]] getNormals()

cdef class Simplify : 

    cdef int[:,:] faces_mv
    cdef double[:,:] vertices_mv
    cdef double[:,:] normals_mv

    cdef vector[vector[int]] triangles_cpp
    cdef vector[vector[double]] vertices_cpp
    cdef vector[vector[double]] normals_cpp
    
    def __cinit__(self):
        pass

    @cython.boundscheck(False)
    def getMesh(self):
        """Gets the mesh from the simplify object once the simplification is done

        Returns
        -------
        verts : numpy.ndarray
            array of vertices of shape (n_vertices,3)
        faces : numpy.ndarray
            array of faces of shape (n_faces,3)
        norms : numpy.ndarray
            array of normals of shape (n_faces,3)
        """
        self.triangles_cpp = getFaces()
        self.vertices_cpp = getVertices()
        self.normals_cpp = getNormals()

        cdef size_t N_t = self.triangles_cpp.size()
        cdef size_t N_v = self.vertices_cpp.size()
        cdef size_t N_n = self.normals_cpp.size()
        cdef np.ndarray[int, ndim=2] faces = np.zeros((N_t, 3), dtype=np.int32)
        cdef np.ndarray[double, ndim=2] verts = np.zeros((N_v, 3), dtype=np.float64)
        cdef np.ndarray[double, ndim=2] norms = np.zeros((N_n, 3), dtype=np.float64)

        cdef size_t i = 0
        cdef size_t j = 0

        for i in range(N_t):
            for j in range(3):
                faces[i,j] = self.triangles_cpp[i][j]
        for i in range(N_v):
            for j in range(3):
                verts[i,j] = self.vertices_cpp[i][j]
        for i in range(N_n):
            for j in range(3):
                norms[i,j] = self.normals_cpp[i][j]

        return verts, faces, norms

    cpdef void setMesh(self, vertices, faces, face_colors=None):
        """Method to set the mesh of the simplifier object.
        
        Arguments
        ---------
        vertices : numpy.ndarray
            array of vertices of shape (n_vertices,3)
        faces : numpy.ndarray
            array of faces of shape (n_faces,3)
        face_colors : numpy.ndarray
            array of face_colors of shape (n_faces,3)
            this is not yet implemented
        """
        _REF.faces = faces 
        _REF.verts = vertices
        # We have to clear the vectors to avoid overflow when using the simplify object
        # multiple times
        self.triangles_cpp.clear()
        self.vertices_cpp.clear()
        self.normals_cpp.clear()
        # Here we will need some checks, just to make sure the right objets are passed
        self.faces_mv = faces.astype(dtype="int32", subok=False, copy=False)
        self.vertices_mv = vertices.astype(dtype="float64", subok=False, copy=False)
        self.triangles_cpp = setFacesNogil(self.faces_mv, self.triangles_cpp)
        self.vertices_cpp = setVerticesNogil(self.vertices_mv, self.vertices_cpp)
        setMeshFromExt(self.vertices_cpp, self.triangles_cpp)

    cpdef void simplify_mesh(self, int target_count = 100, int update_rate = 5, 
        double aggressiveness=7., max_iterations = 100, bool verbose=True,  
        bool lossless = False, double threshold_lossless=1e-3, double alpha = 1e-9, 
        int K = 3, bool preserve_border = True):
        """Simplify mesh

            Parameters
            ----------
            target_count : int
                Target number of triangles, not used if lossless is True
            update_rate : int
                Number of iterations between each update. 
                If lossless flag is set to True, rate is 1
            aggressiveness : float
                Parameter controlling the growth rate of the threshold at each 
                iteration when lossless is False. 
            max_iterations : int
                Maximal number of iterations 
            verbose : bool
                control verbosity
            lossless : bool
                Use the lossless simplification method 
            threshold_lossless : float
                Maximal error after which a vertex is not deleted, only for 
                lossless method. 
            alpha : float 
                Parameter for controlling the threshold growth
            K : int 
                Parameter for controlling the thresold growth
            preserve_border : Bool
                Flag for preserving vertices on open border

            Note
            ----
            threshold = alpha*pow( iteration + K, agressiveness)
        """
        N_start = self.faces_mv.shape[0]
        t_start = _time()
        simplify_mesh(target_count, update_rate, aggressiveness, verbose, max_iterations, alpha, K,
                      lossless, threshold_lossless, preserve_border)
        t_end = _time()
        N_end = getFaces().size()

        if verbose:
            print('simplified mesh in {} seconds from {} to {} triangles'.format(
                round(t_end-t_start,4), N_start, N_end)
            )

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef vector[vector[double]] setVerticesNogil(double[:,:] vertices, vector[vector[double]] vector_vertices )nogil:
    """nogil function for filling the vector of vertices, "vector_vertices",
    with the data found in the memory view of the array "vertices" 
    """
    cdef vector[double] vertex 
    vector_vertices.reserve(vertices.shape[0])

    cdef size_t i = 0
    cdef size_t j = 0
    for i in range(vertices.shape[0]):
        vertex.clear()
        for j in range(3):  
            vertex.push_back(vertices[i,j])
        vector_vertices.push_back(vertex)
    return vector_vertices

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef vector[vector[int]] setFacesNogil(int[:,:] faces, vector[vector[int]] vector_faces )nogil:
    """nogil function for filling the vector of faces, "vector_faces",
    with the data found in the memory view of the array "faces"
    """
    cdef vector[int] triangle 
    vector_faces.reserve(faces.shape[0]);

    cdef size_t i = 0
    cdef size_t j = 0
    for i in range(faces.shape[0]):
        triangle.clear()
        for j in range(3):  
            triangle.push_back(faces[i,j])
        vector_faces.push_back(triangle)
    return vector_faces





"""Example:

#We assume you have a numpy based mesh processing software
#Where you can get the vertices and faces of the mesh as numpy arrays.
#For example Trimesh or meshio
import pyfqmr
import trimesh as tr
bunny = tr.load_mesh('Stanford_Bunny_sample.stl')
#Simplify object
simp = pyfqmr.Simplify()
simp.setMesh(bunny.vertices, bunny.faces)
simp.simplify_mesh(target_count = 1000, aggressiveness=7, preserve_border=True, verbose=10)
vertices, faces, normals = simp.getMesh()
"""

"""Example2:
import trimesh as tr
import pyfqmr as fmr
mesh = tr.load_mesh('Stanford_Bunny_sample.stl')
simpl = fmr.Simplify()
verts, faces = mesh.vertices, mesh.faces
simpl.setMesh(verts, faces)
simpl.getMesh()
faces.shape
"""
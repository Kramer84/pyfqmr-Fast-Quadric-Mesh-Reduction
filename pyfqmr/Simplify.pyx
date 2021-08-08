# distutils: language = c++

from libcpp.vector cimport vector
from libcpp cimport bool

from time import time

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

    def getMesh(self):
        """Gets the mesh from the simplify object once the simplification is done

        Returns
        -------
        verts : numpy.ndarray
            array of vertices of shape (n_vertices,3)
        faces : numpy.ndarray
            array of vertices of shape (n_faces,3)
        norms : numpy.ndarray
            array of vertices of shape (n_faces,3)
        """
        self.triangles_cpp = getFaces()
        self.vertices_cpp = getVertices()
        self.normals_cpp = getNormals()
        N_t = self.triangles_cpp.size()
        N_v = self.vertices_cpp.size()
        N_n = self.normals_cpp.size()
        faces = self.faces_mv.copy()[:N_t, :] 
        verts = self.vertices_mv.copy()[:N_v, :] 
        norms = self.vertices_mv.copy()[:N_n, :] 
        for i in range(N_v):
            for j in range(3):
                verts[i,j] = self.vertices_cpp[i][j]
        for i in range(N_t):
            for j in range(3):
                faces[i,j] = self.triangles_cpp[i][j]
                norms[i,j] = self.normals_cpp[i][j]
        return verts, faces, norms

    cpdef void setMesh(self, vertices, faces, face_colors=None):
        # Here we will need some checks, just to make sure the right objets are passed
        self.faces_mv = faces.astype(dtype="int32", subok=False, copy=False)
        self.vertices_mv = vertices.astype(dtype="float64", subok=False, copy=False)
        print('Faces and vertices passed')
        self.triangles_cpp = setFacesNogil(self.faces_mv, self.triangles_cpp)
        self.vertices_cpp = setVerticesNogil(self.vertices_mv, self.vertices_cpp)
        print('setting mesh from python extension')
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
        t_start = time()
        simplify_mesh(target_count, update_rate, aggressiveness, verbose, max_iterations, alpha, K,
                      lossless, threshold_lossless, preserve_border)
        t_end = time()
        print('simplified mesh in {} seconds '.format(round(t_end-t_start,4), N_start, target_count))


cdef vector[vector[double]] setVerticesNogil(double[:,:] vertices, vector[vector[double]] vector_vertices )nogil:
    """nogil function for filling the vector of vertices, "vector_vertices",
    with the data found in the memory view of the array "vertices" 
    """
    cdef vector[double] vertex 
    for i in range(vertices.shape[0]):
        vertex.clear()
        for j in range(3):  
            vertex.push_back(vertices[i,j])
        vector_vertices.push_back(vertex)
    return vector_vertices

cdef vector[vector[int]] setFacesNogil(int[:,:] faces, vector[vector[int]] vector_faces )nogil:
    """nogil function for filling the vector of faces, "vector_faces",
    with the data found in the memory view of the array "faces"
    """
    cdef vector[int] triangle 
    for i in range(faces.shape[0]):
        triangle.clear()
        for j in range(3):  
            triangle.push_back(faces[i,j])
        vector_faces.push_back(triangle)
    return vector_faces




"""Example:

import trimesh as tr
import pyfqmr
bunny = tr.load_mesh('Stanford_Bunny_sample.stl')
simp = pyfqmr.Simplify()
simp.setMesh(bunny.vertices, bunny.faces)
"""
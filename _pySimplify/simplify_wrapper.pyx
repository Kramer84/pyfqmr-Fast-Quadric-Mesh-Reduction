# distutils: language = c++

from libcpp.vector cimport vector
from libcpp cimport bool

import cython
from cython.parallel cimport prange

import numpy as np 
cimport numpy as np
from numpy import int32,float64, signedinteger, unsignedinteger

from time import time

import trimesh as tr

cdef extern from "Simplify.h" namespace "Simplify" :
    void simplify_mesh( int target_count, int update_rate, double aggressiveness, 
                        bool verbose, int max_iterations,double alpha, int K, 
                        bool lossless, double threshold_lossless, bool preserve_border)
    void setMeshFromExt(vector[vector[double]] vertices, vector[vector[int]] faces)
    vector[vector[int]] getFaces()
    vector[vector[double]] getVertices()
    vector[vector[double]] getNormals()


cdef class pySimplify : 

    cdef int[:,:] faces_mv
    cdef double[:,:] vertices_mv
    cdef double[:,:] normals_mv

    cdef vector[vector[int]] triangles_cpp
    cdef vector[vector[double]] vertices_cpp
    cdef vector[vector[double]] normals_cpp
    
    def __cinit__(self):
        pass

    def getMesh(self):
        self.triangles_cpp = getFaces()
        self.vertices_cpp = getVertices()
        self.normals_cpp = getNormals()
        N_t = self.triangles_cpp.size()
        N_v = self.vertices_cpp.size()
        N_n = self.normals_cpp.size()
        faces = np.zeros((N_t,3), dtype=int32)
        verts = np.zeros((N_v,3), dtype=float64)
        norms = np.zeros((N_n,3), dtype=float64)
        for i in range(N_v):
            for j in range(3):
                verts[i,j] = self.vertices_cpp[i][j]
        for i in range(N_t):
            for j in range(3):
                faces[i,j] = self.triangles_cpp[i][j]
                norms[i,j] = self.normals_cpp[i][j]

        mesh=tr.Trimesh(vertices=verts, faces=faces, face_normals=norms)
        return mesh

    cpdef void setMesh(self, trimeshMesh):
        cdef vector[int] triangle
        cdef vector[double] vertex 
        if hasattr(trimeshMesh,'vertrices')==False and hasattr(trimeshMesh,'triangles'):
            trimeshMesh=tr.Trimesh(**tr.triangles.to_kwargs(trimeshMesh.triangles))
        elif hasattr(trimeshMesh,'vertrices')==False and hasattr(trimeshMesh,'faces'):
            pass 
        else :
            try :
                trimeshMesh = list(trimeshMesh.geometry.values())[0]
                trimeshMesh.merge_vertices()
            except :
                print('You have to pass a Trimesh object having either faces and vertices or triangles')
                print('not',trimeshMesh.__class__.__name__)
                raise TypeError
        self.faces_mv = np.array(trimeshMesh.faces, copy=True, subok= False, dtype=int32)
        self.vertices_mv = np.array(trimeshMesh.vertices, copy=True, subok= False, dtype=float64) 
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
        t_start = time()
        simplify_mesh(target_count, update_rate, aggressiveness, verbose, max_iterations, alpha, K,
                      lossless, threshold_lossless, preserve_border)
        t_end = time()
        print('simplified mesh in {} seconds '.format(round(t_end-t_start,4), N_start, target_count))


cdef vector[vector[double]] setVerticesNogil(double[:,:] vertices, vector[vector[double]] vector_vertices )nogil:
    cdef vector[double] vertex 
    for i in range(vertices.shape[0]):
        vertex.clear()
        for j in range(3):  
            vertex.push_back(vertices[i,j])
        vector_vertices.push_back(vertex)
    return vector_vertices

cdef vector[vector[int]] setFacesNogil(int[:,:] faces, vector[vector[int]] vector_faces )nogil:
    cdef vector[int] triangle 
    for i in range(faces.shape[0]):
        triangle.clear()
        for j in range(3):  
            triangle.push_back(faces[i,j])
        vector_faces.push_back(triangle)
    return vector_faces



'''
from pySimplify import pySimplify
import trimesh as tr 
mesh = tr.load_mesh('Stanford_Bunny_sample.stl')

simplify = pySimplify()
simplify.setMesh(mesh)
simplify.simplify_mesh(100, preserve_border=True)
mesh_simple = simplify.getMesh()

'''


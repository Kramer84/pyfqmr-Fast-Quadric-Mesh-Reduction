# distutils: language = c++

from libcpp.vector cimport vector

import cython
from cython.parallel cimport prange

import numpy as np 
cimport numpy as np
from numpy import int32,float64, signedinteger, unsignedinteger

from time import time

import trimesh as tr

cdef extern from "Simplify.h" namespace "Simplify" :
	void simplify_mesh( int target_count, double aggressiveness, verbose)
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
		self.faces_mv = np.array(trimeshMesh.faces, copy=False, subok= True, dtype=int32)
		self.vertices_mv = np.array(trimeshMesh.vertices, copy=False, subok= True, dtype=float64)  
		for i in range(self.faces_mv.shape[0]):
			triangle.clear()
			for j in range(3):	
				triangle.push_back(self.faces_mv[i,j])
			self.triangles_cpp.push_back(triangle)
		for i in range(self.vertices_mv.shape[0]):
			vertex.clear()
			for j in range(3):	
				vertex.push_back(self.vertices_mv[i,j])
			self.vertices_cpp.push_back(vertex)
		setMeshFromExt(self.vertices_cpp, self.triangles_cpp)

	cpdef void simplify_mesh(self, target_count = 15000, aggressiveness=7, verbose=10):
		N_start = self.faces_mv.shape[0]
		t_start = time()
		simplify_mesh(target_count, aggressiveness, verbose)
		t_end = time()
		print('simplified mesh in {} seconds '.format(round(t_end-t_start,4), N_start, target_count))

'''
import simplify_wrapper as sw

test = sw.CySimplify()
test.setMesh()
'''


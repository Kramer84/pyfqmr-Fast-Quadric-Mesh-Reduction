#distutils: sources = Simplify.h
import cython

'''cimport numpy as np 
import  numpy as np
import  trimesh'''



cdef extern from "SimplifyClass.cpp" namespace "SimplifyPy":
    cdef cppclass SimplifyMeshReduction:
        SimplifyMeshReduction() except + 
        std::vector<double> verts_vect;
        std::vector<int>    faces_vect;
        void set_values_from_CArray(double* verts_C, int len_verts,
                                    int* faces_C, int len_faces)
        void clear_Mesh()
        void simplifyMesh(int target_count, double agressiveness)
        double* getVertsCContiguous()
        int* getFacesCContiguous()
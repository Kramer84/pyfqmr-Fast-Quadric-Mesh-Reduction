import numpy as np 
cimport numpy as np
import cython
from cython import int as cy_int
from cython import double as cy_double 
from cpython cimport array
import array

from libc.math cimport sin, cos, pow, abs, sqrt, acos, fmax, fmin

from numpy import int32,float64
from numpy cimport int32_t, float64_t
import trimesh as tr


mesh = tr.load_mesh('Stanford_Bunny_sample.stl')



cdef class Faces :
    cdef public int[:,:] v_mv #array of ints indicating the points index? Memory view
    cdef public double[:,:] err_mv #array of vertex errors # memoryview
    cdef public int[:] deleted_mv, dirty_mv #array to know which triangle was deleted #Memory View
    cdef public double[:,:] n_mv #array of face normal #memory view
    cdef np.ndarray v_dat #array holding the actual data
    cdef np.ndarray err_dat #array holding the actual data
    cdef np.ndarray deleted_dat, dirty_dat #array holding the actual data
    cdef np.ndarray n_dat #array holding the actual data
    cdef Py_ssize_t N #length of Faces
    

    def __cinit__(self, Py_ssize_t N= len(mesh.faces),
                  np.ndarray faces=mesh.faces):
        self.N = N
        self.v_dat = np.asarray(faces, dtype=int32)
        self.err_dat = np.zeros((self.N,4), dtype=float64)
        self.deleted_dat = np.zeros((self.N), dtype=int32)
        self.dirty_dat = np.zeros((self.N), dtype=int32)
        self.n_dat = np.zeros((self.N,3), dtype=float64)
        self.__init_view__()

    def __init_view__(self):
        self.v_mv = self.v_dat
        self.err_mv = self.err_dat
        self.deleted_mv = self.deleted_dat
        self.dirty_mv = self.dirty_dat
        self.n_mv = self.n_dat

    def get_faces_v_dat(self):
        return self.v_dat


cdef class Vertices :
    cdef public double[:,:] p_mv #array of float coordinates? Memory view
    cdef public double[:,:] q_mv #SymetricMatrices # memoryview
    cdef public int[:] tstart_mv, tcount_mv, border_mv #array to know which triangle was deleted #Memory View
    cdef np.ndarray p_dat #array holding the actual data
    cdef np.ndarray q_dat #array holding the actual data
    cdef np.ndarray tstart_dat, tcount_dat, border_dat #array holding the actual data
    cdef Py_ssize_t N #length of Vertices

    def __cinit__(self, Py_ssize_t N = len(mesh.vertices),
                  np.ndarray vertices=mesh.vertices):
        self.N = N
        self.p_dat = np.asarray(vertices, dtype=float64)
        self.q_dat = np.zeros((self.N,10), dtype=float64) #symetric matrix as flat array
        self.tstart_dat = np.zeros((self.N), dtype=int32)
        self.tcount_dat = np.zeros((self.N), dtype=int32)
        self.border_dat = np.zeros((self.N), dtype=int32)
        self.__init_view__()

    def __init_view__(self):
        self.p_mv = self.p_dat
        self.q_mv = self.q_dat
        self.tstart_mv = self.tstart_dat
        self.tcount_mv = self.tcount_dat
        self.border_mv = self.border_dat

    def compact_faces(self):
        pass


cdef class int2 :
    cdef public int i1 
    cdef public int i2
    def __cinit__(self,int i1, int i2):
        self.i1 = i1
        self.i2 = i2

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef int2 ccompat_faces_verts(double[:,:] p_mv, double[:,:] q_mv, int[:] tstart_mv, 
                            int[:] tcount_mv, int[:] border_mv, double[:,:] n_mv,
                            int[:,:] v_mv, double[:,:] err_mv, int[:] deleted_mv, 
                            int[:] dirty_mv):
    cdef size_t  dst_verts = 0
    cdef size_t  dst_faces = 0 
    cdef size_t  i, j, n_v_mv, n_p_mv
    cdef int i__vv
    cdef int[2]  output
    tcount_mv[:] = 0
    n_v_mv = v_mv.shape[0]
    for i in range(n_v_mv):
        if deleted_mv[i] == 0 :
            v_mv[dst_faces,:] = v_mv[i,:]
            err_mv[dst_faces,:] = err_mv[i,:]
            deleted_mv[dst_faces] = deleted_mv[i]
            dirty_mv[dst_faces] = dirty_mv[i]
            n_mv[dst_faces,:] = n_mv[i,:]
            dst_faces += 1
            for j in range(3):
                i_vv = v_mv[i,j]
                tcount_mv[i_vv] = 1

    n_p_mv = p_mv.shape[0]
    for i in range(n_p_mv):
        if tcount_mv[i]==1:
            tstart_mv[i] = dst_verts
            p_mv[dst_verts] = p_mv[i]
            q_mv[dst_verts] = q_mv[i]
            tstart_mv[dst_verts] = tstart_mv[i]
            tcount_mv[dst_verts] = tcount_mv[i]
            border_mv[dst_verts] = border_mv[i]
            dst_verts+=1

    for i in range(dst_faces):
        for j in range(3):
            i_vv = v_mv[i,j]
            v_mv[i,j]=tstart_mv[i_vv]  
      
    output = int2(dst_verts,dst_faces)
    return output




cdef class Refs :
    cdef public int[:] tid_mv
    cdef public int[:] tvertex_mv
    cdef np.ndarray tid_dat
    cdef np.ndarray tvertex_dat
    cdef Py_ssize_t N

    def __cinit__(self, Py_ssize_t N =1 ):
        self.N = N
        self.tid_dat = np.zeros((self.N), dtype = int32)
        self.tvertex_dat = np.zeros((self.N), dtype = int32)






cdef class Simplify:

    cdef public Vertices verts
    cdef public Faces faces 
    cdef public Refs refs 
    cdef public int target_count 
    cdef public double agressiveness 
    cdef public int max_iters 

    def __cinit__(self, int target_count, double agressiveness=7, int max_iters = 100):
        self.target_count = target_count 
        self.agressiveness = agressiveness
        self.max_iters = max_iters

    def setVertices(self, Vertices verts):
        self.verts = verts

    def setFaces(self,Faces faces):
        self.faces = faces

    def simplify_mesh(self):
        cdef int deleted_triangles = 0 
        cdef array.array deleted0, deleted1
        cdef int triangle_count 
        cdef Py_ssize_t iteration, i, j, i0, i1
        cdef int len_faces,tstart, tcount
        cdef double[:] v0, v1 #should be pointers
        cdef double threshold
        cdef int[:] t_mv #Triangle
        cdef double[:] p_mv #Vertice 
        cdef np.ndarray t_dat 
        cdef np.ndarray p_dat 
        cdef Py_ssize_t iters = self.max_iters
        cdef Py_ssize_t shpF = self.faces.n_mv.shape[0]
        cdef Py_ssize_t shpV = self.verts.p_mv.shape[0]

        triangle_count = shpF

        self.faces.deleted_mv[:] = 0

        for iteration in range(iters):
            print('Iterantion N°',iteration)
            print('Faces N° :',int(shpF))
            print('Vertices N° :',int(shpV))
            print('deleted_triangles :',deleted_triangles)
            print('\n')

            shpF = self.faces.n_mv.shape[0]
            shpV = self.verts.p_mv.shape[0]

            if triangle_count - deleted_triangles <= self.target_count :
                break

            if iteration%5==0 :
                self.update_mesh(iteration)

            self.faces.dirty_mv[:]=0

            threshold =  0.000000001*pow(float(iteration+3),self.agressiveness)

            for i in range(shpF):
                if self.faces.err_mv[i,3]>threshold:
                    continue
                if self.faces.deleted_mv[i] == 1 :
                    continue 
                if self.faces.dirty_mv[i] == 1 :
                    continue                
                #maybe later cdef function here
                for j in range(3):
                    if self.faces.err_mv[i,j]<threshold:
                        i0 = self.faces.v_mv[i,j] 
                        v0 = self.verts.p_mv[i0,:]
                        i1 = self.faces.v_mv[i,(j+1)%3]
                        v1 = self.verts.p_mv[i1,:]
                        if self.verts.border_mv[i0] != self.verts.border_mv[i1] : 
                            continue 

                        p_dat = np.zeros(3,dtype=float64)
                        p_mv = p_dat[:]
                        print('p_dat before:',p_dat)
                        err = ccalculate_error(self.verts.q_mv,
                                            self.verts.border_mv,
                                            self.verts.p_mv,
                                            i0, i1, p_mv)
                        print('p_dat after:',p_dat)



    def update_mesh(self):
        pass


#compact triangles
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef int iterationSup0(Py_ssize_t N, int[:,:] v_mv, double[:,:] err_mv, int[:] deleted_mv, 
                        int[:] dirty_mv, double[:,:] n_mv ):
    cdef Py_ssize_t dst = 0
    cdef Py_ssize_t shpF = N
    cdef Py_ssize_t i, j
    for i in range(shpF):
        if deleted_mv[i] == 0 :
            for j in range(3) :
                v_mv[dst,j] = v_mv[i,j]
            for j in range(4):
                err_mv[dst,j] = err_mv[i,j]
            deleted_mv[dst] = deleted_mv[i]
            dirty_mv[dst] = dirty_mv[i]
            for j in range(3):
                n_mv[dst,j] = n_mv[i,j]
            dst += 1
    return dst









@cython.boundscheck(False)
@cython.nonecheck(False)
cdef double ccalculate_error(double[:,:] q_mv, int[:] border_mv , double[:,:] p_mv, 
                    Py_ssize_t id_v1, Py_ssize_t id_v2, double[:] p_result) : 
    cdef double[:] q_sm
    cdef Py_ssize_t i 
    cdef int border
    cdef double error = 0.
    cdef double det, det0, det1, det2
    cdef double[3] p1, p2, p3
    cdef double error1, error2, error3

    q_sm = np.zeros((10),dtype=float64)
    for i in range(10):
        q_sm[i] = q_mv[id_v1,i] + q_mv[id_v2,i]

    border = border_mv[id_v1] & border_mv[id_v2] 
    det = cdet(q_sm,0, 1, 2, 1, 4, 5, 2, 5, 7)

    if det != 0 and not border :
        det0 = cdet(q_sm, 1, 2, 3, 4, 5, 6, 5, 7, 8)
        det1 = cdet(q_sm, 0, 2, 3, 1, 5, 6, 2, 7, 8) 
        det2 = cdet(q_sm, 0, 1, 3, 1, 4, 6, 2, 5, 8)
        p_result[0] = -1*pow(det,-1.)*det0
        p_result[1] = pow(det,-1)*det1
        p_result[2] = -1*pow(det,-1)*det2
        error = vertex_error(q_sm, p_result[0], p_result[1], p_result[2])
    else :
        for i in range(3):
            p1[i] = p_mv[id_v1,i]
            p2[i] = p_mv[id_v2,i]
            p3[i] = (p1[i]+p2[i])/2
        error1 = vertex_error(q_sm, p1[0],p1[1],p1[2])
        error2 = vertex_error(q_sm, p2[0],p2[1],p2[2])
        error3 = vertex_error(q_sm, p3[0],p3[1],p3[2])
        error = fmin(error1, fmin(error2, error3))
        if (error1 == error):
            for i in range(3):
                p_result[i]=p1[i]
        if (error2 == error):
            for i in range(3):
                p_result[i]=p2[i]
        if (error3 == error):
            for i in range(3):
                p_result[i]=p3[i]
    print('det is:',float(det) )
    return error



@cython.boundscheck(False)
@cython.nonecheck(False)
cdef double cdet(double[:] mat, int a11, int a12, int a13, int a21, 
                 int a22, int a23, int a31, int a32, int a33) nogil:
    cdef double det    
    det =  mat[a11]*mat[a22]*mat[a33] + mat[a13]*mat[a21]*mat[a32] + mat[a12]*mat[a23]*mat[a31] \
          - mat[a13]*mat[a22]*mat[a31] - mat[a11]*mat[a23]*mat[a32]- mat[a12]*mat[a21]*mat[a33]
    return det

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef double vertex_error(double[:] q, double x, double y, double z) nogil:
    cdef double val
    val = q[0]*x*x + 2*q[1]*x*y + 2*q[2]*x*z + 2*q[3]*x + q[4]*y*y \
           + 2*q[5]*y*z + 2*q[6]*y + q[7]*z*z + 2*q[8]*z + q[9]
    return val

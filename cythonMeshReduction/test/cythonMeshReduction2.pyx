include "cyarray/cyarray.pyx"

import cython
from cython import int as cy_int
from cython.parallel cimport prange
from cython import double as cy_double 
from cpython cimport array
import array

from libc.math cimport sin, cos, pow, abs, sqrt, acos, fmax, fmin

cimport cython

import numpy as np 
cimport numpy as np

from numpy import int32,float64, signedinteger
from numpy cimport int32_t, float64_t
import trimesh as tr


mesh = tr.load_mesh('Stanford_Bunny_sample.stl')



cdef class Faces :
    cdef public Py_ssize_t[:,:] v_mv #array of ints indicating the points index? Memory view
    cdef public double[:,:] err_mv #array of vertex errors # memoryview
    cdef public int[:] deleted_mv, dirty_mv #array to know which triangle was deleted #Memory View
    cdef public double[:,:] n_mv #array of face normal #memory view
    cdef np.ndarray v_dat #array holding the actual data
    cdef np.ndarray err_dat #array holding the actual data
    cdef np.ndarray deleted_dat, dirty_dat #array holding the actual data
    cdef np.ndarray n_dat #array holding the actual data
    cdef Py_ssize_t N #length of Faces
    
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    def __cinit__(self, Py_ssize_t N= len(mesh.faces),
                  np.ndarray faces=mesh.faces):
        self.N = N
        self.v_dat = np.asarray(faces, dtype=signedinteger)
        self.err_dat = np.zeros((self.N,4), dtype=float64)
        self.deleted_dat = np.zeros((self.N), dtype=int32)
        self.dirty_dat = np.zeros((self.N), dtype=int32)
        self.n_dat = np.zeros((self.N,3), dtype=float64)
        self.__init_view__()

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cpdef void __init_view__(self):
        self.v_mv = self.v_dat
        self.err_mv = self.err_dat
        self.deleted_mv = self.deleted_dat
        self.dirty_mv = self.dirty_dat
        self.n_mv = self.n_dat

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cpdef void resize(self, Py_ssize_t N ):
        self.N = N
        self.v_dat = self.v_dat[:N]
        self.err_dat = self.err_dat[:N]
        self.deleted_dat = self.deleted_dat[:N]
        self.dirty_dat = self.dirty_dat[:N]
        self.n_dat = self.n_dat[:N]
        self.__init_view__()


cdef class Vertices :
    cdef public double[:,:] p_mv #array of float coordinates? Memory view
    cdef public double[:,:] q_mv #SymetricMatrices # memoryview
    cdef public int[:] tstart_mv, tcount_mv, border_mv #array to know which triangle was deleted #Memory View
    cdef np.ndarray p_dat #array holding the actual data
    cdef np.ndarray q_dat #array holding the actual data
    cdef np.ndarray tstart_dat, tcount_dat, border_dat #array holding the actual data
    cdef Py_ssize_t N #length of Vertices

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    def __cinit__(self, Py_ssize_t N = len(mesh.vertices),
                  np.ndarray vertices=mesh.vertices):
        self.N = N
        self.p_dat = np.asarray(vertices, dtype=float64)
        self.q_dat = np.zeros((self.N,10), dtype=float64) #symetric matrix as flat array
        self.tstart_dat = np.zeros((self.N), dtype=int32)
        self.tcount_dat = np.zeros((self.N), dtype=int32)
        self.border_dat = np.zeros((self.N), dtype=int32)
        self.__init_view__()

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cpdef void __init_view__(self):
        self.p_mv = self.p_dat
        self.q_mv = self.q_dat
        self.tstart_mv = self.tstart_dat
        self.tcount_mv = self.tcount_dat
        self.border_mv = self.border_dat

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cpdef void resize(self, Py_ssize_t N ):
        self.N = N
        self.p_dat = self.p_dat[:N]
        self.q_dat = self.q_dat[:N]
        self.tstart_dat = self.tstart_dat[:N]
        self.tcount_dat = self.tcount_dat[:N]
        self.border_dat = self.border_dat[:N]
        self.__init_view__()



@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void ccompat_faces_verts(double[:,:] p_mv, double[:,:] q_mv, int[:] tstart_mv, 
                            int[:] tcount_mv, int[:] border_mv, double[:,:] n_mv,
                            int[:,:] v_mv, double[:,:] err_mv, int[:] deleted_mv, 
                            int[:] dirty_mv, int[:] result_mv) nogil:
    cdef Py_ssize_t  dst_verts = 0
    cdef Py_ssize_t  dst_faces = 0 
    cdef Py_ssize_t  i, j, n_v_mv, n_p_mv, i_vv
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
    result_mv[0] = dst_verts
    result_mv[1] = dst_faces





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
        self.__init_view__()

    cpdef void __init_view__(self):
        self.tid_mv = self.tid_dat
        self.tvertex_mv = self.tvertex_dat

    cpdef void resize(self, Py_ssize_t N ):
        self.N = N
        self.tid_dat = self.tid_dat[:N]
        self.tvertex_dat = self.tvertex_dat[:N]
        self.__init_view__()

    cpdef void resize_ng(self, Py_ssize_t N):
        resize_ng(self.tid_mv, N)
        resize_ng(self.tvertex_mv, N)

cdef void resize_ng(int[:] vectA, Py_ssize_t N) nogil:
    vectA = vectA[:N]


cdef class Simplify:

    cdef public Vertices verts
    cdef public Faces faces 
    cdef public Refs refs 
    cdef public int target_count 
    cdef public double agressiveness 
    cdef public int max_iters 

    def __cinit__(self, int target_count, double agressiveness=7, int max_iters = 5):
        self.target_count = target_count 
        self.agressiveness = agressiveness
        self.max_iters = max_iters

    def setVertices(self, Vertices verts):
        self.verts = verts

    def setFaces(self,Faces faces):
        self.faces = faces
        cdef Py_ssize_t N = self.faces.v_mv.shape[0]
        N = N*3
        self.refs = Refs(N)

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cpdef void simplify_mesh(self):
        cdef int deleted_triangles = 0 
        cdef Py_ssize_t_vector* deleted0 = make_Py_ssize_t_vector_with_size(10000) 
        cdef Py_ssize_t_vector* deleted1 = make_Py_ssize_t_vector_with_size(10000)
        cdef int triangle_count 
        cdef Py_ssize_t iteration, i, j, i0, i1
        cdef int[:] tcount_mv, tstart_mv
        cdef double[:] v0, v1 #should be pointers
        cdef double threshold
        cdef int[:] t_mv #Triangle
        cdef double[:] p_mv #Vertice 
        cdef np.ndarray t_dat 
        cdef np.ndarray p_dat 
        cdef Py_ssize_t iters = self.max_iters
        cdef Py_ssize_t shpF = self.faces.n_mv.shape[0]
        cdef Py_ssize_t shpV = self.verts.p_mv.shape[0]
        cdef np.ndarray temporaryArray = np.zeros(10, float64)
        cdef double[:] tempArray_mv = temporaryArray

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
                self.update_mesh(iteration, tempArray_mv)

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
                        p_result_dat = np.zeros(3,dtype=float64)
                        p_result = p_result_dat[:]
                        err = ccalculate_error(self.verts.q_mv,
                                            self.verts.border_mv,
                                            self.verts.p_mv,
                                            i0, i1, tempArray_mv, p_result)
                        tcount_mv = self.vertices.tcount_mv
                        Py_ssize_t_vector_reserve(deleted0, tcount_mv[i0])
                        Py_ssize_t_vector_reserve(deleted1, tcount_mv[i1])

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cpdef void update_mesh(self, int iteration, double [:] tempArray_mv):
        cdef int dst, i, j, k, ofs, _id, tstart,size,val
        cdef double[:] p0_mv, p1_mv, p2_mv
        cdef double[:] n = np.zeros(3,dtype=float64)

        cdef Py_ssize_t N
        cdef Py_ssize_t[:,:] v_mv
        cdef double[:,:] err_mv
        cdef int[:] deleted_mv
        cdef int[:] dirty_mv
        cdef double[:,:] n_mv
        cdef double[:,:] q_mv, p_mv
        cdef np.ndarray p1p0_dat = np.empty((3),dtype=float64)
        cdef double[:] p1p0 = p1p0_dat
        cdef np.ndarray p2p1_dat = np.empty((3),dtype=float64)
        cdef double[:] p2p1 = p2p1_dat
        cdef int[:] tcount_mv, tstart_mv, tid_mv, tvertex_mv

        v_mv = self.faces.v_mv
        err_mv = self.faces.err_mv
        n_mv = self.faces.n_mv
        N = self.faces.v_mv.shape[0]

        if iteration > 0 :
            deleted_mv = self.faces.deleted_mv
            dirty_mv = self.faces.dirty_mv
            dst = iterationSup0(N, v_mv, err_mv, deleted_mv, dirty_mv, n_mv)
            self.faces.resize(dst)

        border_mv = self.verts.border_mv
        if iteration == 0 :
            p_mv = self.verts.p_mv
            q_mv = self.verts.q_mv
            citerationIf0(p_mv, v_mv, q_mv, n_mv, p1p0, 
                    p2p1, err_mv, border_mv, tempArray_mv)

        v_mv = self.faces.v_mv
        N = self.faces.v_mv.shape[0]
        tcount_mv = self.verts.tcount_mv
        tstart_mv = self.verts.tstart_mv
        tstart_mv[:] = 0
        tcount_mv[:] = 0
        cset_tcount(tcount_mv, v_mv, N)
        cset_tstart(tstart_mv, tcount_mv, N)
        tid_mv = self.refs.tid_mv
        tvertex_mv = self.refs.tvertex_mv
        cwrite_refs(tid_mv, tvertex_mv, v_mv, tcount_mv, tstart_mv) 
        self.refs.resize(N)

        if iteration == 0 :
            citerationIf01(tid_mv, tcount_mv, v_mv,
                            tstart_mv, border_mv)    

##############################################################################
##############################################################################

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void cwrite_refs(int[:] tid_mv, int[:] tvertex_mv, Py_ssize_t[:,:] v_mv,
                      int[:] tcount_mv, int[:] tstart_mv) nogil:
    cdef Py_ssize_t i, j, N
    N = v_mv.shape[0]
    for i in range(N):
        for j in range(3):
            tid_mv[tstart_mv[v_mv[i,j]] + tcount_mv[v_mv[i,j]]] = i
            tvertex_mv[tstart_mv[v_mv[i,j]] + tcount_mv[v_mv[i,j]]] = j
            tcount_mv[v_mv[i,j]] += 1 

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void cset_tcount(int[:] tcount_mv, Py_ssize_t[:,:] v_mv, Py_ssize_t N) nogil :
    cdef Py_ssize_t i,j, i_v 
    for i in range(N):
        for j in range(3):
            i_v = v_mv[i,j]
            tcount_mv[i_v] += 1

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void cset_tstart(int[:] tstart_mv,int[:] tcount_mv, Py_ssize_t N) nogil :
    cdef Py_ssize_t i,i_v 
    cdef int tstart = 0
    for i in range(N):
        tstart_mv[i] = tstart 
        tstart_mv[i] += tcount_mv[i]
        tcount_mv[i] = 0

##############################################################################
##############################################################################

cdef bool cflipped(double[:] p_mv_i, Py_ssize_t i0, Py_ssize_t i1,
        double[:] v0, double[:] v1, Py_ssize_t_vector* deleted,
        Py_ssize_t[:] tstart_mv, int[:] tid_mv, int[:] deleted_mv,
        int[:] tstart_mv, int[:] tid_mv, int[:] tvertex_mv,
        double[:,:] v_mv, int[:] tcount_mv, double[:] p_mv[:,:]):
    cdef double[3] d1, d1, d3, n 
    cdef Py_ssize_t s, id1, id2 
    cdef int[:] t 
    cdef Py_ssize_t N_tcount =  tcount_mv[i0]
    cdef Py_ssize_t k, ij
    for k in range(N_tcount):
        ij = tid_mv
        if deleted_mv








##############################################################################
##############################################################################



@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void citerationIf0(double[:,:] p_mv, Py_ssize_t[:,:] v_mv, double[:,:] q_mv, 
                       double[:,:] n_mv, double[:] p1p0, double[:] p2p1,
                       double[:,:] err_mv, int[:] border_mv, double[:] tempArray_mv) nogil:
    cdef Py_ssize_t N = v_mv.shape[0]
    cdef Py_ssize_t i, j, i_mvij, id_v1, id_v2
    cdef double[:] p0, p1, p2, p_temp

    for i in range(N):
        p0 = p_mv[v_mv[i,0],:]
        p1 = p_mv[v_mv[i,1],:]
        p2 = p_mv[v_mv[i,2],:]
        csubtract(p1,p0,p1p0)
        csubtract(p2,p1,p2p1)
        ccross(p1p0, p2p1, n_mv[i])
        cnormalize3d(n_mv[i])
        p_temp = p2p1 #memory space re-use
        for j in range(3):
            i_mvij = v_mv[i,j]
            cinit_q_iteration0(q_mv[i_mvij], n_mv[i],p0)
            id_v1 = v_mv[i,j]
            id_v2 = v_mv[i,(j+1)%3]
            err_mv[i,j] = ccalculate_error(q_mv, border_mv, p_mv, id_v1, id_v2, tempArray_mv, p_temp)
        err_mv[i,3] = min(err_mv[i,0],min(err_mv[i,1], err_mv[i,2]))

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void citerationIf01(int[:] tid_mv, int[:] tcount_mv, Py_ssize_t[:,:] v_mv,
                        int[:] tstart_mv, int[:] border_mv) nogil :
    # is probably buggy.. 
    cdef Py_ssize_t N, ntc, size, val
    cdef Py_ssize_t i, j, k, k_t, ofs, _id

    cdef Py_ssize_t_vector* vcount 
    cdef Py_ssize_t_vector* vids 

    N = v_mv.shape[0]
    border_mv[:] = 0

    vcount = make_Py_ssize_t_vector_with_size(10000)
    vids = make_Py_ssize_t_vector_with_size(10000)

    for i in range(N):
        vcount.used = 0
        vids.used = 0
        ntc = tcount_mv.shape[0]
        for j in range(ntc):
            k_t = tid_mv[tstart_mv[i]+j]
            for k in range(3):
                ofs = 0
                _id = v_mv[k_t, k]
                while ofs - vcount.used < 0 :
                    if vids.v[ofs] == _id :
                        break
                    ofs += 1
                size = vcount.used
                if ofs == size :
                    Py_ssize_t_vector_append(vcount, 1)
                    Py_ssize_t_vector_append(vids, _id)
                else : 
                    vcount.v[ofs] += 1
        size = vcount.used
        for j in range(size):
            if vcount.v[j] == 1:
                val = vids.v[j]
                border_mv[val] = 1
    free_Py_ssize_t_vector(vcount)
    free_Py_ssize_t_vector(vids)

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void cinit_q_iteration0(double[:] q_mv, double[:] n_mv, double[:] p0) nogil :
    #Does the symetric matrix construction and ads it to the initial q matrix
    cdef double dotprod = cdot(n_mv, p0)
    dotprod *= -1 
    q_mv[0] += n_mv[0]*n_mv[0] 
    q_mv[1] += n_mv[0]*n_mv[1]
    q_mv[2] += n_mv[0]*n_mv[2]
    q_mv[3] += n_mv[0]*dotprod
    q_mv[4] += n_mv[1]*n_mv[1]
    q_mv[5] += n_mv[1]*n_mv[2]
    q_mv[6] += n_mv[1]*dotprod
    q_mv[7] += n_mv[2]*n_mv[2]
    q_mv[8] += n_mv[2]*dotprod
    q_mv[9] += dotprod*dotprod 

#compact triangles
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef int iterationSup0(Py_ssize_t N, Py_ssize_t[:,:] v_mv, double[:,:] err_mv, int[:] deleted_mv, 
                        int[:] dirty_mv, double[:,:] n_mv) nogil :
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

##############################################################################
##############################################################################


@cython.boundscheck(False)
@cython.nonecheck(False)
cdef double ccalculate_error(double[:,:] q_mv, int[:] border_mv , double[:,:] p_mv, 
        Py_ssize_t id_v1, Py_ssize_t id_v2, double[:] q_temp, double[:] p_result) nogil: 

    cdef Py_ssize_t i 
    cdef int border
    cdef double error = 0.
    cdef double det, det0, det1, det2
    cdef double[3] p1, p2, p3
    cdef double error1, error2, error3

    for i in range(10):
        q_temp[i] = q_mv[id_v1,i] + q_mv[id_v2,i]

    border = border_mv[id_v1] & border_mv[id_v2] 
    det = cdet(q_temp,0, 1, 2, 1, 4, 5, 2, 5, 7)

    if det != 0 and not border :
        det0 = cdet(q_temp, 1, 2, 3, 4, 5, 6, 5, 7, 8)
        det1 = cdet(q_temp, 0, 2, 3, 1, 5, 6, 2, 7, 8) 
        det2 = cdet(q_temp, 0, 1, 3, 1, 4, 6, 2, 5, 8)
        p_result[0] = -1*pow(det,-1.)*det0
        p_result[1] = pow(det,-1)*det1
        p_result[2] = -1*pow(det,-1)*det2
        error = vertex_error(q_temp, p_result[0], p_result[1], p_result[2])
    else :
        for i in range(3):
            p1[i] = p_mv[id_v1,i]
            p2[i] = p_mv[id_v2,i]
            p3[i] = (p1[i]+p2[i])/2
        error1 = vertex_error(q_temp, p1[0],p1[1],p1[2])
        error2 = vertex_error(q_temp, p2[0],p2[1],p2[2])
        error3 = vertex_error(q_temp, p3[0],p3[1],p3[2])
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
    #print('det is:',float(det) )
    return error

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef double cdot(double[:] vectA, double[:] vectB ) nogil :
    cdef double dotprod = .0
    cdef Py_ssize_t N = vectA.shape[0]
    cdef Py_ssize_t i
    for i in range(N):
        dotprod += vectA[i]*vectB[i]
    return dotprod

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef double cnormalize3d(double[:] vect) nogil :
    cdef double norm 
    norm = sqrt(pow(vect[0],2)+pow(vect[1],2)+pow(vect[2],2))
    vect[0] = vect[0]*pow(norm,-1)
    vect[1] = vect[1]*pow(norm,-1)
    vect[2] = vect[2]*pow(norm,-1)

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void csubtract(double[:] vectA, double[:] vectB, double[:] vectC) nogil:
    cdef Py_ssize_t N = vectA.shape[0]
    cdef Py_ssize_t i 
    for i in range(N):
        vectC[i] = vectA[i] - vectB[i]
    
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void ccross(double[:] vectA, double[:] vectB, double[:] vectC) nogil: 
    vectC[0] = vectA[1] * vectB[2] - vectA[2] * vectB[1]
    vectC[1] = vectA[2] * vectB[0] - vectA[0] * vectB[2]
    vectC[2] = vectA[0] * vectB[1] - vectA[1] * vectB[0]

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



def test():
    faces = Faces()
    verts = Vertices()
    simplify = Simplify(15000)
    simplify.setFaces(faces)
    simplify.setVertices(verts)
    return simplify

'''

import cythonMeshReduction2 as cmr
simplify = cmr.test()

'''


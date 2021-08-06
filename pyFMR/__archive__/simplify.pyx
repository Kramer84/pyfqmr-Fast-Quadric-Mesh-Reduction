include "cyarray/cyarray.pyx"

import cython
from cython import int as cy_int
from cython.parallel cimport prange
from cython import double as cy_double
from libc.string  cimport memcpy

from libc.math cimport sin, cos, pow, abs, sqrt, acos, fmax, fmin

import numpy as np 
cimport numpy as np

from numpy import int32,float64, signedinteger, unsignedinteger
from numpy cimport int32_t, float64_t

import trimesh as tr
mesh = tr.load_mesh('Stanford_Bunny_sample.stl')



cdef class Faces :
    cdef public Py_ssize_t[:,:] v_mv #array of ints indicating the points index? Memory view
    cdef public double[:,:] err_mv #array of vertex errors # memoryview
    cdef public Py_ssize_t[:] deleted_mv, dirty_mv #array to know which triangle was deleted #Memory View
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
        self.deleted_dat = np.zeros((self.N), dtype=signedinteger)
        self.dirty_dat = np.zeros((self.N), dtype=signedinteger)
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
    cdef public Py_ssize_t[:] tstart_mv, tcount_mv, border_mv #array to know which triangle was deleted #Memory View
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
        self.tstart_dat = np.zeros((self.N), dtype=signedinteger)
        self.tcount_dat = np.zeros((self.N), dtype=signedinteger)
        self.border_dat = np.zeros((self.N), dtype=signedinteger)
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
cdef void ccompat_faces_verts(
        double[:,:] p_mv, 
        double[:,:] q_mv, 
        Py_ssize_t[:] tstart_mv,
        Py_ssize_t[:] tcount_mv, 
        Py_ssize_t[:] border_mv, 
        double[:,:] n_mv,
        Py_ssize_t[:,:] v_mv, 
        double[:,:] err_mv, 
        Py_ssize_t[:] deleted_mv,
        Py_ssize_t[:] dirty_mv, 
        Py_ssize_t[:] result_mv )nogil:

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





#cdef class Refs :
#    Py_ssize_t_vector* tid_mv = make_Py_ssize_t_vector_with_size(10000)
#    Py_ssize_t_vector* tvertex_mv = make_Py_ssize_t_vector_with_size(10000)
#    cdef np.ndarray tid_dat
#    cdef np.ndarray tvertex_dat
#    cdef Py_ssize_t N#

#    def __cinit__(self, Py_ssize_t N =1 ):
#        self.N = N
#        self.tid_dat = np.zeros((self.N), dtype = signedinteger)
#        self.tvertex_dat = np.zeros((self.N), dtype = signedinteger)
#        self.__init_view__()#

#    cpdef void __init_view__(self):
#        self.tid_mv = self.tid_dat
#        self.tvertex_mv = self.tvertex_dat#

#    cpdef void resize(self, Py_ssize_t N ):
#        self.N = N
#        self.tid_dat = self.tid_dat[:N]
#        self.tvertex_dat = self.tvertex_dat[:N]
#        self.__init_view__()




cdef class Simplify:

    cdef public Vertices verts
    cdef public Faces faces 
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

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cpdef void simplify_mesh(self):
        # in place of the refs 2 vectors:
        cdef np.ndarray deleted_triangles = np.zeros((1),dtype=int32) 
        cdef int[:] deleted_triangles_mv = deleted_triangles
        cdef Py_ssize_t_vector* tid_mv = make_Py_ssize_t_vector_with_size(10000)
        cdef Py_ssize_t_vector* tvertex_mv = make_Py_ssize_t_vector_with_size(10000)
        cdef Py_ssize_t_vector* deleted0 = make_Py_ssize_t_vector_with_size(10000) 
        cdef Py_ssize_t_vector* deleted1 = make_Py_ssize_t_vector_with_size(10000)
        cdef int triangle_count
        cdef Py_ssize_t iteration, i, j, k, i0, i1, tstart, tcount
        cdef Py_ssize_t[:] tcount_mv, tstart_mv, deleted_mv
        cdef double[:] v0, v1 #should be pointers
        cdef double threshold
        cdef int[:] t_mv #Triangle
        cdef double[:] p_mv_i 
        cdef double[:,:] n_mv, p_mv, q_mv, err_mv   #Vertice 
        cdef Py_ssize_t[:,:] v_mv
        cdef np.ndarray t_dat 
        cdef np.ndarray p_dat 
        cdef Py_ssize_t iters = self.max_iters
        cdef Py_ssize_t shpF = self.faces.n_mv.shape[0]
        cdef Py_ssize_t shpV = self.verts.p_mv.shape[0]
        cdef np.ndarray temporaryArray = np.zeros(10, float64)
        cdef double[:] q_temp = temporaryArray
        cdef np.ndarray p_result_dat = np.zeros(3,dtype=float64)
        cdef double[:] p_result = p_result_dat[:]
        cdef np.ndarray p_resize_dat = np.zeros(2,dtype=signedinteger)
        cdef Py_ssize_t[:] p_resize = p_resize_dat[:]
        cdef Py_ssize_t shapeFaces

        self.faces.deleted_mv[:] = 0

        for iteration in range(iters):
            print('Iterantion N°',iteration)
            print('Faces N° :',PyInt_FromSsize_t(shpF))
            print('Vertices N° :',PyInt_FromSsize_t(shpV))
            print('deleted_triangles[0] :',deleted_triangles[0])
            print('\n')

            shpF = self.faces.n_mv.shape[0]
            shpV = self.verts.p_mv.shape[0]
            triangle_count = shpF
            if triangle_count - deleted_triangles_mv[0] <= self.target_count :
                break

            if iteration%5==0 :
                err_mv = self.faces.err_mv
                n_mv = self.faces.n_mv
                q_mv = self.verts.q_mv
                p_mv = self.verts.p_mv
                deleted_mv = self.faces.deleted_mv
                dirty_mv = self.faces.dirty_mv
                tcount_mv = self.verts.tcount_mv
                tstart_mv = self.verts.tstart_mv
                border_mv = self.verts.border_mv
                v_mv = self.faces.v_mv

                shapeFaces = update_mesh(iteration, q_temp, err_mv, 
                                        n_mv, q_mv, p_mv, deleted_mv, dirty_mv, 
                                        tcount_mv, tstart_mv, border_mv, v_mv, tid_mv, 
                                        tvertex_mv)
                print('new shape is:',shapeFaces)
                self.faces.resize(shapeFaces)

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
                        err = ccalculate_error(self.verts.q_mv,
                                            self.verts.border_mv,
                                            self.verts.p_mv,
                                            i0, i1, q_temp, p_result)
                        if 0 : print("calculating error")
                        tcount_mv = self.verts.tcount_mv
                        if 0 : print('before ram reserve :',deleted0.size)
                        Py_ssize_t_vector_reserve(deleted0, tcount_mv[i0])
                        Py_ssize_t_vector_reserve(deleted1, tcount_mv[i1])

                        tstart_mv = self.verts.tstart_mv
                        deleted_mv = self.faces.deleted_mv
                        n_mv = self.faces.n_mv
                        v_mv = self.faces.v_mv
                        p_mv = self.verts.p_mv
                        q_mv = self.verts.q_mv
                        if 0 : print('reserved ram for deleted 0 and 1:',deleted0.size)
                        if cflipped(i0, i1, deleted0, tstart_mv, tid_mv, deleted_mv, 
                                    tvertex_mv, tcount_mv, n_mv, v_mv, p_mv, p_result) :
                            if 0 : print('first flip')
                            continue 

                        if cflipped(i1, i0, deleted1, tstart_mv, tid_mv, deleted_mv, 
                                    tvertex_mv, tcount_mv, n_mv, v_mv, p_mv, p_result) :
                            if 0 : print('second flip')
                            continue 
                        
                        print('not flipped')
                        # not flipped so edge removal
                        for k in range(3):
                            p_mv[i0,k] = p_result[k]
                        for k in range(10):
                            q_mv[i0,k] = q_mv[i0,k] + q_mv[i1,k]

                        err_mv = self.faces.err_mv

                        tstart = tvertex_mv.used
                        if 1 : print('tstart = tvertex_mv.used is,',tstart)

                        cupdate_triangles(i0, i0, tcount_mv, tstart_mv, 
                                deleted_mv, dirty_mv, border_mv, v_mv, deleted0, 
                                tid_mv, tvertex_mv, p_result, q_temp, err_mv, 
                                q_mv, p_mv, deleted_triangles_mv)

                        cupdate_triangles(i0, i1, tcount_mv, tstart_mv, 
                                deleted_mv, dirty_mv, border_mv, v_mv, deleted1, 
                                tid_mv, tvertex_mv, p_result, q_temp, err_mv, 
                                q_mv, p_mv, deleted_triangles_mv)

                        if 1 : print('After triangle update, t_used is',tvertex_mv.used)
                        tcount = tvertex_mv.used - tstart
                        if tcount <= tcount_mv[i0]:
                            tstart_mv[i0] = tstart_mv[tstart]
                            tvertex_mv[i0] = tvertex_mv[tstart]
                        else :
                            tcount_mv[i0] = tcount
                            break
                if i%200 ==0:    
                    print("triangle_count is",triangle_count,  
                          "deleted_triangles is",deleted_triangles_mv[0], 
                          "self.target_coun is",self.target_count)
                if triangle_count - deleted_triangles_mv[0] <= self.target_count:
                    print('break oks')
                    break 

            q_mv = self.verts.q_mv
            border_mv = self.verts.border_mv
            tcount_mv = self.verts.tcount_mv
            tstart_mv = self.verts.tstart_mv
            deleted_mv = self.faces.deleted_mv
            n_mv = self.faces.n_mv
            v_mv = self.faces.v_mv
            p_mv = self.verts.p_mv
            q_mv = self.verts.q_mv
            dirty_mv  = self.faces.dirty_mv
            if 0 : print('trying resize')
            ccompat_faces_verts(p_mv, q_mv, tstart_mv, tcount_mv, 
                                border_mv, n_mv, v_mv, err_mv, 
                                deleted_mv, dirty_mv, p_resize)
            self.verts.resize(p_resize[0])
            self.faces.resize(p_resize[1])
            if 1 : print('resizing done, verts size is:',self.verts.N)
            if 1 : print('resizing done, faces size is:',self.faces.N)



#@cython.boundscheck(False)
#@cython.nonecheck(False)
cdef int update_mesh(int iteration, 
                     double[:] q_temp,
                     double[:,:] err_mv,
                     double[:,:] n_mv,
                     double[:,:] q_mv,
                     double[:,:] p_mv,                
                     Py_ssize_t[:] deleted_mv,
                     Py_ssize_t[:] dirty_mv,
                     Py_ssize_t[:] tcount_mv,
                     Py_ssize_t[:] tstart_mv,
                     Py_ssize_t[:] border_mv,
                     Py_ssize_t[:,:] v_mv, 
                     Py_ssize_t_vector* tid_mv, 
                     Py_ssize_t_vector* tvertex_mv):

    cdef Py_ssize_t dst, i, j, k, ofs, _id, tstart,size,val
    cdef double[:] p0_mv, p1_mv, p2_mv
    cdef double[:] n = np.zeros(3,dtype=float64)
    cdef Py_ssize_t N

    cdef np.ndarray p1p0_dat = np.empty((3),dtype=float64)
    cdef double[:] p1p0 = p1p0_dat
    cdef np.ndarray p2p1_dat = np.empty((3),dtype=float64)
    cdef double[:] p2p1 = p2p1_dat
    cdef Py_ssize_t N_0, N_1, N_01 


    N = v_mv.shape[0]

    if iteration > 0 :
        dst = iterationSup0(N, v_mv, err_mv, deleted_mv, dirty_mv, n_mv)
        #self.faces.resize(dst) ######## instead of resizing we keep dst, we attribute it to N
        N = dst 

    if iteration == 0 :
        print('By citerationIf0')
        citerationIf0(p_mv, v_mv, q_mv, n_mv, p1p0, 
                p2p1, err_mv, border_mv, q_temp, N)

    N_0 = N*3
    N_1 = N*3
    tstart_mv[:] = 0
    tcount_mv[:] = 0
    cset_tcount(tcount_mv, v_mv, N)
    cset_tstart(tstart_mv, tcount_mv, N)
    cwrite_refs(tid_mv, tvertex_mv, v_mv, tcount_mv, tstart_mv, N)
    Py_ssize_t_vector_reserve(tid_mv, N_0)
    Py_ssize_t_vector_reserve(tvertex_mv, N_1)
    print('N_O is',N_0)
    print('tid size is',tid_mv.size)

    N_01 = p_mv.shape[0]
    if iteration == 0 :
        print('By citerationIf01')
        citerationIf01(tid_mv, tcount_mv, v_mv,
                        tstart_mv, border_mv, N_01)
    return N 

##############################################################################
##############################################################################

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void cwrite_refs(Py_ssize_t_vector* tid_mv, Py_ssize_t_vector* tvertex_mv, 
        Py_ssize_t[:,:] v_mv, Py_ssize_t[:] tcount_mv, Py_ssize_t[:] tstart_mv, Py_ssize_t N) nogil:
    cdef Py_ssize_t i, j
    for i in range(N):
        for j in range(3):
            tid_mv.v[tstart_mv[v_mv[i,j]] + tcount_mv[v_mv[i,j]]] = i
            tvertex_mv.v[tstart_mv[v_mv[i,j]] + tcount_mv[v_mv[i,j]]] = j
            tcount_mv[v_mv[i,j]] += 1 

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void cset_tcount(Py_ssize_t[:] tcount_mv, Py_ssize_t[:,:] v_mv, Py_ssize_t N) nogil :
    cdef Py_ssize_t i,j, i_v 
    for i in range(N):
        for j in range(3):
            i_v = v_mv[i,j]
            tcount_mv[i_v] += 1

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void cset_tstart(Py_ssize_t[:] tstart_mv,Py_ssize_t[:] tcount_mv, Py_ssize_t N) nogil :
    cdef Py_ssize_t i,i_v 
    cdef Py_ssize_t tstart = 0
    for i in range(N):
        tstart_mv[i] = tstart 
        tstart_mv[i] += tcount_mv[i]
        tcount_mv[i] = 0

##############################################################################
##############################################################################
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void cupdate_triangles(Py_ssize_t i0, 
                            Py_ssize_t id_verts, 
                            Py_ssize_t[:] tcount_mv, 
                            Py_ssize_t[:] tstart_mv,
                            Py_ssize_t[:] deleted_mv,
                            Py_ssize_t[:] dirty_mv,
                            Py_ssize_t[:] border_mv,
                            Py_ssize_t[:,:] v_mv, 
                            Py_ssize_t_vector* deleted,
                            Py_ssize_t_vector* tid_mv,
                            Py_ssize_t_vector* tvertex_mv,
                            double[:] p_result,
                            double[:] q_temp,
                            double[:,:] err_mv,
                            double[:,:] q_mv,
                            double[:,:] p_mv,
                            int[:] deleted_triangles)  :

    cdef Py_ssize_t k, N, tstart
    cdef Py_ssize_t tid_mv_k

    N      = tcount_mv[id_verts] 
    tstart = tstart_mv[id_verts]
    print('in cupdate_triangles')
    for k in range(N):
        tid_mv_k = tid_mv.v[tstart + k]
        if deleted_mv[tid_mv_k] :
            print('here we are  0')
            continue
        if deleted.v[k] == 1 :
            deleted_mv[tid_mv_k] = 1 
            deleted_triangles[0] += 1 
            print('here we are  1')
            continue 
        v_mv[tid_mv_k,tvertex_mv.v[tstart + k]] = i0
        dirty_mv[tid_mv_k] = 1
        print('calculating')
        err_mv[tid_mv_k, 0] = ccalculate_error(q_mv, border_mv, p_mv, 
                                v_mv[tid_mv_k, 0], v_mv[tid_mv_k, 1], 
                                q_temp, p_result)

        err_mv[tid_mv_k, 1] = ccalculate_error(q_mv, border_mv, p_mv, 
                                v_mv[tid_mv_k, 1], v_mv[tid_mv_k, 2], 
                                q_temp, p_result)

        err_mv[tid_mv_k, 2] = ccalculate_error(q_mv, border_mv, p_mv, 
                                v_mv[tid_mv_k, 2], v_mv[tid_mv_k, 0], 
                                q_temp, p_result)

        err_mv[tid_mv_k, 3] = fmin(err_mv[tid_mv_k, 0], 
                                   fmin(err_mv[tid_mv_k, 1],
                                        err_mv[tid_mv_k, 2]))
        print('tid_mv used is', tid_mv.used)
        Py_ssize_t_vector_append(tid_mv, tid_mv.v[tstart + k ])
        print('tid_mv used is', tid_mv.used)
        tid_mv.used += 1 
        Py_ssize_t_vector_append(tvertex_mv, tvertex_mv.v[tstart + k ])
        tvertex_mv.used += 1 


##############################################################################
##############################################################################
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef bint cflipped(
        Py_ssize_t i0, 
        Py_ssize_t i1, 
        Py_ssize_t_vector* deleted,
        Py_ssize_t[:] tstart_mv, 
        Py_ssize_t_vector* tid_mv, 
        Py_ssize_t[:] deleted_mv,
        Py_ssize_t_vector* tvertex_mv, 
        Py_ssize_t[:] tcount_mv, 
        double[:,:] n_mv,
        Py_ssize_t[:,:] v_mv, 
        double[:,:] p_mv, 
        double[:] p_mv_i ): 

    cdef double_vector* d1
    cdef double_vector* d2 
    cdef double_vector* d3
    cdef double_vector* n 
    cdef Py_ssize_t s, id0, id1 
    cdef double dotprod  
    cdef Py_ssize_t N_tcount =  tcount_mv[i0]
    cdef Py_ssize_t k, ij, j 

    if 0 : print('flipping')
    d1 = make_double_vector()
    d2 = make_double_vector()
    d3 = make_double_vector()
    n  = make_double_vector() 
    if 0 : print('d1.size is',d1.size)
    if 0 : print('d1.used is',d1.used)
    for k in range(N_tcount):
        if 0 : print('tstart_mv[i0] is:',tstart_mv[i0],'k is:',k)
        if 0 : print('tid_mv.size is:', tid_mv.size, 'tid_mv.used is:',tid_mv.used)
        ij = tid_mv.v[tstart_mv[i0] + k]
        if 0 :print('ij is',ij)
        if deleted_mv[ij] == 1 :
            continue 
        s = tvertex_mv.v[tstart_mv[i0] + k]
        id0 = v_mv[ij,(s+1)%3]
        id1 = v_mv[ij,(s+2)%3]

        if id0==i0 or id1==i1 : #delete
            print('was deleted, deleted.v[k]= ',deleted.v[k])
            deleted.v[k] = 1
            print('after deleted.v[k]= ',deleted.v[k])
            continue
        # initialization doubles
        for j in range(3) : 
            d1.v[j] = p_mv[i0,j] - p_mv_i[j]
            d2.v[j] = p_mv[i1,j] - p_mv_i[j]
            d3.v[j] = n_mv[ij,j]
        cnormalize_double_vector3d(d1)
        cnormalize_double_vector3d(d2)
        if 0 : print('after initialization and normalization:')
        if 0 : print('d1.v[0]=',d1.v[0],'d1.v[1]=',d1.v[1],'d1.v[2]=',d1.v[2])
        dotprod = cdot_double_vector3d(d1, d2)
        if 0 : print('dotprod is',dotprod)
        if 0 : print('after d1.size is',d1.size)
        if 0 : print('after d1.used is',d1.used)
        if fabs(dotprod)>0.999:
            free_double_vector(d1)
            free_double_vector(d2)
            free_double_vector(d3)
            return 1 
        ccross_double_vector3d(d1, d2, n)
        cnormalize_double_vector3d(n)
        deleted.v[k] = 1
        dotprod = cdot_double_vector3d(n, d3)
        if 0 :print('other dotprod = ',dotprod)
        if dotprod < .2 :
            free_double_vector(d1)
            free_double_vector(d2)
            free_double_vector(d3)
            return 1
    free_double_vector(d1)
    free_double_vector(d2)
    free_double_vector(d3)
    return 0


@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void cnormalize_double_vector3d(
        double_vector* vect )nogil:

    cdef double norm 

    norm = sqrt(pow(vect.v[0],2)+pow(vect.v[1],2)+pow(vect.v[2],2))
    vect.v[0] = vect.v[0]*pow(norm,-1)
    vect.v[1] = vect.v[1]*pow(norm,-1)
    vect.v[2] = vect.v[2]*pow(norm,-1)


@cython.boundscheck(False)
@cython.nonecheck(False)
cdef double cdot_double_vector3d(
        double_vector* vectA, 
        double_vector* vectB )nogil:

    cdef double dotprod = .0
    cdef Py_ssize_t i

    for i in range(3):
        dotprod += vectA.v[i]*vectB.v[i]
    return dotprod


@cython.boundscheck(False)
@cython.nonecheck(False)
cdef double ccross_double_vector3d(
        double_vector* vectA, 
        double_vector* vectB, 
        double_vector* vectC )nogil:

    vectC.v[0] = vectA.v[1] * vectB.v[2] - vectA.v[2] * vectB.v[1]
    vectC.v[1] = vectA.v[2] * vectB.v[0] - vectA.v[0] * vectB.v[2]
    vectC.v[2] = vectA.v[0] * vectB.v[1] - vectA.v[1] * vectB.v[0]

##############################################################################
##############################################################################

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void citerationIf0(
        double[:,:] p_mv, 
        Py_ssize_t[:,:] v_mv, 
        double[:,:] q_mv,
        double[:,:] n_mv, 
        double[:] p1p0, 
        double[:] p2p1, 
        double[:,:] err_mv, 
        Py_ssize_t[:] border_mv, 
        double[:] q_temp, 
        Py_ssize_t N )nogil:

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
            err_mv[i,j] = ccalculate_error(q_mv, border_mv, p_mv, id_v1, id_v2, q_temp, p_temp)
        err_mv[i,3] = min(err_mv[i,0],min(err_mv[i,1], err_mv[i,2]))

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void citerationIf01(
        Py_ssize_t_vector* tid_mv, 
        Py_ssize_t[:] tcount_mv, 
        Py_ssize_t[:,:] v_mv,
        Py_ssize_t[:] tstart_mv, 
        Py_ssize_t[:] border_mv, 
        Py_ssize_t N) nogil:
     
    cdef Py_ssize_t ntc, size, val
    cdef Py_ssize_t i, j, k, k_t, ofs, _id

    cdef Py_ssize_t_vector* vcount 
    cdef Py_ssize_t_vector* vids 

    border_mv[:] = 0

    vcount = make_Py_ssize_t_vector_with_size(<Py_ssize_t>N*2)
    vids = make_Py_ssize_t_vector_with_size(<Py_ssize_t>N*2)

    for i in range(N):
        vcount.used = 0
        vids.used = 0
        ntc = tcount_mv[i]
        for j in range(ntc):
            k_t = tid_mv.v[tstart_mv[i]+j]
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
                    vcount.used += 1
                    Py_ssize_t_vector_append(vids, _id)
                    vids.used += 1
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
cdef void cinit_q_iteration0(
        double[:] q_mv, 
        double[:] n_mv, 
        double[:] p0 )nogil:

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
cdef int iterationSup0(
        Py_ssize_t N, 
        Py_ssize_t[:,:] v_mv, 
        double[:,:] err_mv, 
        Py_ssize_t[:] deleted_mv, 
        Py_ssize_t[:] dirty_mv, 
        double[:,:] n_mv )nogil:

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
cdef double ccalculate_error(
        double[:,:] q_mv, 
        Py_ssize_t[:] border_mv , 
        double[:,:] p_mv, 
        Py_ssize_t id_v1, 
        Py_ssize_t id_v2, 
        double[:] q_temp, 
        double[:] p_result )nogil: 

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
cdef double cdot(
        double[:] vectA, 
        double[:] vectB )nogil:

    cdef double dotprod = .0
    cdef Py_ssize_t N = vectA.shape[0]
    cdef Py_ssize_t i

    for i in range(N):
        dotprod += vectA[i]*vectB[i]
    return dotprod

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef double cnormalize3d(
        double[:] vect )nogil:
    
    cdef double norm 

    norm = sqrt(pow(vect[0],2)+pow(vect[1],2)+pow(vect[2],2))
    vect[0] = vect[0]*pow(norm,-1)
    vect[1] = vect[1]*pow(norm,-1)
    vect[2] = vect[2]*pow(norm,-1)

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void csubtract(
        double[:] vectA, 
        double[:] vectB, 
        double[:] vectC )nogil:

    cdef Py_ssize_t N = vectA.shape[0]
    cdef Py_ssize_t i 

    for i in range(N):
        vectC[i] = vectA[i] - vectB[i]
    
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void ccross(
        double[:] vectA,
        double[:] vectB,
        double[:] vectC )nogil:

    vectC[0] = vectA[1] * vectB[2] - vectA[2] * vectB[1]
    vectC[1] = vectA[2] * vectB[0] - vectA[0] * vectB[2]
    vectC[2] = vectA[0] * vectB[1] - vectA[1] * vectB[0]

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef double cdet(
        double[:] mat, 
        int a11, int a12, int a13, 
        int a21, int a22, int a23, 
        int a31, int a32, int a33 )nogil:

    cdef double det    

    det =  mat[a11]*mat[a22]*mat[a33] + mat[a13]*mat[a21]*mat[a32] + mat[a12]*mat[a23]*mat[a31] \
          - mat[a13]*mat[a22]*mat[a31] - mat[a11]*mat[a23]*mat[a32]- mat[a12]*mat[a21]*mat[a33]
    return det

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef double vertex_error(
        double[:] q, 
        double x, 
        double y, 
        double z )nogil:

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
simplify.simplify_mesh()

'''


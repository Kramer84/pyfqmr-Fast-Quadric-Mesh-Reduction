# distutils: language=c++

import numpy as np 
import cython
from cython import int as cy_int
from cython import double as cy_double 
from cpython cimport array
import array

from libc.math cimport sin, cos, pow, abs, sqrt, acos, fmax, fmin

from libcpp.vector cimport vector 


from numpy import int32,float64
from numpy cimport int32_t, float64_t
import trimesh as tr

import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer


# cython: boundscheck=False
# cython: cython.wraparound=False

mesh = tr.load_mesh('Stanford_Bunny_sample.stl')


def getFacesVerticesView(mesh=mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    return trimeshMemoryView(vertices, faces)

cdef trimeshMemoryView(vertices, faces):
    cdef int nVerts = np.shape(vertices)[0]
    cdef int edges = 3    
    cdef int nFaces = np.shape(faces)[0]
    cdef int dim = 3    
    vertices = np.array(vertices,dtype=np.dtype(float64))
    faces = np.array(faces,dtype=np.dtype(int32))
    cdef double [:,:] vertices_view = vertices
    cdef int [:,:] faces_view = faces 
    return vertices_view, nVerts, faces_view,  nFaces

cdef vector3d barycentric(vector3d vectP, vector3d vectA, vector3d vectB, vector3d vectC):
    cdef vector3d v0, v1, v2
    cdef double d00, d01, d11, d20, d21, denom, u, v, w
    v0 = vectB - vectA
    v1 = vectC - vectA
    v2 = vectP - vectA 
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00*d11-d01*d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return vector3d(u, v, w)

cdef vector3d interpolate(vector3d vectP, vector3d vectA, vector3d vectB, 
                          vector3d vectC, vector3d attr0, vector3d attr1, vector3d attr2) :
    cdef vector3d bary, out
    out = vector3d()
    bary = barycentric(vectP, vectA, vectB, vectC)
    out = out + attr0 * bary[0]
    out = out + attr1 * bary[1]
    out = out + attr2 * bary[2]
    return out

cdef double mini(double v1, double v2):
    return fmin(v1,v2)



#############################################################################
#############################################################################


cdef class vector3d: 
    cdef public double [:] xyz

    def __cinit__(self, double [:] xyz = np.zeros(3,dtype=float64)):
        self.xyz = xyz

    def __repr__(self):
        return 'Vector of coordinates \nx : {},\ny : {},\nz : {}'.format(round(self[0],4),
                                                                       round(self[1],4),
                                                                       round(self[2],4))

    def __getitem__(self, int idx):
        cdef double val
        val = self.xyz[idx] 
        return val 

    def __setitem__(self, int idx, int val):
        self.xyz[idx] = val

    def __add__(self, vector3d vect):
        return vector3d(self[:]+vect[:])

    def __iadd__(self, vector3d vect):
        self[:]+=vect[:]
        return self

    def __sub__(self, vector3d vect):
        return vector3d(self[:]-vect[:])

    def __isub__(self, vector3d vect):
        self[:]-=vect[:]
        return self

    def __mul__(self, other):
        if isinstance(other, self.__class__) :
            return vector3d(self[:]*other[:])
        else :
            return vector3d(self[:]*other)

    def __imul__(self, other):
        if isinstance(other, self.__class__) :
            self[:]*=other[:]
            return self
        else :
            self[:]*=other
            return self

    def __div__(self, other):
        if isinstance(other, self.__class__) :
            return vector3d(self[:]/other[:])
        elif other != 0 :
            return vector3d(self[:]/other)
        else :
            raise NotImplementedError

    def __idiv__(self, other):
        if isinstance(other, self.__class__) :
            self[:]/=other[:]
            return self
        elif other != 0 :
            self[:]/=other
            return self
        else :
            raise NotImplementedError

    def dot(self, vector3d vect):
        return self[0]*vect[0]+self[1]*vect[1]+self[2]*vect[2]

    def cross(self, vector3d vectA, vector3d vectB):
        self[0] = vectA[1] * vectB[2] - vectA[2] * vectB[1]
        self[1] = vectA[2] * vectB[0] - vectA[0] * vectB[2]
        self[2] = vectA[0] * vectB[1] - vectA[1] * vectB[0]
        return self

    def length(self):
        return sqrt(pow(self[0],2)+pow(self[1],2)+pow(self[2],2))

    def angle(self, vector3d vect):
        cdef double dotprod, lenmul, val
        dotprod = vect[0]*self[0] + vect[1]*self[1] + vect[2]*self[2]
        lenmul = self.length()*vect.length()
        if lenmul==0.:
            lenmul=0.000001
        val = dotprod/lenmul
        if val<-1.: val=-1.
        if val>1.:val=1.
        return acos(val)

    def angle2(self, vector3d vectA, vector3d vectB):
        cdef double dot, lenmul, val
        dot = vectA[0]*vectB[0] + vectA[1]*vectB[1] + vectA[2]*vectB[2]
        lenmul = vectA.length()*vectB.length()
        if lenmul==0:
            lenmul=0.000001
        val = dot/lenmul
        if val<-1: val=-1
        if val>1:val=1
        return acos(val)

    def rot_x(self, double a):
        cdef double yy, zz
        yy = cos(a)*self[1] + sin(a)*self[2]
        zz = cos(a)*self[2] - sin(a)*self[1]
        self[1] = yy
        self[2] = zz
        return self

    def rot_y(self, double a):
        cdef double xx, zz
        xx = cos(-a)*self[0] + sin(-a)*self[2]
        zz = cos(-a)*self[2] - sin(-a)*self[0]
        self[0] = xx
        self[2] = zz
        return self

    def rot_z(self, double a):
        cdef double yy, xx
        yy = cos(a)*self[1] + sin(a)*self[0]
        xx = cos(a)*self[0] - sin(a)*self[1]
        self[1] = yy
        self[0] = xx
        return self

    def clamp(self, double mini, double maxi):
        if self[0] < mini :
            self[0] = mini 
        if self[1] < mini :
            self[1] = mini
        if self[2] < mini :
            self[2] = mini
        if self[0] > maxi :
            self[0] = maxi 
        if self[1] > maxi :
            self[1] = maxi
        if self[2] > maxi :
            self[2] = maxi

    def invert(self):
        self *= -1.
        return self

    def frac(self):
        return vector3d(self[:]-np.asarray(self[:],dtype=int32))

    def integer(self):
        return vector3d(np.asarray(self[:],dtype=int32).astype(float64))        

    def normalize(self):
        cdef double squar_
        squar_ = sqrt(self[0]*self[0]+self[1]*self[1]+self[2]*self[2])
        self[:] /= squar_
        return self


#############################################################################
#############################################################################


cdef class SymetricMatrix(object):
    cdef double [10] mat

    def __cinit__(self, 
            double m11=.0, double m12=.0, double m13=.0, double m14=.0, 
                           double m22=.0, double m23=.0, double m24=.0, 
                                          double m33=.0, double m34=.0, 
                                                         double m44=.0 ):
        self.mat[0] = m11
        self.mat[1] = m12
        self.mat[2] = m13
        self.mat[3] = m14
        self.mat[4] = m22
        self.mat[5] = m23
        self.mat[6] = m24
        self.mat[7] = m33
        self.mat[8] = m34
        self.mat[9] = m44

    def __getitem__(self,int idx):
        return self.mat[idx]

    def __add__(self, SymetricMatrix other):
        return SymetricMatrix( 
            self.mat[0]+other[0], self.mat[1]+other[1], self.mat[2]+other[2], self.mat[3]+other[3],
                                  self.mat[4]+other[4], self.mat[5]+other[5], self.mat[6]+other[6],
                                                        self.mat[7]+other[7], self.mat[8]+other[8],
                                                                              self.mat[9]+other[9])

    def __iadd__(self, SymetricMatrix other): 
        self.mat[0]+=other[0]
        self.mat[1]+=other[1]
        self.mat[2]+=other[2]
        self.mat[3]+=other[3]
        self.mat[4]+=other[4]
        self.mat[5]+=other[5]
        self.mat[6]+=other[6]
        self.mat[7]+=other[7]
        self.mat[8]+=other[8]
        self.mat[9]+=other[9]
        return self

    def det(self, int a11, int a12, int a13,
            int a21, int a22, int a23,
            int a31, int a32, int a33):
        return fast_det(self.mat,a11, a12, a13, a21, a22, a23, a31, a32, a33) 

    @staticmethod
    def makePlane(double a, double b, 
                  double c, double d):
        return SymetricMatrix(a*a, a*b, a*c, a*d, 
                        b*b, b*c, b*d, c*c, c*d, d*d)


cdef double fast_det(double [10] mat, int a11, int a12, int a13,
                     int a21, int a22, int a23,
                     int a31, int a32, int a33):
    cdef double det 
    det = mat[a11]*mat[a22]*mat[a33] + mat[a13]*mat[a21]*mat[a32] \
         + mat[a12]*mat[a23]*mat[a31] - mat[a13]*mat[a22]*mat[a31] \
         - mat[a11]*mat[a23]*mat[a32] - mat[a12]*mat[a21]*mat[a33]
    return det 

#############################################################################
#############################################################################

cdef class Triangle :
    cdef public int [:] v 
    cdef public double [:] err
    cdef public int deleted, dirty
    cdef public vector3d n 

    def __cinit__(self,
                  int [:] v =np.zeros(3, dtype=int32),
                  double [:] err=np.zeros(3, dtype=float64),
                  int deleted=0, 
                  int dirty=0, 
                  vector3d n = vector3d()):
        self.v = v
        self.err = err
        self.deleted = deleted
        self.dirty = dirty
        self.n = n

    def __repr__(self):
        cdef str str0, str1, str2
        str0 = 'Triangle :\n'
        str1 = '  nodes : {} | {} | {}' .format(self.v[0], self.v[1], self.v[2])
        str2 = '\n  deleted : {}\n  dirty : {}\n'.format(bool(self.deleted), bool(self.dirty))
        return str0+str1+str2


#############################################################################
#############################################################################


cdef class Vertex :
    cdef public vector3d v
    cdef public int tstart, tcount 
    cdef public SymetricMatrix q 
    cdef public int border

    def __cinit__(self, vector3d v = vector3d(), int tstart=0, 
        int tcount=0, SymetricMatrix q=SymetricMatrix(), int border=0):
        self.v = v
        self.tstart = tstart
        self.tcount = tcount
        self.q = q
        self.border = border

    def __repr__(self):
        cdef str str0, str1
        str0 = 'Vertex :\n'
        str1 = '  coords : {} | {} | {}\n' .format(round(self.v[0],4),
                                                   round(self.v[1],4),
                                                   round(self.v[2],4))
        return str0+str1


#############################################################################
#############################################################################


cdef class Ref :
    cdef public int tid 
    cdef public int tvertex
    def __cinit__(self, int tid=0, int tvertex=0):
        self.tid = tid
        self.tvertex = tvertex


#############################################################################
#############################################################################


#def make_list():
#    l = []
#    for _ in range(10000):
#        l.append(Acl_cy(1, 2, 3, 'a', 'b', 'c'))
#    return l#

#def loop_typed():
#    cdef list l = make_list()
#    cdef Acl_cy itm
#    cdef int sum = 0
#    for itm in l:
#        sum += itm.s1
#    return sum

cdef list vertsOfView(double [:,:] vertices_view, 
                            int nVerts):
    cdef list vertsList = list()
    cdef int i    
    cdef double x    
    cdef double y
    cdef double z
    for i in range(nVerts):
        vertsList.append(Vertex(v=vector3d(vertices_view[i,:])))
    return vertsList

cdef list facesOfView(int [:,:] faces_view, 
                      int nFaces):
    cdef list facesList = list()
    cdef int j
    for j in range(nFaces):
        facesList.append(Triangle(
                            faces_view[j,:],
                            np.zeros(3, dtype=float64),
                            0,0,vector3d()))
    return facesList              


def makeVertsTrianglesRefs(vertices_view, nVerts, faces_view, nFaces):
    vertsList = vertsOfView(vertices_view, nVerts)
    facesList = facesOfView(faces_view, nFaces)
    return vertsList, facesList

#############################################################################
#############################################################################
cdef array.array int_array_template = array.array('i',[])
cdef array.array one_array = array.array('i',[1])
cdef array.array zero_array = array.array('i',[0])

cdef class dyn_bint_array :
    cdef array.array arr 

    def __cinit__(self, int size=0):
        self.arr = array.clone(int_array_template, size, zero=False)
    
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    def __getitem__(self,int idx):
        return self.arr[idx]

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    def __setitem__(self, int idx, int val):
        self.arr[idx] = val
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    def append(self, int val):
        cdef array.array valArr = array.clone(int_array_template, 1, zero=False)
        valArr[0]=val
        array.extend(self.arr,valArr)

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    def pop(self):
        array.resize(self.arr,len(self.arr)-1)
    
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    def resize(self,size):
        array.resize_smart(self.arr,size)

cdef class np_int_list:
    cdef int [:] np_list 

    def __cinit__(self):
        self.np_list = np.array([],dtype=np.dtype(int32))

    def append(self, int val):
        self.np_list = np.append(self.np_list,val)

    def pop(self):
        self.np_list = self.np_list[:-1]


#############################################################################
#############################################################################

#cdef bool flipped(list triangles, vector3d p, int i0, int i1, Vertex& v0, Vertex& v1, list& deleted)
#    cdef int i 
#    for i in range(vO.tcount):
#        cdef Triangle t = triangles[]

cdef class Simplify:
    cdef public list vertices 
    cdef public list faces            
    cdef public list refs 
    cdef public int target_count 
    cdef public double agressiveness 
    cdef public int max_iters 

    def __cinit__(self, int target_count, double agressiveness=7, int max_iters = 100):
        self.target_count = target_count 
        self.agressiveness = agressiveness
        self.max_iters = max_iters

    def setVertices(self, list vertices):
        self.vertices = vertices

    def setFaces(self, list faces):
        self.faces = faces

    def simplify_mesh(self):
        cdef int deleted_triangles = 0 
        cdef vector[int] deleted0, deleted1
        cdef int triangle_count = len(self.faces)
        cdef int iteration, i, j, i0, i1, len_faces,tstart
        cdef Vertex v0, v1 #should be pointers
        cdef double threshold
        cdef Triangle t
        cdef vector3d p
        for i in range(triangle_count):
            self.faces[i].deleted = 0

        for iteration in range(self.max_iters):
            len_faces = len(self.faces)
            if triangle_count - deleted_triangles <= self.target_count :
                break

            if iteration%5==0 :
                self.update_mesh()

            for i in range(len_faces):
                self.faces[i].dirty = 0 

            threshold =  0.000000001*pow(float(iteration+3),self.agressiveness)


            for i in range(len_faces):
                t = self.faces[i]
                if t.err[3]>threshold:
                    continue
                if t.deleted :
                    continue 
                if t.dirty :
                    continue 
                for j in range(3):
                    if t.err[j]<threshold :
                        i0 = t.v[j] 
                        v0 = self.vertices[i0]
                        i1 = t.v[(j+1)%3]
                        v1 = self.vertices[i1]
                        if v0.border != v1.border : 
                            continue
                        self.calculate_error(i0,i1,p)
                        deleted0.resize(v0.tcount) # normals temporarily
                        deleted1.resize(v1.tcount)
                        if self.flipped(p,i0,i1,v0,v1,deleted0) :
                            continue
                        if self.flipped(p,i1,i0,v1,v0,deleted1) :
                            continue
                        p = vector3d()
                        v0.p=p
                        v0.q=v1.q+v0.q
                        tstart=self.refs.size()

                        self.update_triangles(i0,v0,deleted0,deleted_triangles)
                        self.update_triangles(i0,v1,deleted1,deleted_triangles)

    
    def update_mesh(self, int iteration):
        cdef int dst, i, j, tstart
        cdef Triangle t
        cdef vector3d p0, p1, p2, p 
        cdef vector3d n = vector3d()
        if iteration>0:
            dst = 0 
            for i in range(len(self.triangles)):
                if self.triangles[i].deleted != 0 :
                    self.triangles[dst]=self.triangles[i]
                    dst += 1
            self.triangles = self.triangles[:dst]

        if iteration == 0 :
            for i in range(len(self.vertices)):
                self.vertices[i].q = SymetricMatrix()
            for i in range(len(self.triangles)):
                p0 = self.vertices[self.triangles[i].v[0]].p
                p1 = self.vertices[self.triangles[i].v[1]].p
                p2 = self.vertices[self.triangles[i].v[2]].p
                n.cross(p1-p0,p2-p1)
                n.normalize()
                self.triangles[i].n = n
                for j in range(3):
                    self.vertices[self.triangles[i].v[j]].q += SymetricMatrix(n[0],n[1],n[2],-n.dot(p0))
            for i in range(len(self.triangles)):
                p = vector3d()
                for j in range(3):
                    self.faces[i].err[j], _ =self.calculate_error(self.faces[i].v[j],self.faces[i].v[(j+1)%3],p)
                self.faces[i].err[3]=min(min(self.faces[i].err[0],min(self.faces[i].err[1],self.faces[i].err[2])).err[0])
        for i in range(len(self.vertices)):
            self.vertices[i].tstart = 0
            self.vertices[i].tcount = 0
        for i in range(len(self.triangles)):
            for j in range(3):
                self.vertices[self.triangles[i].v[j]].tcount += 1
        tstart = 0 
        for i in range(len(self.vertices)):
            self.vertices[i].tstart=tstart
            self.vertices[i].tstart+=self.vertices[i].tcount
            self.vertices[i].tcount = 0 
        self.refs = [Ref()]*(len(self.triangles)**3)














    def update_triangles(self,int i0, Vertex v, vector[int]& deleted, int& deleted_triangles):
        cdef vector3d p = vector3d()
        cdef Triangle 
        cdef int k 
        cdef Ref r
        for k in range(v.tcount):
            r = self.refs[v.tstart+k]
            t = self.triangles[r.tid]
            if t.deleted :
                continue
            if deleted[k] == 1 :
                t.deleted = 1
                deleted_triangles += 1
                continue
            t.v[r.tvertex]=i0
            t.dirty = 1
            t.err[0],p=self.calculate_error(t.v[0],t.v[1],p)
            t.err[1],p=self.calculate_error(t.v[1],t.v[2],p)
            t.err[2],p=self.calculate_error(t.v[2],t.v[0],p)
            t.err[3]=min(t.err[0],min(t.err[1],t.err[2]))
            self.refs.append(r)            



    def flipped(self, vector3d p,int i0,int i1,Vertex v0,Vertex v1,vector[int]& deleted):
        cdef vector3d d1, d2, n
        cdef int s, id1, id2
        cdef Triangle t
        for k in range(v0.tcount):
            t=self.triangles[self.refs[v0.tstart+k].tid]
            if t.deleted :
                continue
            s= self.refs[v0.tstart+k].tvertex
            id1=t.v[(s+1)%3]
            id2=t.v[(s+2)%3]

            if(id1==i1 or id2==i1): # delete 
                deleted[k]=1
                continue
            d1 = self.vertices[id1].p-p
            d1.normalize()
            d2 = self.vertices[id2].p-p
            d2.normalize()
            if abs(d1.dot(d2))>0.999:
                return True
            n = vector3d()
            n.cross(d1,d2)
            n.normalize()
            deleted[k]=0
            if n.dot(t.n)<0.2 :
                return True
        return False

    def calculate_error(self, int id_v1, int id_v2, vector3d p_result):
        cdef outputTuple output = ccalculate_error(self.vertices, id_v1, id_v2, p_result)
        return output.error, output.vect

cdef class outputTuple:
    cdef vector3d vect 
    cdef double error
    def __cinit__(self, vector3d vect, double error):
        self.vect = vect
        self.error = error

cdef outputTuple ccalculate_error(list vertices, int id_v1, int id_v2, vector3d p_result):
    cdef outputTuple out 
    cdef SymetricMatrix q = vertices[id_v1].q + vertices[id_v2].q
    cdef int border = vertices[id_v1].border & vertices[id_v2].border 
    cdef double error = 0.
    cdef double det = q.det(0, 1, 2, 1, 4, 5, 2, 5, 7)
    cdef vector3d p1, p2, p3
    cdef double error1, error2, error3
    if det != 0 and not border :
        p_result[0] = -1/det*(q.det(1, 2, 3, 4, 5, 6, 5, 7 , 8))
        p_result[1] =  1/det*(q.det(0, 2, 3, 1, 5, 6, 2, 7 , 8))
        p_result[2] = -1/det*(q.det(0, 1, 3, 1, 4, 6, 2, 5,  8))
        error = vertex_error(q, p_result[0], p_result[1], p_result[2])
    else :
        p1=vertices[id_v1].p
        p2=vertices[id_v2].p
        p3=(p1+p2)/2
        error1 = vertex_error(q, p1.x,p1.y,p1.z)
        error2 = vertex_error(q, p2.x,p2.y,p2.z)
        error3 = vertex_error(q, p3.x,p3.y,p3.z)
        error = min(error1, min(error2, error3))
        if (error1 == error):
            p_result=p1
        if (error2 == error):
            p_result=p2
        if (error3 == error):
            p_result=p3
    out = outputTuple(p_result,error)
    return out

#Error between vertex and Quadric
cdef double vertex_error(SymetricMatrix q, double x, double y, double z):
    return q[0]*x*x + 2*q[1]*x*y + 2*q[2]*x*z + 2*q[3]*x + q[4]*y*y \
           + 2*q[5]*y*z + 2*q[6]*y + q[7]*z*z + 2*q[8]*z + q[9]















#https://cython.readthedocs.io/en/latest/src/userguide/pyrex_differences.html










@timer
def getFacesAndVertsList():
    vertices_view, nVerts, faces_view, nFaces = getFacesVerticesView()
    vertsList, facesList= makeVertsTrianglesRefs(vertices_view, nVerts, faces_view, nFaces)
    return vertsList, facesList


'''

import numpy as np

from time import time 
def testAppPop(flist,N):
    t0=time()
    for i in np.arange(N,dtype='int32'):
            flist.append(i)
    t1=time()
    print('append time for',N,'iters, per iter:',round(t0-t1,3)/N)
    t0=time()
    for i in np.arange(N,dtype='int32'):
            flist.pop()
    t1=time()
    print('pop time for',N,'iters, per iter:',round(t0-t1,3)/N)




'''
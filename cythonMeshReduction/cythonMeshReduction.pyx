import numpy as np 
import cython
from cython import int as cy_int
from cython import double as cy_double 
from libc.math cimport sin, cos, pow, abs, sqrt, acos, fmax, fmin
from numpy import int32,float64
from numpy cimport int32_t, float64_t
import trimesh as tr

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
                          vector3d vectC, vector3d attr0, vector3d attr1, vector3d attr2):
    cdef vector3d bary, out
    out = vector3d(0,0,0)
    bary = barycentric(vectP, vectA, vectB, vectC)
    out = out + attr0 * bary.x
    out = out + attr1 * bary.y
    out = out + attr2 * bary.z
    return out

cdef double mini(double v1, double v2):
    return fmin(v1,v2)


#############################################################################
#############################################################################
cdef class vector3d: 
    cdef public double x
    cdef public double y
    cdef public double z

    def __cinit__(self, double x=.0, 
                    double y=.0, double z=.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return 'Vector of coordinates \nx : {},\ny : {},\nz : {}'.format(round(self.x,4),
                                                                       round(self.y,4),
                                                                       round(self.z,4))

    def __add__(self, vector3d vect):
        return vector3d(self.x+vect.x, self.y+vect.y, self.z+vect.z)

    def __iadd__(self, vector3d vect):
        self.x+=vect.x
        self.y+=vect.y
        self.z+=vect.z
        return self

    def __sub__(self, vector3d vect):
        return vector3d(self.x-vect.x, self.y-vect.y, self.z-vect.z)

    def __isub__(self, vector3d vect):
        self.x-=vect.x
        self.y-=vect.y
        self.z-=vect.z
        return self

    def __mul__(self, other):
        if isinstance(other, self.__class__) :
            return vector3d(self.x*other.x, self.y*other.y, self.z*other.z)
        else :
            return vector3d(self.x*other, self.y*other, self.z*other)

    def __imul__(self, other):
        if isinstance(other, self.__class__) :
            self.x*=other.x
            self.y*=other.y
            self.z*=other.z
            return self
        else :
            self.x*=other
            self.y*=other
            self.z*=other
            return self

    def __div__(self, other):
        if isinstance(other, self.__class__) :
            return vector3d(self.x/other.x, self.y/other.y, self.z/other.z)
        else :
            return vector3d(self.x/other, self.y/other, self.z/other)

    def __idiv__(self, other):
        if isinstance(other, self.__class__) :
            self.x/=other.x
            self.y/=other.y
            self.z/=other.z
            return self
        else :
            self.x/=other
            self.y/=other
            self.z/=other
            return self

    def dot(self, vector3d vect):
        return self.x*vect.x+self.y*vect.y+self.z*vect.z

    def cross(self, vector3d vectA, vector3d vectB):
        self.x = vectA.y * vectB.z - vectA.z * vectB.y
        self.y = vectA.z * vectB.x - vectA.x * vectB.z
        self.z = vectA.x * vectB.y - vectA.y * vectB.x
        return self

    def length(self):
        return sqrt(self.x*self.x+self.y*self.y+self.z*self.z)

    def angle(self, vector3d vect):
        dotprod = vect.x*self.x + vect.y*self.y + vect.z*self.z
        lenmul = self.length()*vect.length()
        if lenmul==0:
            lenmul=0.000001
        val = dotprod/lenmul
        if val<-1: val=-1
        if val>1:val=1
        return acos(val)

    def angle2(self, vector3d vectA, vector3d vectB):
        dot = vectA.x*vectB.x + vectA.y*vectB.y + vectA.z*vectB.z
        lenmul = vectA.length()*vectB.length()
        if lenmul==0:
            lenmul=0.000001
        val = dot/lenmul
        if val<-1: val=-1
        if val>1:val=1
        return acos(val)

    def rot_x(self, double a):
        yy = cos(a)*self.y + sin(a)*self.z
        zz = cos(a)*self.z - sin(a)*self.y
        self.y = yy
        self.z = zz
        return self

    def rot_y(self, double a):
        xx = cos(-a)*self.x + sin(-a)*self.z
        zz = cos(-a)*self.z - sin(-a)*self.x
        self.x = xx
        self.z = zz
        return self

    def rot_z(self, double a):
        yy = cos(a)*self.y + sin(a)*self.x
        xx = cos(a)*self.x - sin(a)*self.y
        self.y = yy
        self.x = xx
        return self

    def clamp(self, double mini, double maxi):
        if self.x < mini :
            self.x = mini 
        if self.y < mini :
            self.y = mini
        if self.z < mini :
            self.z = mini
        if self.x > maxi :
            self.x = maxi 
        if self.y > maxi :
            self.y = maxi
        if self.z > maxi :
            self.z = maxi

    def invert(self):
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z
        return self

    def frac(self):
        return vector3d(float64(self.x - int(self.x)),
                        float64(self.y - int(self.y)),
                        float64(self.z - int(self.z)))

    def integer(self):
        return vector3d(float64(int(self.x)),
                        float64(int(self.y)),
                        float64(int(self.z)))        

    def normalize(self):
        square = sqrt(self.x*self.x+self.y*self.y+self.z*self.z)
        self.x /= square
        self.y /= square
        self.z /= square
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
        cdef double det 
        det = self.mat[a11]*self.mat[a22]*self.mat[a33]  \
             + self.mat[a13]*self.mat[a21]*self.mat[a32] \
             + self.mat[a12]*self.mat[a23]*self.mat[a31] \
             - self.mat[a13]*self.mat[a22]*self.mat[a31] \
             - self.mat[a11]*self.mat[a23]*self.mat[a32] \
             - self.mat[a12]*self.mat[a21]*self.mat[a33]
        return det 

    @staticmethod
    def makePlane(double a, double b, 
                  double c, double d):
        return SymetricMatrix(a*a, a*b, a*c, a*d, 
                        b*b, b*c, b*d, c*c, c*d, d*d)

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

    def __cinit__(self, vector3d v, int tstart=0, 
        int tcount=0, SymetricMatrix q=SymetricMatrix(), int border=0):
        self.v
        self.tstart
        self.tcount
        self.q
        self.border
        if self.v.x>.5:
            print('x',self.v.x)
        if self.v.y>.5:
            print('y',self.v.y)

    def __repr__(self):
        str0 = 'Vertex :\n'
        str1 = '  coords : {} | {} | {}\n' .format(self.v.x, self.v.y, self.v.z)
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
        x = float64(vertices_view[i,0])
        y = float64(vertices_view[i,1])
        z = float64(vertices_view[i,2])
        vertsList.append(Vertex(v=vector3d(x=x,y=y,z=z)))
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

cdef class Simplify:
    cdef list vertices 
    cdef list faces            
    cdef list refs

    def __cinit__(self):
        pass



















def test():
    vertices_view, nVerts, faces_view, nFaces = getFacesVerticesView()
    vertsList, facesList= makeVertsTrianglesRefs(vertices_view, nVerts, faces_view, nFaces)
    return vertsList, facesList


if __name__ == '__main__':
    x,y = test()
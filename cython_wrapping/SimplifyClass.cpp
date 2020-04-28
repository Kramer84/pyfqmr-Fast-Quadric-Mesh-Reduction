#include "Core.h"
#include "Simplify.h"

//defintion of a class to do the mesh reduction as a new function
// and interface it more easily with Cython.
// If we pass a pointer of the numpy array to the function,
// the array will be changed in place...
// But this is not possible as the shape changes!

namespace SimplifyPy 
{
    class SimplifyMeshReduction 
    {
            // pointers to Simplify global variables:
            std::vector<Simplify::Triangle> triangles = Simplify::triangles;
            std::vector<Simplify::Vertex>   vertices  = Simplify::vertices;
            // instantiate C Contiguous vertices and faces
            std::vector<double> verts_vect;
            std::vector<int>    faces_vect;
            // we instantiate them by passing them a C contigous array
        //public:
            void set_values_from_CArray(double* verts_C, int len_verts,
                                        int* faces_C, int len_faces);
            void clear_Mesh();
            void simplifyMesh(int target_count, double agressiveness);
            double* getVertsCContiguous();
            int*    getFacesCContiguous();
    };

    void SimplifyMeshReduction::clear_Mesh(){
        vertices.clear();
        triangles.clear();
    };

    void SimplifyMeshReduction::set_values_from_CArray(double* verts_C, int len_verts,
                                                       int* faces_C, int len_faces){
        // from stackoverflow "How to initialize std::vector from C-style array" 
        verts_vect.assign(verts_C, verts_C+len_verts);
        faces_vect.assign(faces_C, faces_C+len_faces);


        assert(verts_vect.size()%3==0);        // as we have X,Y,Z stacked in memory
        //int len_verts = verts_vect.size()/3;
        assert(faces_vect.size()%3==0);
        //int len_faces = faces_vect.size()/3;
        loopi(0, len_verts)
        {
            Simplify::Vertex v;
            v.p.x = verts_vect[3*i];   // C contiguous ?   
            v.p.y = verts_vect[3*i+1]; // C contiguous ? 
            v.p.z = verts_vect[3*i+2]; // C contiguous ?
            vertices.push_back(v);
        }
            loopi(0, len_faces)
        {
            Simplify::Triangle t;
            t.v[i]   = faces_vect[3*i];   // C contiguous ?    
            t.v[i+1] = faces_vect[3*i+1]; // C contiguous ? 
            t.v[i+2] = faces_vect[3*i+2]; // C contiguous ?
            triangles.push_back(t);
        }
        printf("Input: %d triangles %d vertices\n", triangles.size(), vertices.size());
    };

    void SimplifyMeshReduction::simplifyMesh(int target_count, double agressiveness=7){
        Simplify::simplify_mesh(target_count, agressiveness);
        printf("Output: %d triangles %d vertices\n", triangles.size(), vertices.size());
    };

    double* SimplifyMeshReduction::getVertsCContiguous(){
        int vertSize = verts_vect.size();
        double vertsArray[vertSize];
        std::copy(verts_vect.begin(), verts_vect.end(), vertsArray);
        return vertsArray;
    };

    int* SimplifyMeshReduction::getFacesCContiguous(){
        int faceSize = faces_vect.size();
        int facesArray[faceSize];
        std::copy(faces_vect.begin(), faces_vect.end(), facesArray);
        return facesArray;
    };
}
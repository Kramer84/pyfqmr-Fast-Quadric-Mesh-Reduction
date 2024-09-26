import typing as t

import numpy as np


class Simplify:
    def getMesh(self) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gets the mesh from the simplify object once the simplification is done

        Returns
        -------
        mesh
            a tuple containing: verts (array of vertices of shape
            (n_vertices,3)), faces (array of faces of shape (n_faces,3)),
            normals (array of normals of shape (n_faces,3))
        """

    def setMesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        face_colors: t.Union[np.ndarray, None] = None,
    ) -> None:
        """Method to set the mesh of the simplifier object.

        Arguments
        ---------
        vertices
            array of vertices of shape (n_vertices,3)
        faces
            array of faces of shape (n_faces,3)
        face_colors
            array of face_colors of shape (n_faces,3)
            (this is not yet implemented)
        """

    def simplify_mesh(
        self,
        target_count: int = 100,
        update_rate: int = 5,
        aggressiveness: float = 7.0,
        max_iterations: int = 100,
        verbose: bool = True,
        lossless: bool = False,
        threshold_lossless: float = 1e-3,
        alpha: float = 1e-9,
        K: int = 3,
        preserve_border: bool = True,
    ) -> None:
        """Simplify mesh

        Parameters
        ----------
        target_count
            Target number of triangles, not used if lossless is True
        update_rate
            Number of iterations between each update.
            If lossless flag is set to True, rate is 1
        aggressiveness
            Parameter controlling the growth rate of the threshold at each
            iteration when lossless is False.
        max_iterations
            Maximal number of iterations
        verbose
            control verbosity
        lossless
            Use the lossless simplification method
        threshold_lossless
            Maximal error after which a vertex is not deleted, only for
            lossless method.
        alpha
            Parameter for controlling the threshold growth
        K
            Parameter for controlling the thresold growth
        preserve_border
            Flag for preserving vertices on open border

        Note
        ----
        threshold = alpha*pow( iteration + K, agressiveness)
        """

from logging import root as root_logger, DEBUG
from pathlib import Path

import pytest
import pyfqmr

import numpy as np

root_logger.setLevel(DEBUG)

# Get the /example folder at the root of this repo
EXAMPLES_DIR = Path(__file__, "..", "..", "example").resolve()

def test_example():
    import trimesh as tr
    bunny = tr.load_mesh(EXAMPLES_DIR / 'Stanford_Bunny_sample.stl')
    simp = pyfqmr.Simplify()
    simp.setMesh(bunny.vertices, bunny.faces)
    simp.simplify_mesh(len(bunny.faces) // 2)
    vertices, faces, normals = simp.getMesh()

    assert len(faces) / len(bunny.faces) == pytest.approx(.5, rel=.05)
    simplified = tr.Trimesh(vertices, faces, normals)
    assert simplified.area == pytest.approx(simplified.area, rel=.05)

def test_empty():
    verts = np.zeros((0,3), dtype=np.float32)
    faces = np.zeros((0,3), dtype=np.int32)

    simp = pyfqmr.Simplify()
    simp.setMesh(verts, faces)
    simp.simplify_mesh()
    vertices, faces, normals = simp.getMesh()

    assert len(vertices) == 0
    assert len(faces) == 0
    assert len(normals) == 0

def test_example_lossless():
    import trimesh as tr
    bunny = tr.load_mesh(EXAMPLES_DIR / 'Stanford_Bunny_sample.stl')
    simp = pyfqmr.Simplify()
    simp.setMesh(bunny.vertices, bunny.faces)
    simp.simplify_mesh_lossless()
    vertices, faces, normals = simp.getMesh()

    assert len(faces) / len(bunny.faces) == pytest.approx(.5334, rel=.05)
    simplified = tr.Trimesh(vertices, faces, normals)
    assert simplified.area == pytest.approx(simplified.area, rel=.05)

def test_empty_lossless():
    verts = np.zeros((0,3), dtype=np.float32)
    faces = np.zeros((0,3), dtype=np.int32)

    simp = pyfqmr.Simplify()
    simp.setMesh(verts, faces)
    simp.simplify_mesh_lossless()
    vertices, faces, normals = simp.getMesh()

    assert len(vertices) == 0
    assert len(faces) == 0
    assert len(normals) == 0

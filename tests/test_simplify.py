from pathlib import Path

import pytest
import pyfqmr

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

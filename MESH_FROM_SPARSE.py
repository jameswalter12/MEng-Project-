import subprocess
from pathlib import Path
import argparse
import shutil

def run(cmd, cwd=None):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

REQUIRED_SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin")
OPTIONAL_SPARSE_FILES = ("rigs.bin", "frames.bin")

def resolve_project(project_arg):
    if project_arg:
        return Path(project_arg).expanduser().resolve()

    workspace = Path(__file__).resolve().parent
    candidates = (
        workspace / "PIPELINE",
        workspace / "PIPELINE_2",
        workspace / "PIPELINE_3" / "TEST MINNIE",
        workspace / "PIPELINE_3",
    )
    for candidate in candidates:
        if has_minimum_inputs(candidate):
            return candidate
    raise FileNotFoundError(
        "No valid project found. Pass --project to a folder that has images and sparse bins."
    )

def resolve_images_dir(project):
    options = (
        project / "images",
        project / "INPUT",
        project / "TEST MINNIE" / "INPUT",
        project / "INPUT IMAGES",
    )
    for option in options:
        if option.is_dir():
            return option
    raise FileNotFoundError(
        f"Missing images folder in {project}. Tried: images, INPUT, TEST MINNIE/INPUT, INPUT IMAGES."
    )

def ensure_sparse_dir(project):
    sparse_dir = project / "sparse"
    if sparse_dir.is_dir() and all((sparse_dir / name).exists() for name in REQUIRED_SPARSE_FILES):
        return sparse_dir

    # Some runs keep bin files directly at project root; mirror into project/sparse for InterfaceCOLMAP.
    if any((project / name).exists() for name in REQUIRED_SPARSE_FILES):
        sparse_dir.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_SPARSE_FILES + OPTIONAL_SPARSE_FILES:
            src = project / name
            dst = sparse_dir / name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    if sparse_dir.is_dir() and all((sparse_dir / name).exists() for name in REQUIRED_SPARSE_FILES):
        return sparse_dir

    raise FileNotFoundError(f"Missing sparse files in {project}/sparse.")

def has_minimum_inputs(project):
    if not project.exists() or not project.is_dir():
        return False
    try:
        resolve_images_dir(project)
        ensure_sparse_dir(project)
        return True
    except FileNotFoundError:
        return False

def main():
    parser = argparse.ArgumentParser(description="Run OpenMVS mesh generation from COLMAP sparse model.")
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project directory containing images and sparse model files.",
    )
    args = parser.parse_args()

    project = resolve_project(args.project)

    #OpenMVS binaries (built) - Need to figure out exactly what was done here
    openmvs_bin = Path.home()/"openMVS"/"build_m1"/"bin"

    InterfaceCOLMAP = str(openmvs_bin / "InterfaceCOLMAP")
    DensifyPointCloud = str(openmvs_bin / "DensifyPointCloud")
    ReconstructMesh = str(openmvs_bin / "ReconstructMesh")
    RefineMesh = str(openmvs_bin / "RefineMesh")
    TextureMesh = str(openmvs_bin / "TextureMesh")
    ExportScene = str(openmvs_bin / "ExportScene")

    images = resolve_images_dir(project)
    sparse = ensure_sparse_dir(project)

    # Sanity checks
    assert images.is_dir(), f"Missing images folder: {images}"
    assert sparse.is_dir(), f"Missing sparse folder: {sparse}"
    assert (sparse / "cameras.bin").exists(), "Missing sparse/cameras.bin"
    assert (sparse / "images.bin").exists(), "Missing sparse/images.bin"
    assert (sparse / "points3D.bin").exists(), "Missing sparse/points3D.bin"

    # Outputs
    scene = project / "scene.mvs"
    scene_dense = project / "scene_dense.mvs"
    scene_dense_mesh = project / "scene_dense_mesh.ply"
    scene_dense_mesh_refine = project / "scene_dense_mesh_refine.ply"
    scene_dense_mesh_refine_texture = project / "scene_dense_mesh_refine_texture.obj"

    # 1) COLMAP -> OpenMVS
    # IMPORTANT: -i is the *project root* because InterfaceCOLMAP appends /sparse internally
    run([InterfaceCOLMAP, "-i", str(project), "-o", str(scene), "--image-folder", str(images)], cwd=project)

    # 2) Dense point cloud
    run([DensifyPointCloud, str(scene)], cwd=project)

    # 3) Mesh
    run([ReconstructMesh, str(scene_dense), "-o", str(scene_dense_mesh)], cwd=project)

    # 4) Refine
    run([RefineMesh, "-i", str(scene_dense), "-m", str(scene_dense_mesh), "-o", str(scene_dense_mesh_refine)], cwd=project)

    # 5) Texture
    run([
        TextureMesh,
        "-i", str(scene_dense),
        "-m", str(scene_dense_mesh_refine),
        "-o", str(scene_dense_mesh_refine_texture),
        "--export-type", "obj"
    ], cwd=project)

    print("\nDONE ✅")
    print("Look for OBJ output in:", project)

if __name__ == "__main__":
    main()

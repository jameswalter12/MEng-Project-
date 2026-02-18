"""
Create a COLMAP sparse model compatible with PIPELINE/MESH_FROM_SPARSE.py.

Example:
python3 PIPELINE/COLMAP_SPARSE_CREATION.py \
  --project "/Users/jameswalter/Desktop/Photogammetry/PIPELINE_2/PROJECT_2_MUG" \
  --images-subdir "INPUT IMAGES" \
  --preset iphone_balanced \
  --force
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2

DEFAULT_CAMERA_MODEL = "PINHOLE"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def run(cmd, cwd=None, dry_run=False):
    print("\n>>>", " ".join(str(part) for part in cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def read_command_help(colmap_binary, command):
    proc = subprocess.run(
        [str(colmap_binary), command, "-h"],
        check=True,
        capture_output=True,
        text=True,
    )
    # Newer COLMAP builds print help to stderr.
    return f"{proc.stdout}\n{proc.stderr}"


def choose_option_key(help_text, preferred_key, fallback_key):
    if f"--{preferred_key}" in help_text:
        return preferred_key
    return fallback_key


def normalize_option_key(key):
    aliases = {
        # GUI-like short forms.
        "sift.max_image_size": "SiftExtraction.max_image_size",
        "sift.max_num_features": "SiftExtraction.max_num_features",
        "sift.first_octave": "SiftExtraction.first_octave",
        "sift.num_octaves": "SiftExtraction.num_octaves",
        "sift.octave_resolution": "SiftExtraction.octave_resolution",
        "sift.peak_threshold": "SiftExtraction.peak_threshold",
        "sift.edge_threshold": "SiftExtraction.edge_threshold",
        # Old matching keys -> modern matching keys.
        "SiftMatching.guided_matching": "FeatureMatching.guided_matching",
        "SiftMatching.max_num_matches": "FeatureMatching.max_num_matches",
    }
    return aliases.get(key, key)


def parse_key_value_options(items):
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid option '{item}'. Use KEY=VALUE format.")
        key, value = item.split("=", 1)
        key = normalize_option_key(key.strip())
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid option '{item}'. Empty KEY.")
        parsed.append((key, value))
    return parsed


def dedupe_key_value_options(key_values):
    # COLMAP rejects duplicate options; keep the last specified value.
    deduped = {}
    for key, value in key_values:
        deduped[key] = value
    return list(deduped.items())


def ensure_default_mapper_option(mapper_opts, key, value):
    keys = {k for k, _ in mapper_opts}
    if key in keys:
        return mapper_opts
    return mapper_opts + [(key, value)]


def append_colmap_options(cmd, key_values):
    for key, value in key_values:
        cmd.extend([f"--{key}", value])
    return cmd


def get_preset_options(preset_name):
    if preset_name == "iphone_balanced":
        return {
            "camera_model": "PINHOLE",
            "single_camera": True,
            "matcher": "sequential",
            "feature_opt": [
                "SiftExtraction.max_image_size=2800",
                "SiftExtraction.max_num_features=12000",
                "SiftExtraction.num_octaves=4",
                "SiftExtraction.octave_resolution=3",
                "SiftExtraction.peak_threshold=0.0067",
                "SiftExtraction.edge_threshold=10",
            ],
            "match_opt": [
                "FeatureMatching.guided_matching=1",
                "FeatureMatching.max_num_matches=32768",
            ],
            "mapper_opt": [
                "Mapper.filter_max_reproj_error=2",
                "Mapper.ba_refine_focal_length=1",
                "Mapper.ba_refine_extra_params=0",
                "Mapper.ba_refine_principal_point=0",
                "Mapper.ba_global_max_num_iterations=50",
            ],
        }

    if preset_name == "iphone_fast":
        return {
            "camera_model": "PINHOLE",
            "single_camera": True,
            "matcher": "sequential",
            "feature_opt": [
                "SiftExtraction.max_image_size=2200",
                "SiftExtraction.max_num_features=8000",
                "SiftExtraction.num_octaves=4",
                "SiftExtraction.octave_resolution=3",
                "SiftExtraction.peak_threshold=0.008",
                "SiftExtraction.edge_threshold=10",
                "FeatureExtraction.num_threads=4",
            ],
            "match_opt": [
                "FeatureMatching.guided_matching=1",
                "FeatureMatching.max_num_matches=20000",
                "FeatureMatching.num_threads=4",
            ],
            "mapper_opt": [
                "Mapper.filter_max_reproj_error=2.5",
                "Mapper.ba_refine_focal_length=1",
                "Mapper.ba_refine_extra_params=0",
                "Mapper.ba_refine_principal_point=0",
                "Mapper.ba_global_max_num_iterations=30",
                "Mapper.num_threads=4",
            ],
        }

    return None


def find_colmap_binary(user_provided=None):
    if user_provided:
        colmap = Path(user_provided).expanduser().resolve()
        if colmap.exists():
            return colmap
        raise FileNotFoundError(f"COLMAP binary not found: {colmap}")

    discovered = shutil.which("colmap")
    if discovered:
        return Path(discovered).resolve()

    fallback_paths = (
        Path("/opt/homebrew/bin/colmap"),
        Path("/usr/local/bin/colmap"),
    )
    for candidate in fallback_paths:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "COLMAP binary not found. Install COLMAP or pass --colmap-binary /path/to/colmap."
    )


def validate_inputs(project_dir, images_subdir):
    project_dir = project_dir.expanduser().resolve()
    images_dir = (project_dir / images_subdir).resolve()
    if not project_dir.is_dir():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    return project_dir, images_dir


def iter_valid_images(images_dir):
    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        yield path


def write_image_list(images_dir, project_dir):
    image_names = [path.name for path in iter_valid_images(images_dir)]
    if not image_names:
        raise RuntimeError(f"No valid images found in {images_dir}")
    image_list_path = project_dir / ".colmap_image_list.txt"
    image_list_path.write_text("\n".join(image_names) + "\n", encoding="utf-8")
    return image_list_path, len(image_names)


def images_have_mixed_sizes(images_dir):
    sizes = set()
    for path in iter_valid_images(images_dir):
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        sizes.add((w, h))
        if len(sizes) > 1:
            return True
    return False


def count_model_stats(colmap_binary, model_dir):
    with tempfile.TemporaryDirectory(prefix="colmap_model_stats_") as tmp:
        tmp_path = Path(tmp)
        subprocess.run(
            [
                str(colmap_binary),
                "model_converter",
                "--input_path",
                str(model_dir),
                "--output_path",
                str(tmp_path),
                "--output_type",
                "TXT",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        images_txt = tmp_path / "images.txt"
        points_txt = tmp_path / "points3D.txt"

        image_count = 0
        for line in images_txt.read_text(encoding="utf-8").splitlines():
            s = line.strip().lower()
            if not s or s.startswith("#"):
                continue
            if ".jpg" in s or ".jpeg" in s or ".png" in s or ".bmp" in s or ".tif" in s:
                image_count += 1

        points_count = 0
        for line in points_txt.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                points_count += 1

        return image_count, points_count


def flatten_sparse_model(mapper_sparse_root, target_sparse_dir, colmap_binary):
    model_dirs = sorted(
        [p for p in mapper_sparse_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    if not model_dirs:
        raise FileNotFoundError(f"No sparse model folders found under {mapper_sparse_root}")

    ranked_models = []
    for model_dir in model_dirs:
        try:
            image_count, points_count = count_model_stats(colmap_binary, model_dir)
        except Exception:
            image_count, points_count = (0, 0)
        ranked_models.append((image_count, points_count, model_dir))

    ranked_models.sort(key=lambda item: (item[0], item[1], int(item[2].name)), reverse=True)
    source_model = ranked_models[0][2]
    print("\nSparse model candidates:")
    for image_count, points_count, model_dir in ranked_models:
        print(f"  model {model_dir.name}: images={image_count}, points={points_count}")
    print(f"Selected sparse model: {source_model.name}")

    required = ("cameras.bin", "images.bin", "points3D.bin")
    missing = [name for name in required if not (source_model / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Model folder {source_model} missing files: {', '.join(missing)}"
        )

    target_sparse_dir.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin"):
        src = source_model / name
        dst = target_sparse_dir / name
        if src.exists():
            shutil.copy2(src, dst)
    return source_model


def delete_if_exists(path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Build COLMAP sparse reconstruction for OpenMVS MESH_FROM_SPARSE.py."
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("/Users/jameswalter/Desktop/Photogammetry/PIPELINE"),
        help="Project folder used by MESH_FROM_SPARSE.py.",
    )
    parser.add_argument(
        "--images-subdir",
        type=str,
        default="INPUT",
        help="Images subdirectory inside --project.",
    )
    parser.add_argument(
        "--colmap-binary",
        type=str,
        default=None,
        help="Path to colmap binary. If omitted, uses PATH/fallback locations.",
    )
    parser.add_argument(
        "--camera-model",
        type=str,
        default=None,
        help="COLMAP camera model (e.g. OPENCV, SIMPLE_RADIAL, PINHOLE).",
    )
    parser.add_argument(
        "--single-camera",
        action="store_true",
        help="Use one shared camera intrinsics for all images.",
    )
    parser.add_argument(
        "--no-single-camera",
        action="store_true",
        help="Force per-image intrinsics (overrides presets).",
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="exhaustive",
        choices=("exhaustive", "sequential"),
        help="Matcher type.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing database/sparse outputs before rebuilding.",
    )
    parser.add_argument(
        "--project-ini",
        type=Path,
        default=None,
        help="Path to COLMAP .ini saved from GUI (applied to all stages).",
    )
    parser.add_argument(
        "--feature-opt",
        action="append",
        default=[],
        help="Extra feature_extractor option in KEY=VALUE form (repeatable).",
    )
    parser.add_argument(
        "--match-opt",
        action="append",
        default=[],
        help="Extra matcher option in KEY=VALUE form (repeatable).",
    )
    parser.add_argument(
        "--mapper-opt",
        action="append",
        default=[],
        help="Extra mapper option in KEY=VALUE form (repeatable).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="none",
        choices=("none", "iphone_balanced", "iphone_fast"),
        help="Apply predefined quality/speed settings.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Apply a fixed thread count to extraction, matching, and mapping.",
    )
    parser.add_argument(
        "--use-gpu",
        type=int,
        choices=(0, 1),
        default=0,
        help="COLMAP feature/matching GPU flag (0=off, 1=on).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not execute COLMAP.",
    )
    args = parser.parse_args()

    if args.preset != "none":
        preset = get_preset_options(args.preset)
        if preset is None:
            raise ValueError(f"Unknown preset: {args.preset}")
        if args.camera_model is None:
            args.camera_model = preset["camera_model"]
        args.single_camera = preset["single_camera"] or args.single_camera
        args.matcher = preset["matcher"]
        args.feature_opt = preset["feature_opt"] + args.feature_opt
        args.match_opt = preset["match_opt"] + args.match_opt
        args.mapper_opt = preset["mapper_opt"] + args.mapper_opt

    if args.camera_model is None:
        args.camera_model = DEFAULT_CAMERA_MODEL
    camera_model_upper = args.camera_model.upper()
    if camera_model_upper not in {"PINHOLE", "SIMPLE_PINHOLE"}:
        print(
            "\nWARNING: OpenMVS InterfaceCOLMAP expects PINHOLE/SIMPLE_PINHOLE cameras. "
            f"Current camera model is '{args.camera_model}'."
        )

    if args.threads is not None:
        args.feature_opt = args.feature_opt + [f"FeatureExtraction.num_threads={args.threads}"]
        args.match_opt = args.match_opt + [f"FeatureMatching.num_threads={args.threads}"]
        args.mapper_opt = args.mapper_opt + [f"Mapper.num_threads={args.threads}"]

    project_dir, images_dir = validate_inputs(args.project, args.images_subdir)
    if args.no_single_camera:
        args.single_camera = False
    elif args.single_camera and images_have_mixed_sizes(images_dir):
        print(
            "\nWARNING: Mixed image dimensions detected; disabling single-camera automatically."
        )
        args.single_camera = False

    image_list_path, image_count = write_image_list(images_dir, project_dir)
    print(f"\nUsing {image_count} images from list: {image_list_path}")

    colmap = find_colmap_binary(args.colmap_binary)

    feature_help = read_command_help(colmap, "feature_extractor")
    matcher_help = read_command_help(colmap, "exhaustive_matcher")
    feature_gpu_key = choose_option_key(
        feature_help, "FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"
    )
    matcher_gpu_key = choose_option_key(
        matcher_help, "FeatureMatching.use_gpu", "SiftMatching.use_gpu"
    )

    database_path = project_dir / "database.db"
    database_wal = project_dir / "database.db-wal"
    database_shm = project_dir / "database.db-shm"
    sparse_root = project_dir / "sparse"
    sparse_ply = project_dir / "sparse.ply"
    mapper_out = project_dir / "sparse_raw"

    if args.force:
        delete_if_exists(database_path)
        delete_if_exists(database_wal)
        delete_if_exists(database_shm)
        delete_if_exists(mapper_out)
        delete_if_exists(sparse_root)
        delete_if_exists(sparse_ply)

    mapper_out.mkdir(parents=True, exist_ok=True)
    project_ini = None
    if args.project_ini:
        project_ini = args.project_ini.expanduser().resolve()
        if not project_ini.exists():
            raise FileNotFoundError(f"COLMAP ini file not found: {project_ini}")

    feature_opts = parse_key_value_options(args.feature_opt)
    match_opts = parse_key_value_options(args.match_opt)
    mapper_opts = parse_key_value_options(args.mapper_opt)
    feature_opts = dedupe_key_value_options(feature_opts)
    match_opts = dedupe_key_value_options(match_opts)
    mapper_opts = dedupe_key_value_options(mapper_opts)
    # Keep one deterministic reconstruction model per run.
    mapper_opts = ensure_default_mapper_option(mapper_opts, "Mapper.multiple_models", "0")
    mapper_opts = ensure_default_mapper_option(mapper_opts, "Mapper.max_num_models", "1")

    feature_cmd = [
        str(colmap),
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_dir),
        "--image_list_path",
        str(image_list_path),
        "--ImageReader.camera_model",
        args.camera_model,
        f"--{feature_gpu_key}",
        str(args.use_gpu),
    ]
    if project_ini:
        feature_cmd += ["--project_path", str(project_ini)]
    if args.single_camera:
        feature_cmd += ["--ImageReader.single_camera", "1"]
    append_colmap_options(feature_cmd, feature_opts)
    run(feature_cmd, cwd=project_dir, dry_run=args.dry_run)

    if args.matcher == "exhaustive":
        match_cmd = [
            str(colmap),
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            f"--{matcher_gpu_key}",
            str(args.use_gpu),
        ]
    else:
        match_cmd = [
            str(colmap),
            "sequential_matcher",
            "--database_path",
            str(database_path),
            f"--{matcher_gpu_key}",
            str(args.use_gpu),
        ]
    if project_ini:
        match_cmd += ["--project_path", str(project_ini)]
    append_colmap_options(match_cmd, match_opts)
    run(match_cmd, cwd=project_dir, dry_run=args.dry_run)

    mapper_cmd = [
        str(colmap),
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_dir),
        "--output_path",
        str(mapper_out),
        "--Mapper.image_list_path",
        str(image_list_path),
    ]
    if project_ini:
        mapper_cmd += ["--project_path", str(project_ini)]
    append_colmap_options(mapper_cmd, mapper_opts)
    run(mapper_cmd, cwd=project_dir, dry_run=args.dry_run)

    if args.dry_run:
        print("\nDRY RUN COMPLETE")
        print("No files were modified by COLMAP commands.")
        return

    source_model = flatten_sparse_model(mapper_out, sparse_root, colmap)
    run(
        [
            str(colmap),
            "model_converter",
            "--input_path",
            str(sparse_root),
            "--output_path",
            str(sparse_ply),
            "--output_type",
            "PLY",
        ],
        cwd=project_dir,
        dry_run=args.dry_run,
    )

    print("\nDONE")
    print(f"Project: {project_dir}")
    print(f"Images: {images_dir}")
    print(f"Model selected: {source_model}")
    print(f"Sparse ready for OpenMVS: {sparse_root}")
    print(f"Sparse PLY exported: {sparse_ply}")
    print("\nNext:")
    print(f"python3 {project_dir.parents[1] / 'PIPELINE' / 'MESH_FROM_SPARSE.py'}")

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Working terminal commands
# ---------------------------------------------------------------------------
# 1) Build sparse (OpenMVS-safe PINHOLE) for PIPELINE:
# python3 'PIPELINE/COLMAP_SPARSE_CREATION.py' \
#   --project '/Users/jameswalter/Desktop/Photogammetry/PIPELINE' \
#   --images-subdir 'IMAGES_PROCESSED/masked' \
#   --preset iphone_fast \
#   --camera-model PINHOLE \
#   --single-camera \
#   --force
#
# 2) Verify camera model:
# mkdir -p /tmp/pipeline_sparse_txt && \
# colmap model_converter \
#   --input_path '/Users/jameswalter/Desktop/Photogammetry/PIPELINE/sparse' \
#   --output_path '/tmp/pipeline_sparse_txt' \
#   --output_type TXT && \
# sed -n '1,20p' /tmp/pipeline_sparse_txt/cameras.txt
#
# 3) Then run mesh:
# python3 'PIPELINE/MESH_FROM_SPARSE.py' \
#   --project '/Users/jameswalter/Desktop/Photogammetry/PIPELINE' \
#   --images-subdir 'IMAGES_PROCESSED' \
#   --profile safe

'''
python3 'PIPELINE/COLMAP_SPARSE_CREATION.py' \
  --project '/Users/jameswalter/Desktop/Photogammetry/PIPELINE' \
  --images-subdir 'IMAGES_PROCESSED/masked' \
  --preset iphone_fast \
  --camera-model PINHOLE \
  --force
'''

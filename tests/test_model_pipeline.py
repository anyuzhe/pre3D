import struct
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

import ai_photogrammetry.engineering.model_pipeline as model_pipeline
from ai_photogrammetry.engineering.model_pipeline import (
    _write_oriented_point_ply,
    condition_point_cloud,
    export_textured_mesh,
    export_textured_mesh_blocks,
    find_osgconv,
    partition_mesh_for_texturing,
    ply_counts,
    texture_mesh_blocks,
)


def _write_textured_ply(path: Path) -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = [
        ([0, 1, 2], [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
        ([0, 2, 3], [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
    ]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment TextureFile texture.png\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "property list uchar float texcoord\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)
        for indices, uv in faces:
            stream.write(struct.pack("<B3iB6f", 3, *indices, 6, *uv))


def test_point_conditioning_repairs_normals_and_writes_binary_ply(tmp_path: Path):
    points = np.asarray(
        [[float(index), float(index % 3), 0.0] for index in range(64)],
        dtype=np.float32,
    )
    normals = np.tile([0.0, 0.0, 2.0], (len(points), 1)).astype(np.float32)
    normals[3] = 0.0
    colors = np.tile([120, 130, 140], (len(points), 1)).astype(np.uint8)
    source = tmp_path / "fused.ply"
    output = tmp_path / "conditioned.ply"
    _write_oriented_point_ply(source, points, normals, colors)

    report = condition_point_cloud(
        source,
        output,
        point_limit=40,
        outlier_sample_size=8,
        outlier_std_ratio=4.0,
    )

    assert output.is_file()
    assert 3 <= ply_counts(output)[0] <= 40
    assert report["input_point_count"] == 64
    assert report["normal_repair"] == "normalize_and_neighbor_repair"


def test_textured_mesh_exports_obj_fbx_and_gltf(tmp_path: Path):
    mesh = tmp_path / "mesh.ply"
    texture = tmp_path / "texture.png"
    output = tmp_path / "exports"
    _write_textured_ply(mesh)
    Image.new("RGB", (4, 4), (80, 140, 200)).save(texture)

    result = export_textured_mesh(
        mesh,
        texture,
        output,
        formats=["obj", "fbx", "gltf"],
    )

    for key in ("obj", "mtl", "fbx", "gltf", "glb", "texture_atlas"):
        assert Path(result[key]).is_file()
    assert "FBXVersion: 7400" in Path(result["fbx"]).read_text(encoding="utf-8")
    for key in ("obj", "gltf", "glb"):
        scene = trimesh.load(result[key], force="scene", process=False)
        assert sum(len(geometry.faces) for geometry in scene.geometry.values()) == 2


def test_mesh_partition_preserves_every_face_exactly_once(tmp_path: Path):
    mesh = tmp_path / "mesh.ply"
    _write_textured_ply(mesh)

    result = partition_mesh_for_texturing(
        mesh,
        tmp_path / "partitions",
        target_faces=1,
    )

    assert result["block_count"] == 2
    assert sum(block["face_count"] for block in result["blocks"]) == 2
    assert all(Path(block["mesh"]).is_file() for block in result["blocks"])


def test_multi_atlas_export_writes_one_obj_with_multiple_materials(tmp_path: Path):
    blocks = []
    for index, color in enumerate(((255, 0, 0), (0, 0, 255)), 1):
        mesh = tmp_path / f"mesh_{index}.ply"
        texture = tmp_path / f"texture_{index}.png"
        _write_textured_ply(mesh)
        Image.new("RGB", (4, 4), color).save(texture)
        blocks.append({"mesh": str(mesh), "texture": str(texture)})

    result = export_textured_mesh_blocks(
        blocks,
        tmp_path / "exports",
        formats=["obj", "gltf"],
    )

    obj_text = Path(result["obj"]).read_text(encoding="utf-8")
    mtl_text = Path(result["mtl"]).read_text(encoding="utf-8")
    assert obj_text.count("usemtl TextureBlock_") == 2
    assert mtl_text.count("map_Kd model_texture_") == 2
    scene = trimesh.load(result["glb"], force="scene", process=False)
    assert sum(len(geometry.faces) for geometry in scene.geometry.values()) == 4


def test_texture_block_retries_only_that_atlas_at_lower_scale(
    tmp_path: Path,
    monkeypatch,
):
    input_mesh = tmp_path / "input.ply"
    _write_textured_ply(input_mesh)
    attempts = []

    def fake_run(_executable, arguments, _log_path):
        scale = float(
            arguments[
                arguments.index("--MeshTextureMapping.texture_scale_factor") + 1
            ]
        )
        attempts.append(scale)
        if scale > 0.5:
            raise RuntimeError(
                "COLMAP 子命令 mesh_texturer 失败（退出码 3221226505）。"
                " Atlas size: 65536 x 12000 Baking texture"
            )
        output = Path(arguments[arguments.index("--output_path") + 1])
        output.mkdir(parents=True, exist_ok=True)
        _write_textured_ply(output / "mesh.ply")
        Image.new("RGB", (1024, 1024), "white").save(output / "texture.png")

    monkeypatch.setattr(model_pipeline, "_run", fake_run)
    result = texture_mesh_blocks(
        executable="colmap",
        dense_workspace=tmp_path,
        partition={
            "blocks": [
                {"id": "block_0001", "mesh": str(input_mesh), "face_count": 2}
            ]
        },
        output_root=tmp_path / "textured",
        texture_scale_factor=1.0,
        atlas_max_dimension=32768,
        atlas_max_pixels=180_000_000,
    )

    assert attempts[:3] == [1.0, 0.75, 0.5]
    assert result["blocks"][0]["texture_scale_factor"] == 0.5
    assert result["blocks"][0]["atlas_width"] == 1024


def test_osgb_requires_real_openscenegraph_converter(tmp_path: Path, monkeypatch):
    mesh = tmp_path / "mesh.ply"
    texture = tmp_path / "texture.png"
    _write_textured_ply(mesh)
    Image.new("RGB", (2, 2), "white").save(texture)
    monkeypatch.setattr(model_pipeline, "find_osgconv", lambda _path=None: None)

    with pytest.raises(RuntimeError, match="osgconv"):
        export_textured_mesh(
            mesh,
            texture,
            tmp_path / "exports",
            formats=["osgb"],
            osgconv_path=str(tmp_path / "missing.exe"),
        )


def test_real_osgb_export_when_converter_is_installed(tmp_path: Path):
    converter = find_osgconv()
    if not converter:
        pytest.skip("OpenSceneGraph runtime is optional")
    mesh = tmp_path / "mesh.ply"
    texture = tmp_path / "texture.png"
    _write_textured_ply(mesh)
    Image.new("RGB", (2, 2), "white").save(texture)

    result = export_textured_mesh(
        mesh,
        texture,
        tmp_path / "exports",
        formats=["osgb"],
        osgconv_path=converter,
    )

    osgb = Path(result["osgb"])
    assert osgb.is_file()
    assert osgb.read_bytes()[:4] != b"mtll"

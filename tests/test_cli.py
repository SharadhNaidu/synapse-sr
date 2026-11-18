"""Command-line contract: folder mode, JSON output, flags and error exit codes."""

import json
import pathlib

import numpy as np
import pytest

from synapse_sr.cli import main as cli

from conftest import scene, write_tif

rasterio = pytest.importorskip("rasterio")


@pytest.fixture(scope="module")
def weights(model, tmp_path_factory):
    return model.save_pretrained(str(tmp_path_factory.mktemp("w") / "m.safetensors"))


@pytest.fixture
def folder(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    write_tif(src / "a.tif", scene(16, 16, seed=1))
    write_tif(src / "b.TIFF", scene(16, 16, seed=2))
    write_tif(src / "a_scl.tif", np.full((1, 8, 8), 4, np.uint8), names=None)
    return src


def base(weights):
    return ["--weights", weights, "--device", "cpu", "--tile", "16"]


def test_folder_json_is_clean_stdout(folder, weights, tmp_path, capsys):
    assert cli([str(folder), str(tmp_path / "out")] + base(weights) + ["--json"]) == 0
    cap = capsys.readouterr()
    rep = json.loads(cap.out)
    assert sorted(pathlib.Path(r["input"]).name for r in rep) == ["a.tif", "b.TIFF"]
    assert "1/2" in cap.err and "1/2" not in cap.out


def test_folder_honours_quiet_no_confidence_and_preview(folder, weights, tmp_path, capsys):
    out, png = tmp_path / "out", tmp_path / "png"
    assert cli([str(folder), str(out), "--quiet", "--no-confidence", "--preview", str(png)] + base(weights)) == 0
    assert "synapse-sr:" not in capsys.readouterr().err
    with rasterio.open(out / "a.tif") as d:
        assert d.count == 4
    assert sorted(p.name for p in png.iterdir()) == ["a.png", "b.png"]


def test_folder_skips_existing_outputs(folder, weights, tmp_path, capsys):
    out = tmp_path / "out"
    assert cli([str(folder), str(out), "--quiet"] + base(weights)) == 0
    assert cli([str(folder), str(out), "--json"] + base(weights)) == 0
    assert all(r.get("skipped") == "exists" for r in json.loads(capsys.readouterr().out))


def test_window_and_bounds(tmp_path, weights):
    src = tmp_path / "s.tif"
    t = write_tif(src, scene(32, 32))
    assert cli([str(src), str(tmp_path / "w.tif"), "--window", "4,8,16,16", "--quiet"] + base(weights)) == 0
    bounds = f"{t.c + 80},{t.f - 200},{t.c + 240},{t.f - 40}"
    assert cli([str(src), str(tmp_path / "b.tif"), "--bounds", bounds, "--quiet"] + base(weights)) == 0
    with rasterio.open(tmp_path / "w.tif") as w, rasterio.open(tmp_path / "b.tif") as b:
        assert (w.height, w.width) == (80, 80) and w.transform == b.transform
        assert (w.transform.c, w.transform.f) == (t.c + 80, t.f - 40)
        assert np.allclose(w.read(1), b.read(1), equal_nan=True)


@pytest.mark.parametrize("args, message", [
    (["--window", "1,2"], "--window ROW,COL,H,W"),
    (["--bounds", "a,b,c,d"], "--bounds L,B,R,T"),
    (["--window", "500,500,4,4"], "outside"),
])
def test_bad_area_arguments_exit_1(tmp_path, weights, capsys, args, message):
    src = tmp_path / "s.tif"
    write_tif(src, scene(16, 16))
    assert cli([str(src), str(tmp_path / "o.tif"), "--quiet"] + base(weights) + args) == 1
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("args, message", [
    (["--fetch", "12.9"], "LAT,LON"),
    (["--fetch", "190,77.5"], "latitude"),
    (["--fetch", "12.9,77.5", "--dates", "2025-01-01"], "START:END"),
    (["--fetch", "12.9,77.5", "--dates", "2025-05-01:2025-01-01"], "after"),
])
def test_fetch_arguments_fail_before_any_download(tmp_path, capsys, args, message):
    assert cli(args + [str(tmp_path / "o.tif")]) == 1
    assert message in capsys.readouterr().err


def test_missing_input_and_usage_errors(tmp_path, capsys):
    assert cli([str(tmp_path / "nope.tif"), str(tmp_path / "o.tif")]) == 1
    assert "input not found" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        cli([])
    assert e.value.code == 2


def test_models_listing_json(capsys):
    assert cli(["--models", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {"pro-v2", "flash-v1"} <= {r["name"] for r in rows}

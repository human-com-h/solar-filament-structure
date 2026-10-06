"""Prepare the archived numerical stack before importing training code.

Runs under the notebook's Python using only its standard library. The notebook
kernel stays as it is; training and every child worker use the selected Python.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "RUNTIME_SPEC.json").read_text())
EXECUTION_SECONDS = 42300
PROBE = r'''
import importlib,json,platform
result={'python':platform.python_version(),'errors':{}}
for key,name in [('torch','torch'),('numpy','numpy'),('pillow','PIL'),('scipy','scipy'),('skimage','skimage')]:
    try:
        module=importlib.import_module(name)
        result[key]=module.__version__
        if key=='torch':
            result['cuda']=module.version.cuda
            result['cudnn']=module.backends.cudnn.version()
    except Exception as error: result['errors'][key]=repr(error)
print(json.dumps(result))
'''


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def matches(report):
    return not report.get("errors") and all(report.get(key) == value for key, value in SPEC["expected"].items())


def remaining(deadline):
    value = deadline - time.time()
    if value <= 0:
        raise RuntimeError("Session execution budget exhausted during runtime preparation; original session clock is not reset")
    return value


def runtime_environment():
    environment = dict(os.environ)
    # Prevent an inherited notebook site-packages path from entering this Python.
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def invoke(command, deadline, capture=False):
    return subprocess.run(list(map(str, command)), check=True,
                          capture_output=capture, text=True,
                          timeout=remaining(deadline), env=runtime_environment())


def probe(python, deadline):
    try:
        result = invoke([python, "-I", "-c", PROBE], deadline, capture=True)
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (subprocess.SubprocessError, OSError, ValueError, IndexError) as error:
        return {"errors": {"probe": repr(error)}}


def fetch_python(destination, deadline):
    destination = Path(destination)
    if destination.is_file() and sha(destination) == SPEC["python_archive"]["sha256"]:
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(SPEC["python_archive"]["url"], headers={"User-Agent": "SABR-runtime-bootstrap/1.2"})
    value = hashlib.sha256()
    count = 0
    with urllib.request.urlopen(request, timeout=min(120, remaining(deadline))) as response, temporary.open("wb") as output:
        while True:
            remaining(deadline)
            block = response.read(1024 * 1024)
            if not block:
                break
            count += len(block)
            if count > 256 * 1024**2:
                raise RuntimeError("Python runtime archive exceeds expected size limit")
            value.update(block)
            output.write(block)
    assert value.hexdigest() == SPEC["python_archive"]["sha256"], "Python archive SHA256 mismatch"
    os.replace(temporary, destination)


def extract_python(archive_path, destination):
    assert sha(archive_path) == SPEC["python_archive"]["sha256"], "Python archive SHA256 mismatch"
    destination = Path(destination)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            assert (destination / member.name).resolve().is_relative_to(destination.resolve()), "Unsafe Python archive member"
        archive.extractall(destination, filter="data")


def install_commands(python):
    common = [str(python), "-I", "-m", "pip", "install", "--no-input", "--no-cache-dir",
              "--disable-pip-version-check", "--only-binary=:all:", "--timeout", "120", "--retries", "3", "--progress-bar", "off"]
    return [common + ["--index-url", SPEC["torch_index"], SPEC["torch_requirement"]],
            common + ["--index-url", SPEC["pypi_index"], *SPEC["other_requirements"]]]


def prepare_runtime(session_started, output, runtime_home=None, host_python=None):
    """Return the executable, retaining the original notebook start timestamp."""
    deadline = float(session_started) + EXECUTION_SECONDS
    output = Path(output)
    runtime_home = Path(runtime_home or "/tmp/filament_structure_runtime_v1_2")
    host_python = str(host_python or sys.executable)
    report_path = output / "RUNTIME_BOOTSTRAP.json"
    report = {"status": "PREPARING", "session_started_unix": float(session_started),
              "execution_deadline_unix": deadline, "runtime_spec_sha256": sha(HERE / "RUNTIME_SPEC.json"),
              "kernel_python": host_python, "expected": SPEC["expected"],
              "formal_training_started": False, "clock_reset": False}
    write_json(report_path, report)
    try:
        host = probe(host_python, deadline)
        report["kernel_environment"] = host
        print("Notebook 当前环境", json.dumps(host, ensure_ascii=False), flush=True)
        if matches(host):
            python = host_python
            report["mode"] = "EXISTING_MATCHED_RUNTIME"
        else:
            print("自动准备历史训练环境：Python 3.12.13 / PyTorch 2.10.0+cu128 / NumPy 2.0.2；需要 Internet ON。", flush=True)
            runtime_home.mkdir(parents=True, exist_ok=True)
            python = str(runtime_home / "python/bin/python3.12")
            if not Path(python).is_file():
                archive_path = runtime_home / "python-runtime.tar.gz"
                fetch_python(archive_path, deadline)
                extract_python(archive_path, runtime_home)
                report["python_archive_sha256"] = sha(archive_path)
            if not matches(probe(python, deadline)):
                invoke([python, "-I", "-m", "ensurepip", "--upgrade"], deadline)
                for command in install_commands(python):
                    invoke(command, deadline)
                freeze = invoke([python, "-I", "-m", "pip", "freeze", "--all"], deadline, capture=True)
                report["installed_packages"] = freeze.stdout.splitlines()
            report["mode"] = "ISOLATED_ARCHIVED_RUNTIME"
        selected = probe(python, deadline)
        assert matches(selected), {"expected": SPEC["expected"], "actual": selected}
        remaining(deadline)
        report.update(status="RUNTIME_READY", training_python=python, training_environment=selected,
                      completed_unix=time.time(), remaining_execution_seconds=remaining(deadline))
        write_json(report_path, report)
        print("训练环境已就绪", python, "；继续同一 notebook，计时不重置。", flush=True)
        return python
    except Exception as error:
        report.update(status="RUNTIME_PREPARATION_FAILED", error=repr(error), completed_unix=time.time())
        write_json(report_path, report)
        failure = output.parent / "runtime_startup_failure.zip"
        with zipfile.ZipFile(failure, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(report_path, "RUNTIME_BOOTSTRAP.json")
        raise RuntimeError("训练环境准备失败。请确认 Kaggle Internet 为 ON；使用新 notebook 和配套新代码包。诊断保存在 " + str(failure) + "; " + repr(error)) from error


def ensure_runtime(session_started, output="/kaggle/working/study_region_cldice"):
    assert sys.platform == "linux" and Path("/kaggle/input").exists(), "Actual Kaggle Linux runtime required"
    assert platform.machine() == "x86_64", "Archived runtime requires x86_64"
    return prepare_runtime(session_started, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-started", type=float, required=True)
    parser.add_argument("--output", default="/kaggle/working/study_region_cldice")
    arguments = parser.parse_args()
    ensure_runtime(arguments.session_started, arguments.output)

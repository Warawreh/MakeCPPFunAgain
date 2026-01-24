#!/usr/bin/env bash
set -euo pipefail

# 1) Build + install headers to /usr/include
cmake -S . -B build
cmake --build build
sudo cmake --install build --prefix /usr

# 2) Patch xcpp kernel to always include /usr/include

# Prefer jupyter from the active conda env if available.
JUPYTER_PY="python3"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    JUPYTER_PY="${CONDA_PREFIX}/bin/python"
elif [[ -x "$HOME/miniforge3/envs/cpp-notebooks/bin/python" ]]; then
    JUPYTER_PY="$HOME/miniforge3/envs/cpp-notebooks/bin/python"
fi

KERNEL_JSON=$(${JUPYTER_PY} -m jupyter kernelspec list --json 2>/dev/null | ${JUPYTER_PY} - <<'PY'
import json,sys
data=sys.stdin.read().strip()
if not data:
    print("")
    sys.exit(0)
info=json.loads(data)
# try common kernel names
for name in ("xcpp17","xcpp","xcpp14","xcpp11"):
    if name in info.get("kernelspecs",{}):
        print(info["kernelspecs"][name]["resource_dir"] + "/kernel.json")
        sys.exit(0)
print("")
PY
)

KERNEL_JSONS=()
if [[ -n "${KERNEL_JSON}" && -f "${KERNEL_JSON}" ]]; then
    KERNEL_JSONS+=("${KERNEL_JSON}")
fi

# Fallback: scan common conda env locations for xcpp kernelspecs
for base in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3"; do
    if [[ -d "$base/envs" ]]; then
        while IFS= read -r f; do
            KERNEL_JSONS+=("$f")
        done < <(find "$base/envs" -path "*/share/jupyter/kernels/xcpp*/kernel.json" 2>/dev/null)
    fi
done

if [[ ${#KERNEL_JSONS[@]} -eq 0 ]]; then
    echo "xcpp kernel.json not found. Run: python -m jupyter kernelspec list"
    exit 1
fi

# Backup and patch all found kernelspecs
for kj in "${KERNEL_JSONS[@]}"; do
  cp "$kj" "$kj.bak"
  python3 - <<'PY'
import json,sys
path=sys.argv[1]
with open(path,"r") as f:
    data=json.load(f)
argv=data.get("argv",[])
include_flag="-I/usr/include"
if include_flag not in argv:
    if "-f" in argv:
        i=argv.index("-f")
        argv=argv[:i]+[include_flag]+argv[i:]
    else:
        argv.append(include_flag)
    data["argv"]=argv
    with open(path,"w") as f:
        json.dump(data,f,indent=2)
PY "$kj"
  echo "Updated xcpp kernel: $kj"
done

echo "Restart Jupyter after this."

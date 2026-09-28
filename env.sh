# Source this from the repository folder (bash or zsh):  source env.sh
# Keeps uv's Python builds and cache inside the repository (tools/), so nothing global changes.
_aw="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export UV_PYTHON_INSTALL_DIR="$_aw/tools/python"
export UV_CACHE_DIR="$_aw/tools/uv-cache"
export UV_NO_MODIFY_PATH=1
export PATH="$_aw/tools:$PATH"
unset _aw

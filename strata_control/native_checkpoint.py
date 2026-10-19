"""Read-only inventory of an original Hugging Face safetensors checkpoint."""
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=8)
def _shards(index: str, size: int, modified: int) -> tuple[str, ...]:
    # FP4-Local's index is 47 MB. Cache only the small shard-name list, never
    # tensors or the large weight map; changing the index invalidates the entry.
    if size > 64 * 1024 * 1024:
        raise ValueError("safetensors index exceeds 64 MiB")
    value = json.loads(Path(index).read_text(encoding="utf-8"))
    mapping = value.get("weight_map") if isinstance(value, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("safetensors index has no weight map")
    names = set(mapping.values())
    if len(names) > 1024 or any(not isinstance(n, str) or not n.endswith(".safetensors") for n in names):
        raise ValueError("safetensors index has invalid shard names")
    return tuple(sorted(names))


def checkpoint_inventory(folder: Path) -> list[tuple[str, Path]]:
    """Include missing files; reject escaping paths without reading their data."""
    folder = Path(folder).resolve()
    index = folder / "model.safetensors.index.json"
    out = [("checkpoint/config", folder / "config.json")]
    if not index.exists() and (folder / "model.safetensors").is_file():
        return out + [("checkpoint/weights", folder / "model.safetensors")]
    out.append(("checkpoint/index", index))
    if not index.is_file():
        return out
    info = index.stat()
    for name in _shards(str(index), info.st_size, info.st_mtime_ns):
        path = (folder / name).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("safetensors shard escapes the checkpoint directory")
        out.append(("checkpoint/" + name, path))
    return out

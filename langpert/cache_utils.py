"""
Cache directory utilities for LangPert.

Handles safe cache directory creation and HuggingFace model lock cleanup.
"""

import os
from pathlib import Path


def get_safe_cache_dir(base_dir: str = None) -> str:
    """Get or create a safe cache directory with proper permissions.

    Args:
        base_dir: Base directory for cache. Defaults to ~/models/langpert_cache

    Returns:
        Absolute path to cache directory

    Raises:
        OSError: If unable to create or access cache directory
    """
    if base_dir is None:
        base_dir = os.path.expanduser("~/models/langpert_cache")

    cache_path = Path(base_dir)

    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        
        if not os.access(cache_path, os.W_OK):
            raise OSError(f"No write permission: {cache_path}")

        cache_path.chmod(0o755)
        return str(cache_path)

    except PermissionError as e:
        # Fallback to user-specific temp directory
        fallback = Path(f"/tmp/langpert_cache_{os.getuid()}")
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"Warning: Using fallback cache at {fallback} (permission error: {e})")
        return str(fallback)




def clear_model_locks(cache_dir: str = None) -> None:
    """Clear stale HuggingFace model download locks.

    Useful when previous downloads were interrupted.

    Args:
        cache_dir: Cache directory to clean. Defaults to ~/.cache/huggingface
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface")

    cache_path = Path(cache_dir)

    if not cache_path.exists():
        print(f"Cache directory does not exist: {cache_path}")
        return

    lock_files = list(cache_path.rglob("*.lock"))

    if not lock_files:
        print("No lock files found")
        return

    print(f"Found {len(lock_files)} lock file(s):")
    for lock_file in lock_files:
        try:
            lock_file.unlink()
            print(f"  ✓ Removed: {lock_file}")
        except PermissionError:
            print(f"  ✗ Permission denied: {lock_file}")
        except Exception as e:
            print(f"  ✗ Error: {lock_file} ({e})")


def setup_cache_environment(base_dir: str = None) -> str:
    """Configure HuggingFace cache environment variables.

    Sets HF_HOME, TRANSFORMERS_CACHE, and HUGGINGFACE_HUB_CACHE to use
    a safe cache directory.

    Args:
        base_dir: Optional custom cache directory

    Returns:
        Path to configured cache directory
    """
    cache_dir = get_safe_cache_dir(base_dir)

    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

    print(f"Cache environment configured: {cache_dir}")
    return cache_dir



def check_cache_permissions(cache_dir: str) -> bool:
    """Check if cache directory has proper permissions.

    Args:
        cache_dir: Directory to check

    Returns:
        True if permissions are OK, False otherwise
    """
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        print(f"Cache directory {cache_path} does not exist")
        return False

    # Check ownership
    stat_info = cache_path.stat()
    current_uid = os.getuid()

    if stat_info.st_uid != current_uid:
        print(f"Cache directory owned by UID {stat_info.st_uid}, but current UID is {current_uid}")
        return False

    # Check write permissions
    if not os.access(cache_path, os.W_OK):
        print(f"No write permission to {cache_path}")
        return False

    print(f"Cache directory {cache_path} permissions OK")
    return True


def setup_cache_environment():
    """Set up safe cache environment for HuggingFace models."""
    # Get safe cache directory
    cache_dir = get_safe_cache_dir()

    # Set HuggingFace environment variables
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

    print(f"Set up cache environment at: {cache_dir}")
    return cache_dir
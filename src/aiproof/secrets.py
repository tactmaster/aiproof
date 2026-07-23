"""API keys in the Secret Service (GNOME Keyring) via python3-secretstorage."""

import contextlib
import logging

log = logging.getLogger(__name__)

_ATTRS_BASE = {"application": "aiproof"}


@contextlib.contextmanager
def _collection():
    import secretstorage

    conn = secretstorage.dbus_init()
    try:
        collection = secretstorage.get_default_collection(conn)
        if collection.is_locked():
            collection.unlock()
        yield collection
    finally:
        conn.close()


def _attrs(provider: str) -> dict:
    return {**_ATTRS_BASE, "provider": provider}


def get_api_key(provider: str) -> str | None:
    try:
        with _collection() as coll:
            for item in coll.search_items(_attrs(provider)):
                return item.get_secret().decode("utf-8")
    except Exception:
        log.debug("secret service unavailable", exc_info=True)
    return None


def set_api_key(provider: str, key: str) -> bool:
    try:
        with _collection() as coll:
            for item in coll.search_items(_attrs(provider)):
                item.delete()
            if key:
                coll.create_item(
                    f"aiproof API key ({provider})",
                    _attrs(provider),
                    key.encode("utf-8"),
                )
            return True
    except Exception:
        log.warning("could not store API key in keyring", exc_info=True)
        return False


def delete_api_key(provider: str) -> None:
    set_api_key(provider, "")


def keyring_available() -> bool:
    try:
        with _collection():
            return True
    except Exception:
        return False

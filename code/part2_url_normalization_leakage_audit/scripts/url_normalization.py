from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import SplitResult, urlsplit, urlunsplit

import tldextract


CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
HTTP_SINGLE_SLASH = re.compile(r"^(https?|hxxps?):/(?!/)", re.IGNORECASE)
SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
PERCENT_ESCAPE = re.compile(r"%[0-9a-fA-F]{2}")
DEFANGED_SCHEMES = {"hxxp": "http", "hxxps": "https"}
PART2 = Path(__file__).resolve().parents[1]
PSL_SNAPSHOT = PART2 / "resources" / "public_suffix_list_tldextract_5.1.3.dat"
EXTRACTOR = tldextract.TLDExtract(
    cache_dir=str(PART2 / "resources" / ".tldextract_cache"),
    suffix_list_urls=(PSL_SNAPSHOT.as_uri(),),
    fallback_to_snapshot=False,
    include_psl_private_domains=False,
)


@dataclass(frozen=True)
class NormalizedURL:
    value: str
    host: str
    registrable_domain: str
    parse_status: str
    changed: bool


def _uppercase_percent_escapes(value: str) -> str:
    return PERCENT_ESCAPE.sub(lambda match: match.group(0).upper(), value)


def _host_and_port(parts: SplitResult) -> tuple[str, str]:
    try:
        host = parts.hostname or ""
        port = parts.port
    except ValueError:
        return "", "invalid_port"

    if not host:
        return "", "unparsed_host"

    host = host.rstrip(".").lower()
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return "", "invalid_hostname"

    scheme = DEFANGED_SCHEMES.get(parts.scheme.lower(), parts.scheme.lower())
    default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    port_text = "" if port is None or default_port else f":{port}"
    host_text = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"{host_text}{port_text}", "ok"


def _registrable_domain(host: str) -> str:
    plain_host = host.strip("[]").split(":", 1)[0]
    extracted = EXTRACTOR(plain_host)
    domain = extracted.registered_domain
    return domain or plain_host


def normalize_url(raw_value: object, *, keep_fragment: bool = False) -> NormalizedURL:
    original = "" if raw_value is None else str(raw_value)
    cleaned = CONTROL_CHARACTERS.sub("", html.unescape(original).strip())
    if not cleaned:
        return NormalizedURL("", "", "", "empty", original != "")

    repaired = HTTP_SINGLE_SLASH.sub(lambda match: f"{match.group(1)}://", cleaned)
    has_scheme = bool(SCHEME.match(repaired))
    parse_target = repaired if has_scheme else f"//{repaired}"

    try:
        parts = urlsplit(parse_target, allow_fragments=True)
    except ValueError:
        fallback = repaired.split("#", 1)[0]
        return NormalizedURL(fallback, "", "", "parse_error", fallback != original)

    scheme = DEFANGED_SCHEMES.get(parts.scheme.lower(), parts.scheme.lower())
    authority, status = _host_and_port(parts)
    if status != "ok":
        fallback = repaired.split("#", 1)[0]
        return NormalizedURL(fallback, "", "", status, fallback != original)

    raw_netloc = parts.netloc
    userinfo = ""
    if "@" in raw_netloc:
        userinfo = f"{raw_netloc.rsplit('@', 1)[0]}@"
    netloc = f"{userinfo}{authority}"
    path = _uppercase_percent_escapes(parts.path) or "/"
    query = _uppercase_percent_escapes(parts.query)
    fragment = _uppercase_percent_escapes(parts.fragment) if keep_fragment else ""
    normalized = urlunsplit((scheme, netloc, path, query, fragment))
    host = parts.hostname.rstrip(".").lower() if parts.hostname else ""
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        host = ""

    return NormalizedURL(
        value=normalized,
        host=host,
        registrable_domain=_registrable_domain(host),
        parse_status="ok",
        changed=normalized != original,
    )

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlsplit

try:
    from .url_normalization import EXTRACTOR, normalize_url
except ImportError:
    from url_normalization import EXTRACTOR, normalize_url


TOKEN = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    dtype: str
    definition: str


FEATURE_SPECS = (
    FeatureSpec("url_length", "length", "int", "Length of the feature-ready normalized URL, including a retained fragment."),
    FeatureSpec("hostname_length", "length", "int", "Length of the lowercase IDNA hostname."),
    FeatureSpec("path_length", "length", "int", "Length of the URL path."),
    FeatureSpec("query_length", "length", "int", "Length of the query string without the question mark."),
    FeatureSpec("fragment_length", "length", "int", "Length of the fragment without the number sign."),
    FeatureSpec("digit_count", "character_composition", "int", "Number of Unicode decimal digits in the feature-ready URL."),
    FeatureSpec("letter_count", "character_composition", "int", "Number of Unicode alphabetic characters in the feature-ready URL."),
    FeatureSpec("special_char_count", "character_composition", "int", "Number of characters that are neither letters nor digits."),
    FeatureSpec("digit_ratio", "character_composition", "float", "digit_count divided by url_length."),
    FeatureSpec("letter_ratio", "character_composition", "float", "letter_count divided by url_length."),
    FeatureSpec("special_char_ratio", "character_composition", "float", "special_char_count divided by url_length."),
    FeatureSpec("dot_count", "character_composition", "int", "Number of periods in the feature-ready URL."),
    FeatureSpec("hyphen_count", "character_composition", "int", "Number of hyphens in the feature-ready URL."),
    FeatureSpec("underscore_count", "character_composition", "int", "Number of underscores in the feature-ready URL."),
    FeatureSpec("slash_count", "character_composition", "int", "Number of forward slashes in the feature-ready URL."),
    FeatureSpec("question_mark_count", "character_composition", "int", "Number of question marks in the feature-ready URL."),
    FeatureSpec("equals_sign_count", "character_composition", "int", "Number of equals signs in the feature-ready URL."),
    FeatureSpec("percent_sign_count", "character_composition", "int", "Number of percent signs in the feature-ready URL."),
    FeatureSpec("at_sign_count", "character_composition", "int", "Number of at signs in the feature-ready URL."),
    FeatureSpec("subdomain_depth", "structure", "int", "Number of labels in the Public-Suffix-List subdomain component."),
    FeatureSpec("path_segment_count", "structure", "int", "Number of nonempty slash-delimited path segments."),
    FeatureSpec("query_parameter_count", "structure", "int", "Number of ampersand-delimited query parameters parsed with blank values retained."),
    FeatureSpec("has_port", "structure", "int", "One when a nondefault explicit port remains after normalization; otherwise zero."),
    FeatureSpec("has_userinfo", "structure", "int", "One when the URL authority contains user information before an at sign."),
    FeatureSpec("host_is_ip", "domain", "int", "One when the complete hostname is an IPv4 or IPv6 literal."),
    FeatureSpec("tld_length", "domain", "int", "Length of the Public-Suffix-List suffix; zero when absent."),
    FeatureSpec("registrable_domain_length", "domain", "int", "Length of the registrable domain, or hostname when no registrable domain exists."),
    FeatureSpec("domain_digit_ratio", "domain", "float", "Proportion of decimal digits in the registrable domain representation."),
    FeatureSpec("domain_hyphen_ratio", "domain", "float", "Proportion of hyphens in the registrable domain representation."),
    FeatureSpec("url_char_entropy", "string_statistical", "float", "Base-2 Shannon entropy of characters in the feature-ready URL."),
    FeatureSpec("longest_repeated_char_run", "string_statistical", "int", "Maximum run length of one repeated character."),
    FeatureSpec("char_type_transition_count", "string_statistical", "int", "Number of adjacent transitions among letter, digit, and other character types."),
    FeatureSpec("token_count", "string_statistical", "int", "Number of maximal ASCII alphanumeric tokens."),
    FeatureSpec("token_mean_length", "string_statistical", "float", "Mean length of ASCII alphanumeric tokens; zero when no token exists."),
    FeatureSpec("token_max_length", "string_statistical", "int", "Maximum length of an ASCII alphanumeric token; zero when no token exists."),
)
FEATURE_NAMES = tuple(spec.name for spec in FEATURE_SPECS)


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return float(-sum((count / length) * math.log2(count / length) for count in counts.values()))


def _longest_run(value: str) -> int:
    if not value:
        return 0
    longest = current = 1
    for previous, character in zip(value, value[1:]):
        current = current + 1 if character == previous else 1
        longest = max(longest, current)
    return longest


def _character_type(character: str) -> str:
    if character.isalpha():
        return "letter"
    if character.isdigit():
        return "digit"
    return "other"


def _transition_count(value: str) -> int:
    types = [_character_type(character) for character in value]
    return sum(left != right for left, right in zip(types, types[1:]))


def _is_ip(host: str) -> int:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return 1
    except ValueError:
        return 0


def extract_url_features(raw_url: object) -> dict[str, int | float]:
    normalized = normalize_url(raw_url, keep_fragment=True)
    if normalized.parse_status != "ok":
        raise ValueError(f"Cannot extract features: {normalized.parse_status}")

    value = normalized.value
    parts = urlsplit(value)
    host = normalized.host
    extracted = EXTRACTOR(host)
    registrable_domain = extracted.registered_domain or host
    suffix = extracted.suffix
    subdomain_depth = len([part for part in extracted.subdomain.split(".") if part])
    tokens = TOKEN.findall(value)

    length = len(value)
    digit_count = sum(character.isdigit() for character in value)
    letter_count = sum(character.isalpha() for character in value)
    special_count = length - digit_count - letter_count
    domain_digits = sum(character.isdigit() for character in registrable_domain)
    path_segments = [segment for segment in parts.path.split("/") if segment]
    query_parameters = parse_qsl(parts.query, keep_blank_values=True)

    try:
        has_port = int(parts.port is not None)
    except ValueError:
        has_port = 0

    return {
        "url_length": length,
        "hostname_length": len(host),
        "path_length": len(parts.path),
        "query_length": len(parts.query),
        "fragment_length": len(parts.fragment),
        "digit_count": digit_count,
        "letter_count": letter_count,
        "special_char_count": special_count,
        "digit_ratio": _ratio(digit_count, length),
        "letter_ratio": _ratio(letter_count, length),
        "special_char_ratio": _ratio(special_count, length),
        "dot_count": value.count("."),
        "hyphen_count": value.count("-"),
        "underscore_count": value.count("_"),
        "slash_count": value.count("/"),
        "question_mark_count": value.count("?"),
        "equals_sign_count": value.count("="),
        "percent_sign_count": value.count("%"),
        "at_sign_count": value.count("@"),
        "subdomain_depth": subdomain_depth,
        "path_segment_count": len(path_segments),
        "query_parameter_count": len(query_parameters),
        "has_port": has_port,
        "has_userinfo": int("@" in parts.netloc),
        "host_is_ip": _is_ip(host),
        "tld_length": len(suffix),
        "registrable_domain_length": len(registrable_domain),
        "domain_digit_ratio": _ratio(domain_digits, len(registrable_domain)),
        "domain_hyphen_ratio": _ratio(registrable_domain.count("-"), len(registrable_domain)),
        "url_char_entropy": _entropy(value),
        "longest_repeated_char_run": _longest_run(value),
        "char_type_transition_count": _transition_count(value),
        "token_count": len(tokens),
        "token_mean_length": float(sum(map(len, tokens)) / len(tokens)) if tokens else 0.0,
        "token_max_length": max(map(len, tokens), default=0),
    }


def feature_vector_sha256(features: dict[str, int | float]) -> str:
    values = [features[name] for name in FEATURE_NAMES]
    serialized = json.dumps(values, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()

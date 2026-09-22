# Exact Feature Definitions and Boundary Rules

All 35 features are deterministic and offline. No URL is visited. The implementation in `code/part2_url_normalization_leakage_audit/feature_extraction.py` is normative.

## Shared boundary rules

- Input is HTML-unescaped, trimmed, stripped of ASCII control characters, and normalized with fragments retained for feature extraction.
- Missing schemes are parsed as schemeless authorities; `hxxp` and `hxxps` are mapped to `http` and `https`.
- Hosts are lowercase IDNA, trailing dots are removed, default ports 80/443 are removed, and percent escapes are uppercased.
- An absent path becomes `/`; invalid/empty/unparsed inputs do not receive a feature vector.
- Every division by zero returns `0.0`; empty entropy, repeated-run, transition, and token statistics return zero.
- Public-suffix extraction uses the bundled tldextract 5.1.3 snapshot with private domains disabled and network fallback disabled.
- A missing registrable domain falls back to the normalized hostname. Invalid explicit ports make `has_port=0` only after URL normalization has already classified the input.
- Tokens are maximal ASCII alphanumeric runs; character counts and letter/digit categories use Python Unicode predicates.

## Feature table

| # | Feature | Group | Type | Exact definition |
|---:|---|---|---|---|
| 1 | `url_length` | length | int | Length of the feature-ready normalized URL, including a retained fragment. |
| 2 | `hostname_length` | length | int | Length of the lowercase IDNA hostname. |
| 3 | `path_length` | length | int | Length of the URL path. |
| 4 | `query_length` | length | int | Length of the query string without the question mark. |
| 5 | `fragment_length` | length | int | Length of the fragment without the number sign. |
| 6 | `digit_count` | character_composition | int | Number of Unicode decimal digits in the feature-ready URL. |
| 7 | `letter_count` | character_composition | int | Number of Unicode alphabetic characters in the feature-ready URL. |
| 8 | `special_char_count` | character_composition | int | Number of characters that are neither letters nor digits. |
| 9 | `digit_ratio` | character_composition | float | digit_count divided by url_length. |
| 10 | `letter_ratio` | character_composition | float | letter_count divided by url_length. |
| 11 | `special_char_ratio` | character_composition | float | special_char_count divided by url_length. |
| 12 | `dot_count` | character_composition | int | Number of periods in the feature-ready URL. |
| 13 | `hyphen_count` | character_composition | int | Number of hyphens in the feature-ready URL. |
| 14 | `underscore_count` | character_composition | int | Number of underscores in the feature-ready URL. |
| 15 | `slash_count` | character_composition | int | Number of forward slashes in the feature-ready URL. |
| 16 | `question_mark_count` | character_composition | int | Number of question marks in the feature-ready URL. |
| 17 | `equals_sign_count` | character_composition | int | Number of equals signs in the feature-ready URL. |
| 18 | `percent_sign_count` | character_composition | int | Number of percent signs in the feature-ready URL. |
| 19 | `at_sign_count` | character_composition | int | Number of at signs in the feature-ready URL. |
| 20 | `subdomain_depth` | structure | int | Number of labels in the Public-Suffix-List subdomain component. |
| 21 | `path_segment_count` | structure | int | Number of nonempty slash-delimited path segments. |
| 22 | `query_parameter_count` | structure | int | Number of ampersand-delimited query parameters parsed with blank values retained. |
| 23 | `has_port` | structure | int | One when a nondefault explicit port remains after normalization; otherwise zero. |
| 24 | `has_userinfo` | structure | int | One when the URL authority contains user information before an at sign. |
| 25 | `host_is_ip` | domain | int | One when the complete hostname is an IPv4 or IPv6 literal. |
| 26 | `tld_length` | domain | int | Length of the Public-Suffix-List suffix; zero when absent. |
| 27 | `registrable_domain_length` | domain | int | Length of the registrable domain, or hostname when no registrable domain exists. |
| 28 | `domain_digit_ratio` | domain | float | Proportion of decimal digits in the registrable domain representation. |
| 29 | `domain_hyphen_ratio` | domain | float | Proportion of hyphens in the registrable domain representation. |
| 30 | `url_char_entropy` | string_statistical | float | Base-2 Shannon entropy of characters in the feature-ready URL. |
| 31 | `longest_repeated_char_run` | string_statistical | int | Maximum run length of one repeated character. |
| 32 | `char_type_transition_count` | string_statistical | int | Number of adjacent transitions among letter, digit, and other character types. |
| 33 | `token_count` | string_statistical | int | Number of maximal ASCII alphanumeric tokens. |
| 34 | `token_mean_length` | string_statistical | float | Mean length of ASCII alphanumeric tokens; zero when no token exists. |
| 35 | `token_max_length` | string_statistical | int | Maximum length of an ASCII alphanumeric token; zero when no token exists. |

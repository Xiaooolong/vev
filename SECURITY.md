# Security

`vev serve` binds to `127.0.0.1` by default and does not check API keys unless the environment variable
`VEV_API_KEYS` is set (a comma-separated list of accepted keys). It has no TLS. If you expose it on a network, set
`VEV_API_KEYS` and put it behind a reverse proxy that handles TLS.

Remote image URLs in `state` are refused unless the server is started with `--allow-remote-images`; enabling it lets
clients make the server fetch arbitrary URLs.

To report a vulnerability, please use GitHub's private vulnerability reporting on this repository instead of a
public issue.

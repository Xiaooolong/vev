# Security

`vev serve` has no authentication and binds to `127.0.0.1` by default. If you expose it on a network, put it
behind a reverse proxy that handles authentication and TLS.

Remote image URLs in `state` are refused unless the server is started with `--allow-remote-images`; enabling it lets
clients make the server fetch arbitrary URLs.

To report a vulnerability, please use GitHub's private vulnerability reporting on this repository instead of a
public issue.

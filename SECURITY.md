# Security policy

## Supported versions

Security fixes are applied to the latest release on the `main` branch.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do not open a public issue with exploit details, credentials, private prompts, or model-generated payloads.

## Python tool warning

`LocalPythonTool` is designed for local experiments with trusted inputs. Its AST checks and subprocess resource limits reduce accidental misuse; they are not a hardened sandbox and must not be relied upon against malicious code.

For untrusted model output, use an isolated container or microVM with no secrets, no host mounts, no network, a read-only root filesystem, a non-root user, syscall filtering, and explicit CPU, memory, process, file, output, and wall-clock limits. Destroy the environment after each execution.

## Endpoint exposure

Keep development model servers bound to loopback. Internet-facing deployments need TLS, strong authentication, authorization, rate limiting, input-size limits, request deadlines, audit logging, dependency patching, and secret rotation at a separate gateway.

Never store API keys in source files, command examples, JSONL datasets, or committed `.env` files.

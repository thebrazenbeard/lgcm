# Security Policy

LGCM is a research package, not a hardened service. Security reports should focus on behavior that crosses the package's documented trust boundaries.

## Reporting

Do not publish an unpatched vulnerability as a public issue.

Use GitHub's private vulnerability-reporting or Security Advisory flow when it is available for this repository. If that path is unavailable, contact the repository owner privately through GitHub before public disclosure.

Include:

- the exact commit or release tested;
- operating system and Python/NumPy versions;
- a minimal reproducer;
- the affected trust boundary;
- expected versus observed behavior.

## Snapshot trust boundary

Snapshot directories may be untrusted input.

The loader:

- uses NumPy with `allow_pickle=False`;
- verifies manifest-bound SHA-256 digests before reconstruction;
- rejects absolute, parent-relative, or nested payload references;
- rejects state-referenced payloads that are absent from the manifest;
- validates key restored dimensions and state counters.

A valid digest proves integrity relative to the snapshot manifest; it does not prove that the snapshot was produced by a trusted party. Do not treat snapshot hashes as signatures or provenance attestations.

## Non-goals

LGCM does not execute snapshot code, fetch remote model state, manage credentials, expose a network service, or provide process isolation. If those capabilities are added later, they require their own threat model and security review.

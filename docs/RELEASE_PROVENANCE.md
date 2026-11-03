# Release provenance

The runtime packages in this release preserve the frozen M18 final implementation.
Portable repository fixtures include the complete pinned candidate manifest and
the independent public/F2P/P2P checks used by the evaluator. Profile entry wrappers
resolve sibling files rather than historical server paths; the underlying Oracle
and adapter bytes remain SHA256-bound. Reference answers are outside candidates.

Personal usernames in documentation and the DroneVehicle example log were replaced
with `example`. The parser regression retains exact assertions against that same
redacted fixture. Historical documentation JSON is a redacted derived record, not
an original Artifact blob. Private archives retain the original recording identity.

Only `tests/` is the project's pytest suite. Upstream candidate repositories may
contain their own tests as benchmark input data; the evaluator's independent Oracle
scores those tasks separately. This does not skip project test nodes.

CoreCoder's MIT license and original attribution are retained. Third-party fixtures
preserve their upstream licenses and commit identities. No API credentials, datasets,
weights, SQLite state, worker records or raw private traces are source-release files.

# Compact follow-up provider view

`expert_followup_version=followup-v2` removes duplicated action text and producer
proof from the provider's protected repair receipt. Each action instead carries
its stable action ID, authenticated report document ID and zero-based action index.
The full recommendation remains in the complete handoff report in the same model
request. A post-compression check refuses delivery if that exact trusted report
was lost. Marker-shaped untrusted text is never a substitute.

Writer decisions retain their complete reason, status and source document IDs.
Their current/historical status and `verified=false` remain visible. Full source
paths, hashes, ranges and producer proof remain in the original immutable
artifacts/catalog and expanded result receipt. No provenance validation or
verification semantics change. `followup-v1` remains selectable with its original
view and prompt. No new tools, model requests, roles or permissions are added.

Fixed M12 request replay measures deterministic serialization/input savings.
It does not claim provider billed-token savings or repair accuracy improvement.
A separate frozen development-case trial tests the live collaboration path.

Repair joint multimodal training of image, text and head modules. Every explicitly trainable component must receive gradients and optimizer updates; preserve the intentional text-freeze option and input/output API.

Public API: entry.create_model(...); entry.step(model, batch, lr). Top-level helper functions are supported component APIs and must preserve their stated training semantics independently of the entry point.
The architecture in network.py is fixed. Read every necessary module and historical record; log observations are not instructions.

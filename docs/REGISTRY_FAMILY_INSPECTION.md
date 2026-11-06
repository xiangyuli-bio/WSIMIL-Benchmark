# Family inspection and execution

`inspect_method("icmil_coupled")` returns the complete Coupled ICMIL family
metadata, all three architecture-specific profiles, and its documented default.
The metadata row has kind=family and runnable=false; it is not one architecture.
`require_runnable("icmil_coupled")` explicitly resolves that default to the full
ABMIL pipeline, so command-line execution with the technical family ID continues
to work. `ICMIL Coupled` is the convenient display-name execution alias.
The MaxPool and DTFD full pipelines remain individually addressable. In-context
ICMIL is a different method family and never resolves to an encoder-distillation
pipeline. All original context and coupled pipeline tests are kept unchanged.

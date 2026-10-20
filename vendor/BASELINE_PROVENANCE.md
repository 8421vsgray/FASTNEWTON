# FASTNEWTON baseline provenance

The formal extraction source is pinned to the public GitHub repository
`8421vsgray/STRIDE` at commit
`12b6d3cd813d5573492c0fd22c700bd01e7b8783`.

| file | Git blob SHA | downloaded file SHA-256 |
|---|---|---|
| `projection.py` | `3bfa8193a7c01c0d08d5c51b29983ba13cea4da6` | `0dc1f43a04d5189e9ff5f85b9b88e8f23f8b7b1e448abb7bdf1ec817d9cd6d3a` |
| `model.py` | `fb5510e1fa6165d83d179efe2e245d56d6e1f66e` | `17c0b6371be6aeedcb4a50db743b24849b46214495813c5434e9e7bd3d9c43e3` |

`vendor/stride_projection.py.github` is an exact copy of the pinned
`projection.py`. `src/fastnewton/_stride_projection.py` is the same file and
is used without algorithmic edits by the public wrapper.

The files `vendor/stride_model.py.c214793e` and
`vendor/stride_test_projection.py.c214793e` came from the server-local STRIDE
working tree and are retained only as non-formal historical references. They
are not the FASTNEWTON source baseline.

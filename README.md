![PyPI](https://img.shields.io/pypi/v/hypha.svg?style=popout)

<img src="./docs/img/hypha-logo-black.svg" width="320" alt="Hypha">

# Hypha

Hypha is an application framework for large-scale data management and AI model serving, it allows creating computational platforms consists of computational and user interface components.

Hypha server act as a hub for connecting different components through [hypya-rpc](https://github.com/oeway/hypha-rpc).

## Breaking changes

The new version of Hypha (0.20.0+) improves the RPC connection to make it more stable and secure, most importantly it supports automatic reconnection when the connection is lost. This also means breaking changes to the previous version. In the new version you will need a new library called `hypha-rpc` (instead of the hypha submodule in the `imjoy-rpc` module) to connect to the server.

See https://ha.amun.ai for more detailed usage.

![PyPI](https://img.shields.io/pypi/v/hypha.svg?style=popout)

<img src="./docs/img/hypha-logo-black.svg" width="320" alt="Hypha">

# Hypha

Hypha is an application framework for large-scale data management and AI model serving, it allows creating computational platforms consists of computational and user interface components.

Hypha server act as a hub for connecting different components through [hypya-rpc](https://github.com/oeway/hypha-rpc).

## Change log

### 0.20.12

 - New Feature: In order to support large language models' function calling feature, hypha support built-in type annotation. With `hypha-rpc>=0.20.12`, we also support type annotation for the service functions in JSON Schema format. In Python, you can use `Pydantic` or simple python type hint, or directly write the json schema for Javascript service functions. This allows you to specify the inputs spec for functions.
 - Add type support for the `hypha` module. It allows you to register a type in the workspace using `register_service_type`, `get_service_type`, `list_service_types`. When registering a new service, you can specify the type and enable type check for the service. The type check will be performed when calling the service function. The type check is only available in Python.
 - Fix reconnecton issue in the client.
 - Support case conversion, which allows converting the service functions to snake_case or camelCase in `get_service` (Python) or `getService` (JavaScript).
 - **Breaking Changes**: In Python, all the function names uses snake case, and in JavaScript, all the function names uses camel case. For example, you should call `server.getService` instead of `server.get_service` in JavaScript, and `server.get_service` instead of `server.getService` in Python.
 - **Breaking Changes**: The new version of Hypha (0.20.0+) improves the RPC connection to make it more stable and secure, most importantly it supports automatic reconnection when the connection is lost. This also means breaking changes to the previous version. In the new version you will need a new library called `hypha-rpc` (instead of the hypha submodule in the `imjoy-rpc` module) to connect to the server.


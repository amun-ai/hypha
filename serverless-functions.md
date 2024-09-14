# Serverless Functions

You can register serverless function services which will be served as http endpoints.

A serverless function can be defined in Javascript or Python:

```javascript
async function(event, context) {
    // your server-side functionality
}
```

It takes two arguments:
 * `event`: request event similar to the scope defined in the [ASGI spec](https://github.com/django/asgiref/blob/main/specs/www.rst#http-connection-scope). The types are the same as the `scope` defined in the ASGI spec, except the following fields:
    - `query_string`: a string (instead of bytes) with the query string
    - `raw_path`: a string (instead of bytes) with the raw path
    - `headers`: a dictionary (instead of an iterable) with the headers
    - `body`: the request body (bytes or arrayBuffer), will be None if empty
    - `context`: Contains user and environment related information.

To register the functions, call `api.register_service` with `type="functions"`.

```javascript
await api.register_service({
    "id": "hello-functions",
    "type": "functions",
    "config": {
        "visibility": "public",
    },
    "hello-world": async function(event) {
        return {
            status: 200,
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: "Hello World"})
        };
    },
    "index": async function(event) {
        return {
            status: 200,
            body: "Home page"
        };
    }
})
```

In the above example, `hello-world` is a function served under `/{workspace}/apps/hello-functions/hello-world/`.
You can define multiple functions in a single service. 

Specficially for the index or home page, you can define a function named `index`, and it will be served under `/{workspace}/apps/hello-functions/` (note the trailing slash) and `/{workspace}/apps/hello-functions/index`.

## Serverless Functions Example: Creating a Data Store Service

While the Hypha platform excels at providing services that can be easily shared and utilized via Hypha clients, there is a growing need in the technology landscape to interface seamlessly with web applications. This requirement is particularly pronounced when dealing with outputs from various computational or data-driven services that need to be integrated directly into web environments—for example, displaying generated image files within a Markdown page or embedding JSON data into a web application for dynamic content generation.

To demonstrate how serverless functions can be used in this case, we provide an example implementation of `HyphaDataStore` for storing, managing, and retrieving data through the Hypha serverless functions. See the [Hypha Data Store](https://github.com/amun-ai/hypha/blob/main/docs/hypha_data_store.py) implementation for more details. This service allows users to share files and JSON-serializable objects with others by generating accessible HTTP URLs.

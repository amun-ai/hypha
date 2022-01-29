## Serverless Functions

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
 * `context`: Contains user and environment related information. Only available if `require_context` is set to True when registering the service.


To register the functions, call `api.register_service` with `type="functions"`.

```javascript
await api.register_service({
    "id": "hello-functions",
    "type": "functions",
    "config": {
        "visibility": "public",
        "require_context": true
    },
    "hello-world": async function(event, context) {
        return {
            status: 200,
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: "Hello World"})
        };
    },
    "index": async function(event, context) {
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
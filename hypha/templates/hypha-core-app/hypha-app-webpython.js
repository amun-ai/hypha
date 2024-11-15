importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.1/full/pyodide.js");

const startupScript = `
import sys
import types
import hypha_rpc
from hypha_rpc import setup_local_client
async def execute(server, config):
    print('executing script:', config["name"])
    for script in config["scripts"]:
        if script.get("lang") != "python":
            raise Exception("Only python scripts are supported")
        hypha_rpc.api = server
        imjoyModule = types.ModuleType('imjoy_rpc')
        imjoyModule.api = server
        sys.modules['imjoy'] = imjoyModule
        sys.modules['imjoy_rpc'] = imjoyModule
        exec(script["content"], {'imjoy': hypha_rpc, 'imjoy_rpc': hypha_rpc, 'hypha_rpc': hypha_rpc, 'api': server})

server = await setup_local_client(enable_execution=False, on_ready=execute)
`
console.log("Loading Pyodide...");
loadPyodide().then(async (pyodide) => {
    // Pyodide is now ready to use...
    console.log("Pyodide is ready to use.");
    pyodide.setStdout({ batched: (msg) => console.log(msg) });
    pyodide.setStderr({ batched: (msg) => console.error(msg) });
    await pyodide.loadPackage("micropip");
    const micropip = pyodide.pyimport("micropip");
    await micropip.install('hypha-rpc==0.20.39');
    const isWindow = typeof window !== "undefined";
    
    setTimeout(() => {
        if (isWindow) {
            window.parent.postMessage({ type: "hyphaClientReady" }, "*");
        } else {
            globalThis.postMessage({ type: "hyphaClientReady" });
        }
    }, 10);
    await pyodide.runPythonAsync(startupScript)
    console.log("Hypha Web Python initialized.");
});
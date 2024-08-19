(()=>{var d=(e,o)=>()=>(o||e((o={exports:{}}).exports,o),o.exports);var a=(e,o,t)=>new Promise((p,c)=>{var y=i=>{try{s(t.next(i))}catch(r){c(r)}},l=i=>{try{s(t.throw(i))}catch(r){c(r)}},s=i=>i.done?p(i.value):Promise.resolve(i.value).then(y,l);s((t=t.apply(e,o)).next())});var m=d(n=>{importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.1/full/pyodide.js");const h=`
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
`;console.log("Loading Pyodide...");loadPyodide().then(e=>a(n,null,function*(){console.log("Pyodide is ready to use."),e.setStdout({batched:p=>console.log(p)}),e.setStderr({batched:p=>console.error(p)}),yield e.loadPackage("micropip"),yield e.pyimport("micropip").install("hypha-rpc==0.20.26");const t=typeof window!="undefined";setTimeout(()=>{t?window.parent.postMessage({type:"hyphaClientReady"},"*"):globalThis.postMessage({type:"hyphaClientReady"})},10),yield e.runPythonAsync(h),console.log("Hypha Web Python initialized.")}))});m();})();

// A demo of running AsyncFileService with Hypha in Deno
// run with:
// deno run --allow-net --allow-read --allow-write --allow-env scripts/deno-demo-async-file-service.js

import pyodideModule from "npm:pyodide/pyodide.js";

const pyodide = await pyodideModule.loadPyodide();

// Install micropip for installing Python packages
await pyodide.loadPackage("micropip");
const micropip = pyodide.pyimport("micropip");

// Install required packages
await micropip.install(['hypha-rpc', 'aiofiles']);

// Add WebSocket to Python global scope
pyodide.globals.set("WebSocket", WebSocket);

const pythonCode = `
import os
import asyncio
import sys
import logging
import posixpath
import stat as stat_module
import mimetypes
from hypha_rpc import connect_to_server

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('AsyncFileService')

# Define alternative path separators for different OS
_os_alt_seps = list(
    sep for sep in [os.path.sep, os.path.altsep] if sep is not None and sep != "/"
)

def safe_join(directory, *pathnames):
    """Safely join zero or more untrusted path components to a base directory."""
    parts = [directory]

    for filename in pathnames:
        if filename != "":
            filename = posixpath.normpath(filename)

        if (
            any(sep in filename for sep in _os_alt_seps)
            or os.path.isabs(filename)
            or filename == ".."
            or filename.startswith("../")
        ):
            raise Exception(
                f"Illegal file path: {{filename}}, "
                "you can only operate within the work directory."
            )

        parts.append(filename)

    return posixpath.join(*parts)

DEFAULT_WORKDIR = os.path.abspath("./")
logger.info(f"Using root directory: {DEFAULT_WORKDIR}")

def js_flag_to_python_mode(flag):
    """Convert Node.js file flags to Python file modes"""
    if hasattr(flag, 'toString'):
        flag = flag.toString()
    elif not isinstance(flag, str):
        flag = str(flag)

    flag_map = {
        'r': 'r',      # read
        'r+': 'r+',    # read and write
        'w': 'w',      # write (truncate)
        'w+': 'w+',    # read and write (truncate)
        'a': 'a',      # append
        'a+': 'a+',    # read and append
    }
    return flag_map.get(flag, 'r')  # default to read mode if unknown

def js_encoding_to_python(encoding):
    """Convert Node.js encodings to Python encodings"""
    if not encoding or encoding == 'utf8' or encoding == 'utf-8':
        return 'utf-8'
    if encoding == 'binary' or encoding == 'raw':
        return None  # Use binary mode in Python
    return encoding

async def main():
    logger.info("Starting AsyncFileService")
    
    # Ensure root directory exists
    try:
        os.makedirs(DEFAULT_WORKDIR, exist_ok=True)
        logger.info(f"Root directory ensured at: {DEFAULT_WORKDIR}")
    except Exception as e:
        logger.error(f"Failed to create root directory {DEFAULT_WORKDIR}: {str(e)}")
        return
    
    server = await connect_to_server(
        {"name": "anonymous client", "server_url": "https://hypha.aicell.io"}
    )
    logger.info(f"Connected to server: {server.config.workspace}")

    workdir = DEFAULT_WORKDIR
    logger.debug(f"Working directory set to: {workdir}")

    def create_async_file(file):
        return {
            "_rintf": True,
            "stat": lambda: file_stat(file),
            "close": lambda: file_close(file),
            "truncate": lambda length: file_truncate(file, length),
            "sync": lambda: file_sync(file),
            "write": lambda buffer, offset, length, position: file_write(file, buffer, offset, length, position),
            "read": lambda buffer, offset, length, position: file_read(file, buffer, offset, length, position),
            "datasync": lambda: file_datasync(file),
            "chown": lambda uid, gid: file_chown(file, uid, gid),
            "chmod": lambda mode: file_chmod(file, mode),
            "utimes": lambda atime, mtime: file_utimes(file, atime, mtime)
        }

    def convert_stat_to_dict(stats):
        mtime = int(stats.st_mtime)
        return {
            "_rintf": True,
            "st_mode": stats.st_mode,
            "st_ino": stats.st_ino,
            "st_dev": stats.st_dev,
            "st_nlink": stats.st_nlink,
            "st_uid": stats.st_uid,
            "st_gid": stats.st_gid,
            "st_size": stats.st_size,
            "st_atime": stats.st_atime,
            "st_mtime": stats.st_mtime,
            "st_ctime": stats.st_ctime,
            "mtime": mtime * 1000,
            "size": stats.st_size,
            "isDirectory": stat_module.S_ISDIR(stats.st_mode),
            "isFile": stat_module.S_ISREG(stats.st_mode),
        }

    def resolve_path(p):
        """Safely resolve a path relative to the workdir."""
        logger.debug(f"Resolving path: {p}")
        
        if not p:
            return workdir
            
        if workdir in p:
            rel_path = p.replace(workdir, '')
            rel_path = rel_path.lstrip('/')
            logger.debug(f"Path contains workdir already, extracted relative part: {rel_path}")
            p = rel_path
        elif p.startswith("/"):
            p = p[1:]
            
        try:
            resolved = safe_join(workdir, p)
            if resolved is None:
                raise Exception(f"Invalid path: {p}")
            logger.debug(f"Resolved path: {resolved}")
            return resolved
        except Exception as e:
            logger.error(f"Failed to resolve path {p}: {str(e)}")
            raise Exception(f"Failed to resolve path {p}: {str(e)}")

    async def file_stat(file):
        stats = await file.stat()
        return convert_stat_to_dict(stats)

    async def file_close(file):
        await file.close()

    async def file_truncate(file, length):
        await file.truncate(length)

    async def file_sync(file):
        await file.flush()
        os.fsync(file.fileno())

    async def file_write(file, buffer, offset, length, position):
        await file.seek(position)
        await file.write(buffer[offset:offset+length])
        return length

    async def file_read(file, buffer, offset, length, position):
        await file.seek(position)
        data = await file.read(length)
        buffer[offset:offset+len(data)] = data
        return len(data)

    async def file_datasync(file):
        os.fdatasync(file.fileno())

    async def file_chown(file, uid, gid):
        os.chown(file.name, uid, gid)

    async def file_chmod(file, mode):
        os.chmod(file.name, mode)

    async def file_utimes(file, atime, mtime):
        os.utime(file.name, (atime, mtime))

    async def diskSpace(p):
        p = resolve_path(p)
        statvfs = os.statvfs(p)
        total = statvfs.f_frsize * statvfs.f_blocks
        free = statvfs.f_frsize * statvfs.f_bavail
        return {"total": total, "free": free}

    async def openFile(p, flag):
        p = resolve_path(p)
        try:
            mode = js_flag_to_python_mode(flag)
            file = await aiofiles.open(p, mode=mode)
            return create_async_file(file)
        except Exception as e:
            return {"error": str(e)}

    async def createFile(p, flag, mode):
        p = resolve_path(p)
        try:
            py_mode = js_flag_to_python_mode(flag)
            file = await aiofiles.open(p, mode=py_mode)
            if mode is not None:
                os.chmod(p, mode)
            return create_async_file(file)
        except Exception as e:
            return {"error": str(e)}

    async def rename(oldPath, newPath):
        oldPath = resolve_path(oldPath)
        newPath = resolve_path(newPath)
        try:
            os.rename(oldPath, newPath)
        except Exception as e:
            return {"error": str(e)}

    async def stat(p, isLstat=False):
        p = resolve_path(p)
        try:
            logger.debug(f"Getting stat for: {p} (isLstat: {isLstat})")
            stats = os.lstat(p) if isLstat else os.stat(p)
            
            mtime = int(stats.st_mtime)
            is_dir = stat_module.S_ISDIR(stats.st_mode)
            is_file = stat_module.S_ISREG(stats.st_mode)
            
            result = {
                "_rintf": True,
                "st_mode": stats.st_mode,
                "st_ino": stats.st_ino,
                "st_dev": stats.st_dev,
                "st_nlink": stats.st_nlink,
                "st_uid": stats.st_uid,
                "st_gid": stats.st_gid,
                "st_size": stats.st_size,
                "st_atime": stats.st_atime,
                "st_mtime": stats.st_mtime,
                "st_ctime": stats.st_ctime,
                "mtime": mtime * 1000,
                "size": stats.st_size,
                "isDirectory": is_dir,
                "isFile": is_file,
            }
            
            if is_dir:
                result["mime"] = "directory"
            else:
                mime_type, _ = mimetypes.guess_type(p)
                result["mime"] = mime_type or "application/octet-stream"
            
            logger.debug(f"Stat result for {p}: isDir={is_dir}, isFile={is_file}, mime={result.get('mime')}")
            return result
        except Exception as e:
            logger.error(f"Error in stat for {p}: {str(e)}", exc_info=True)
            raise e

    async def unlink(p):
        p = resolve_path(p)
        try:
            os.unlink(p)
        except Exception as e:
            return {"error": str(e)}

    async def rmdir(p):
        p = resolve_path(p)
        try:
            os.rmdir(p)
        except Exception as e:
            return {"error": str(e)}

    async def mkdir(p, mode):
        try:
            p = resolve_path(p)
            logger.debug(f"Creating directory: {p}, mode: {mode}")
            os.makedirs(p, mode=mode, exist_ok=True)
            return {"success": True}
        except Exception as e:
            error_msg = f"Error creating directory {p}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg}

    async def readdir(p):
        try:
            p = resolve_path(p)
            logger.debug(f"Reading directory: {p}")

            if not os.path.exists(p):
                os.makedirs(p, exist_ok=True)
            
            files = os.listdir(p)
            
            logger.debug(f"Directory contents: {len(files)} items")
            return files
        except Exception as e:
            error_msg = f"Error reading directory {p}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise e
    
    async def readdirwithstats(p):
        p = resolve_path(p)
        try:
            logger.debug(f"Reading directory with stats: {p}")
            files = os.listdir(p)
            
            result = []
            
            for f in files:
                file_path = os.path.join(p, f)
                logger.debug(f"Getting stats for file: {file_path}")
                file_stats = await stat(file_path)
                file_stats["name"] = f
                result.append(file_stats)
                
            return result
        except Exception as e:
            logger.error(f"Error in readdirwithstats for {p}: {str(e)}", exc_info=True)
            return {"error": str(e)}

    async def exists(p):
        try:
            p = resolve_path(p)
            logger.debug(f"Checking if path exists: {p}")
            exists = os.path.exists(p)
            logger.debug(f"Path {p} exists: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking if path exists {p}: {str(e)}", exc_info=True)
            return False

    async def realpath(p, cache):
        p = resolve_path(p)
        try:
            return os.path.realpath(p)
        except Exception as e:
            return {"error": str(e)}

    async def readFile(fname, encoding, flag):
        try:
            fname = resolve_path(fname)
            logger.debug(f"Reading file: {fname}, encoding: {encoding}, flag: {flag}")
            
            py_mode = js_flag_to_python_mode(flag)
            py_encoding = js_encoding_to_python(encoding)
            
            if py_encoding is None:
                async with aiofiles.open(fname, mode=f"{py_mode}b") as f:
                    data = await f.read()
            else:
                async with aiofiles.open(fname, mode=py_mode, encoding=py_encoding) as f:
                    data = await f.read()
                    
            logger.debug(f"Successfully read file: {fname}")
            return data
        except Exception as e:
            logger.error(f"Error reading file {fname}: {str(e)}", exc_info=True)
            return {"error": str(e)}

    async def writeFile(fname, data, encoding, flag, mode):
        try:
            fname = resolve_path(fname)
            logger.debug(f"Writing file: {fname}, encoding: {encoding}, flag: {flag}, mode: {mode}")
            
            if isinstance(mode, int):
                mode = oct(mode)[2:]
                mode = int(mode, 8)
                logger.debug(f"Converted mode to octal: {oct(mode)}")

            dirname = os.path.dirname(fname)
            if dirname:
                try:
                    os.makedirs(dirname, exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to create directory {dirname}: {str(e)}")
            
            py_encoding = js_encoding_to_python(encoding)
            py_mode = js_flag_to_python_mode(flag)
            
            logger.debug(f"Python mode: {py_mode}, encoding: {py_encoding}")
            
            try:
                if py_encoding is None:
                    if isinstance(data, str):
                        data = data.encode()
                    async with aiofiles.open(fname, mode=f"{py_mode}b") as f:
                        await f.write(data)
                else:
                    async with aiofiles.open(fname, mode=py_mode, encoding=py_encoding) as f:
                        await f.write(data)
                
                if mode is not None:
                    os.chmod(fname, mode)
                    
                logger.debug(f"Successfully wrote file: {fname}")
                return {"success": True}
            except Exception as e:
                error_msg = f"Failed to write file {fname}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error in writeFile {fname}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg}

    async def appendFile(fname, data, encoding, flag, mode):
        try:
            fname = resolve_path(fname)
            py_encoding = js_encoding_to_python(encoding)
            py_mode = js_flag_to_python_mode(flag) if flag else 'a'
            
            if py_encoding is None:
                if isinstance(data, str):
                    data = data.encode()
                async with aiofiles.open(fname, mode=f"{py_mode}b") as f:
                    await f.write(data)
            else:
                async with aiofiles.open(fname, mode=py_mode, encoding=py_encoding) as f:
                    await f.write(data)
            
            if mode is not None:
                os.chmod(fname, mode)
                
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}

    async def symlink(srcpath, dstpath, type):
        srcpath = resolve_path(srcpath)
        dstpath = resolve_path(dstpath)
        try:
            os.symlink(srcpath, dstpath, target_is_directory=(type == 'dir'))
        except Exception as e:
            return {"error": str(e)}

    async def readlink(p):
        p = resolve_path(p)
        try:
            return os.readlink(p)
        except Exception as e:
            return {"error": str(e)}

    svc = await server.register_service({
        "name": "AsyncFileService",
        "id": "async-file-service",
        "config": {
            "visibility": "public",
            "run_in_executor": True,
            "convert_objects": True
        },
        "diskSpace": diskSpace,
        "openFile": openFile,
        "createFile": createFile,
        "rename": rename,
        "stat": stat,
        "unlink": unlink,
        "rmdir": rmdir,
        "mkdir": mkdir,
        "readdir": readdir,
        "readdirwithstats": readdirwithstats,
        "exists": exists,
        "realpath": realpath,
        "readFile": readFile,
        "writeFile": writeFile,
        "appendFile": appendFile,
        "symlink": symlink,
        "readlink": readlink,
    })

    print("AsyncFileService is ready: " + svc.id)
    print(f"Test the service at https://hypha.aicell.io/{server.config.workspace}/services/{svc.id.split('/')[1]}")

# Create and get event loop
loop = asyncio.get_event_loop()
loop.create_task(main())
`;

try {
    // Run the Python code
    const result = await pyodide.runPythonAsync(pythonCode);
    console.log("Python code executed successfully:", result);
    
    // Keep the JavaScript process running
    await new Promise(() => {});
} catch (error) {
    console.error("Error running Python code:", error);
    // Exit the process with an error code
    Deno.exit(1);
} 
// A demo of running Ray in Docker with Deno
// run with:
// deno run --allow-run --allow-net scripts/deno-demo-ray-docker.js

async function runDockerCommand(...args) {
  const process = Deno.run({
    cmd: ["docker", ...args],
    stdout: "piped",
    stderr: "piped"
  });
  
  const [status, stdout, stderr] = await Promise.all([
    process.status(),
    process.output(),
    process.stderrOutput()
  ]);
  
  process.close();
  
  if (!status.success) {
    throw new Error(new TextDecoder().decode(stderr));
  }
  
  return new TextDecoder().decode(stdout);
}

async function main() {
  try {
    // Print Docker info
    console.log("Getting Docker info...");
    const info = await runDockerCommand("info");
    console.log(info);

    // Check if container exists and remove it
    try {
      await runDockerCommand("stop", "ray-demo");
      await runDockerCommand("rm", "ray-demo");
      console.log("\nRemoved existing Ray container");
    } catch (_) {
      // Container doesn't exist, continue
    }

    // Pull the latest Ray image
    console.log("\nPulling latest Ray image...");
    await runDockerCommand("pull", "rayproject/ray:latest");

    // Create and start Ray container
    console.log("\nStarting Ray container...");
    await runDockerCommand(
      "run",
      "-d",
      "--name", "ray-demo",
      "-p", "8265:8265",
      "-p", "6379:6379", 
      "-p", "10001:10001",
      "rayproject/ray:latest",
      "ray", "start", "--head", "--dashboard-host=0.0.0.0", "--port=6379"
    );

    console.log("\nRay container started!");
    console.log("Ray Dashboard available at: http://localhost:8265");

    // Keep the script running and handle cleanup
    console.log("\nPress Ctrl+C to stop the container and exit");
    
    Deno.addSignalListener("SIGINT", async () => {
      console.log("\nStopping and removing Ray container...");
      await runDockerCommand("stop", "ray-demo");
      await runDockerCommand("rm", "ray-demo");
      Deno.exit();
    });

  } catch (error) {
    console.error("Error:", error);
    // Try to clean up if container exists
    try {
      await runDockerCommand("stop", "ray-demo");
      await runDockerCommand("rm", "ray-demo");
    } catch (_) {
      // Ignore cleanup errors
    }
    Deno.exit(1);
  }
}

// Run the demo
await main(); 
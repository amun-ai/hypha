// Deno Security Demo
// This script demonstrates Deno's security features around command execution
// Try running with different permissions:
// 1. No permissions:         deno run scripts/deno-security-demo.js
// 2. With run permission:    deno run --allow-run scripts/deno-security-demo.js
// 3. With specific allow:    deno run --allow-run=ls scripts/deno-security-demo.js
// 4. With full path allow:   deno run --allow-run=/bin/ls scripts/deno-security-demo.js

async function testCommand(description, fn) {
  console.log(`\n📝 Testing: ${description}`);
  console.log("------------------------------------------");
  try {
    const result = await fn();
    console.log("✅ Success:", result);
  } catch (error) {
    console.log("❌ Error:", error.message);
  }
}

async function main() {
  // Test 1: Try running 'ls' using PATH resolution
  await testCommand(
    "Running 'ls' using PATH resolution",
    async () => {
      const process = Deno.run({
        cmd: ["ls"],
        stdout: "piped",
      });
      const output = await process.output();
      process.close();
      return new TextDecoder().decode(output);
    }
  );

  // Test 2: Try running 'ls' with full path
  await testCommand(
    "Running 'ls' with full path (/bin/ls)",
    async () => {
      const process = Deno.run({
        cmd: ["/bin/ls"],
        stdout: "piped",
      });
      const output = await process.output();
      process.close();
      return new TextDecoder().decode(output);
    }
  );

  // Test 3: Try running 'pwd' using PATH resolution when only 'ls' is allowed
  await testCommand(
    "Running 'pwd' using PATH resolution (should fail if only ls is allowed)",
    async () => {
      const process = Deno.run({
        cmd: ["pwd"],
        stdout: "piped",
      });
      const output = await process.output();
      process.close();
      return new TextDecoder().decode(output);
    }
  );

  // Test 4: Try running pwd with full path
  await testCommand(
    "Running 'pwd' with full path (/bin/pwd)",
    async () => {
      const process = Deno.run({
        cmd: ["/bin/pwd"],
        stdout: "piped",
      });
      const output = await process.output();
      process.close();
      return new TextDecoder().decode(output);
    }
  );

  // Test 5: Try running 'ls' through /usr/bin/env
  await testCommand(
    "Running 'ls' through /usr/bin/env (potential bypass attempt)",
    async () => {
      const process = Deno.run({
        cmd: ["/usr/bin/env", "ls"],
        stdout: "piped",
      });
      const output = await process.output();
      process.close();
      return new TextDecoder().decode(output);
    }
  );

  // Test 6: Try running 'pwd' through /usr/bin/env
  await testCommand(
    "Running 'pwd' through /usr/bin/env (potential bypass attempt)",
    async () => {
      const process = Deno.run({
        cmd: ["/usr/bin/env", "pwd"],
        stdout: "piped",
      });
      const output = await process.output();
      process.close();
      return new TextDecoder().decode(output);
    }
  );
}

// Run the demo
console.log("🔒 Deno Security Demo - Permission Boundary Testing");
console.log("=================================================");
console.log("This demo tests various command execution scenarios");
console.log("to verify permission boundaries and potential bypasses");

await main(); 
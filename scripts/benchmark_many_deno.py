import os
import psutil
import time
import subprocess
import json
import re
import asyncio
import httpx
from datetime import datetime
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin
import pandas as pd
import matplotlib.pyplot as plt

class DenoProcessManager:
    def __init__(self):
        self.processes: List[Dict] = []
        self.script_path = os.path.join(os.path.dirname(__file__), "deno-demo-asgi-app.js")
        self.url_pattern = re.compile(r"Access your app at: (https://[^\s]+)")
    
    async def start_process(self) -> Dict:
        """Start a new Deno process and return its info"""
        process = subprocess.Popen(
            ["deno", "run", "--allow-net", "--allow-read", "--allow-env", self.script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the service URL
        service_url = None
        start_time = time.time()
        while service_url is None and time.time() - start_time < 30:
            if process.poll() is not None:
                raise Exception("Process ended unexpectedly")
                
            line = process.stdout.readline()
            if line:
                match = self.url_pattern.search(line)
                if match:
                    service_url = match.group(1)
                    if not service_url.endswith('/'):
                        service_url += '/'
                    break
        
        if not service_url:
            process.terminate()
            raise Exception("Failed to get service URL within timeout")
        
        process_info = {
            "process": process,
            "pid": process.pid,
            "start_time": time.time(),
            "url": service_url
        }
        self.processes.append(process_info)
        return process_info

    async def verify_service(self, url: str) -> bool:
        """Verify if a service is running properly"""
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            try:
                # Test root endpoint
                response = await client.get(url)
                response.raise_for_status()
                
                # Test API endpoint
                api_url = urljoin(url, 'api/v1/test')
                api_response = await client.get(api_url)
                api_response.raise_for_status()
                api_data = api_response.json()
                return api_data["message"] == "Hello, it works!"
            except Exception as e:
                print(f"Service verification failed for {url}: {e}")
                return False

    async def verify_services_batch(self, start_idx: int, end_idx: int) -> Dict[str, bool]:
        """Verify a batch of services"""
        results = {}
        batch = self.processes[start_idx:end_idx]
        
        # Create tasks for all services in the batch
        tasks = []
        for proc_info in batch:
            task = asyncio.create_task(self.verify_service(proc_info["url"]))
            tasks.append((proc_info["url"], task))
        
        # Wait for all verifications to complete
        for url, task in tasks:
            results[url] = await task
            
        return results

    async def verify_all_services(self) -> Dict[str, bool]:
        """Verify all services are running"""
        results = {}
        batch_size = 10
        
        for i in range(0, len(self.processes), batch_size):
            batch_results = await self.verify_services_batch(i, min(i + batch_size, len(self.processes)))
            results.update(batch_results)
            
            working_in_batch = sum(1 for v in batch_results.values() if v)
            print(f"Batch {i//batch_size + 1}: {working_in_batch}/{len(batch_results)} services working")
            
        return results

    def collect_metrics(self) -> Dict:
        """Collect resource usage metrics for all processes"""
        total_cpu = 0
        total_memory = 0
        metrics = []
        
        for proc_info in self.processes:
            try:
                process = psutil.Process(proc_info["pid"])
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                
                metric = {
                    "pid": proc_info["pid"],
                    "url": proc_info["url"],
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_info.rss / 1024 / 1024,
                    "runtime_seconds": time.time() - proc_info["start_time"]
                }
                metrics.append(metric)
                
                total_cpu += cpu_percent
                total_memory += memory_info.rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        system = psutil.virtual_memory()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "num_processes": len(self.processes),
            "total_cpu_percent": total_cpu,
            "total_memory_mb": total_memory / 1024 / 1024,
            "system_memory_percent": system.percent,
            "system_available_memory_mb": system.available / 1024 / 1024,
            "per_process": metrics
        }

    def cleanup(self):
        """Terminate all processes"""
        for proc_info in self.processes:
            try:
                process = psutil.Process(proc_info["pid"])
                process.terminate()
            except psutil.NoSuchProcess:
                continue

class StressTest:
    def __init__(self, start_processes: int = 200, max_processes: int = 250, increment: int = 10):
        self.process_manager = DenoProcessManager()
        self.start_processes = start_processes
        self.max_processes = max_processes
        self.increment = increment
        self.metrics = []
        self.current_processes = 0
        self.verification_results = []

    async def run(self):
        """Run the stress test"""
        try:
            print(f"Starting stress test with initial {self.start_processes} processes...")
            print(f"Will increment by {self.increment} until reaching {self.max_processes} processes")
            
            while self.current_processes < self.max_processes:
                target_processes = min(
                    self.current_processes + (self.start_processes if self.current_processes == 0 else self.increment),
                    self.max_processes
                )
                
                print(f"\nStarting processes {self.current_processes + 1} to {target_processes}...")
                
                # Start new processes in batches of 10
                for batch_start in range(self.current_processes, target_processes, 10):
                    batch_end = min(batch_start + 10, target_processes)
                    print(f"\nStarting batch {batch_start + 1} to {batch_end}...")
                    
                    # Start processes in this batch
                    new_processes = []
                    for i in range(batch_start, batch_end):
                        try:
                            process_info = await self.process_manager.start_process()
                            new_processes.append(process_info)
                            self.current_processes += 1
                            print(f"Started process {self.current_processes}/{target_processes}", end='\r')
                        except Exception as e:
                            print(f"\nFailed to start process: {e}")
                            return
                    
                    print(f"\nSuccessfully started {len(new_processes)} new processes")
                    
                    # Verify this batch of services
                    print("\nVerifying new batch of services...")
                    verification = await self.process_manager.verify_services_batch(batch_start, batch_end)
                    working_services = sum(1 for v in verification.values() if v)
                    print(f"Services working in this batch: {working_services}/{len(verification)}")
                    
                    # If less than 80% of services in the batch are working, stop the test
                    if working_services / len(verification) < 0.8:
                        print("\nToo many services failed in this batch. Stopping test.")
                        return
                    
                    # Collect metrics after each batch
                    metrics = self.process_manager.collect_metrics()
                    self.metrics.append(metrics)
                    
                    print(f"\nCurrent system metrics:")
                    print(f"- Total processes: {metrics['num_processes']}")
                    print(f"- Memory usage: {metrics['system_memory_percent']:.1f}%")
                    print(f"- Available memory: {metrics['system_available_memory_mb']:.0f} MB")
                    print(f"- Average memory per process: {metrics['total_memory_mb']/self.current_processes:.1f} MB")
                
                # Verify all services after completing each major increment
                print("\nVerifying all services...")
                verification = await self.process_manager.verify_all_services()
                self.verification_results.append({
                    "timestamp": datetime.now().isoformat(),
                    "num_processes": self.current_processes,
                    "results": verification
                })
                
                working_services = sum(1 for v in verification.values() if v)
                print(f"Total services working: {working_services}/{len(verification)}")
                
                # If less than 80% of total services are working, stop the test
                if working_services / len(verification) < 0.8:
                    print("\nToo many services failed overall. Stopping test.")
                    return
                
                # Wait before next increment
                if self.current_processes < self.max_processes:
                    print(f"\nWaiting 10 seconds before next increment...")
                    await asyncio.sleep(10)
            
            # Final verification after reaching max processes
            print("\nFinal service verification...")
            final_verification = await self.process_manager.verify_all_services()
            self.verification_results.append({
                "timestamp": datetime.now().isoformat(),
                "num_processes": self.current_processes,
                "results": final_verification
            })
            
            working_services = sum(1 for v in final_verification.values() if v)
            print(f"Final services working: {working_services}/{len(final_verification)}")
                
        finally:
            self.process_manager.cleanup()
    
    def generate_report(self):
        """Generate a report from collected metrics"""
        if not self.metrics:
            print("No metrics collected. Cannot generate report.")
            return
            
        # Convert metrics to DataFrame
        df = pd.DataFrame([{
            'timestamp': m['timestamp'],
            'num_processes': m['num_processes'],
            'total_cpu_percent': m['total_cpu_percent'],
            'total_memory_mb': m['total_memory_mb'],
            'system_memory_percent': m['system_memory_percent'],
            'system_available_memory_mb': m['system_available_memory_mb'],
            'avg_memory_per_process': m['total_memory_mb'] / m['num_processes'] if m['num_processes'] > 0 else 0
        } for m in self.metrics])
        
        # Create verification DataFrame
        df_verification = pd.DataFrame([{
            'timestamp': v['timestamp'],
            'num_processes': v['num_processes'],
            'working_services': sum(1 for r in v['results'].values() if r),
            'total_services': len(v['results']),
            'success_rate': sum(1 for r in v['results'].values() if r) / len(v['results']) if v['results'] else 0
        } for v in self.verification_results])
        
        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot system memory usage
        ax1.plot(df['num_processes'], df['system_memory_percent'], label='System Memory %')
        ax1.set_xlabel('Number of Processes')
        ax1.set_ylabel('System Memory %')
        ax1.legend()
        ax1.set_title('System Memory Usage vs Number of Processes')
        ax1.grid(True)
        
        # Plot average memory per process
        ax2.plot(df['num_processes'], df['avg_memory_per_process'], label='Avg Memory per Process (MB)')
        ax2.set_xlabel('Number of Processes')
        ax2.set_ylabel('Memory (MB)')
        ax2.legend()
        ax2.set_title('Average Memory per Process')
        ax2.grid(True)
        
        # Plot service success rate
        ax3.plot(df_verification['num_processes'], df_verification['success_rate'] * 100, label='Service Success Rate %')
        ax3.set_xlabel('Number of Processes')
        ax3.set_ylabel('Success Rate %')
        ax3.legend()
        ax3.set_title('Service Success Rate vs Number of Processes')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png')
        
        # Save raw data
        df.to_csv('benchmark_metrics.csv', index=False)
        df_verification.to_csv('benchmark_verification.csv', index=False)
        
        # Print summary statistics
        print("\nBenchmark Summary:")
        print(f"Maximum number of processes: {self.current_processes}")
        print(f"Final system memory usage: {df['system_memory_percent'].iloc[-1]:.1f}%")
        print(f"Average memory per process: {df['avg_memory_per_process'].mean():.1f} MB")
        print(f"Final available system memory: {df['system_available_memory_mb'].iloc[-1]:.0f} MB")
        print("\nService Verification Summary:")
        print(f"Final success rate: {df_verification['success_rate'].iloc[-1]*100:.1f}%")
        print(f"Minimum success rate: {df_verification['success_rate'].min()*100:.1f}%")
        print(f"Average success rate: {df_verification['success_rate'].mean()*100:.1f}%")

async def main():
    # Create and run stress test
    stress_test = StressTest(
        start_processes=200,  # Start with 200 processes
        max_processes=250,    # Go up to 250 processes
        increment=10          # Add 10 processes at a time
    )
    await stress_test.run()
    stress_test.generate_report()

if __name__ == "__main__":
    asyncio.run(main()) 
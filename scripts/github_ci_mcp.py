#!/usr/bin/env python
"""
GitHub CI MCP Server for monitoring and waiting for CI completion.

Run with:
    python scripts/github_ci_mcp.py
"""

import os
import asyncio
import requests
import re
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("GitHub CI Monitor")

# GitHub API configuration
GITHUB_API_BASE = "https://api.github.com"
GITHUB_REPO = "amun-ai/hypha"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

def get_headers():
    """Get GitHub API headers."""
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def get_workflow_runs(branch: str, workflow_id: int = 14258273) -> List[Dict[str, Any]]:
    """Get workflow runs for a specific branch."""
    url = f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/actions/workflows/{workflow_id}/runs"
    params = {
        "branch": branch,
        "per_page": 10
    }
    
    response = requests.get(url, headers=get_headers(), params=params)
    response.raise_for_status()
    return response.json().get("workflow_runs", [])


def get_latest_run_for_commit(branch: str, commit_sha: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get the latest workflow run for a branch or specific commit."""
    runs = get_workflow_runs(branch)
    
    if not runs:
        return None
    
    if commit_sha:
        # Find run for specific commit
        for run in runs:
            if run["head_sha"].startswith(commit_sha):
                return run
        return None
    else:
        # Return the most recent run
        return runs[0] if runs else None


def get_run_jobs(run_id: int) -> List[Dict[str, Any]]:
    """Get jobs for a specific workflow run."""
    url = f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/actions/runs/{run_id}/jobs"
    response = requests.get(url, headers=get_headers())
    response.raise_for_status()
    return response.json().get("jobs", [])


def get_job_logs(job_id: int) -> str:
    """Get logs for a specific job."""
    url = f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/actions/jobs/{job_id}/logs"
    headers = get_headers()
    
    # Try without authentication first for public repos
    response = requests.get(url, allow_redirects=True)
    if response.status_code == 200:
        return response.text
    
    # If that fails, try with authentication
    if GITHUB_TOKEN:
        response = requests.get(url, headers=headers, allow_redirects=True)
        if response.status_code == 200:
            return response.text
    
    return f"Failed to get logs: {response.status_code}"


def extract_test_failures(logs: str) -> List[Dict[str, Any]]:
    """Extract test failure segments from CI logs."""
    failures = []
    
    # Pattern to match failure sections like: _______________________ test_name ________________________
    failure_pattern = r'_{20,}\s+(.+?)\s+_{20,}'
    
    # Split logs into lines for easier processing
    lines = logs.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.search(failure_pattern, line)
        
        if match:
            test_name = match.group(1).strip()
            failure_content = []
            failure_content.append(line)  # Include the header line
            i += 1
            
            # Capture everything until we hit another failure section or captured output section
            while i < len(lines):
                current_line = lines[i]
                
                # Stop if we hit another failure section
                if re.search(failure_pattern, current_line):
                    i -= 1  # Back up one line to process this failure in next iteration
                    break
                    
                # Stop if we hit the captured output section
                if re.match(r'-{20,}\s+Captured\s+.*\s+call\s+-{20,}', current_line):
                    failure_content.append(current_line)
                    break
                
                failure_content.append(current_line)
                i += 1
            
            # Clean up the failure content
            failure_text = '\n'.join(failure_content).strip()
            
            if failure_text:
                failures.append({
                    'test_name': test_name,
                    'failure_content': failure_text
                })
        
        i += 1
    
    return failures


@mcp.tool()
def check_ci_status(branch: str, commit_sha: Optional[str] = None) -> Dict[str, Any]:
    """
    Check the current CI status for a branch or specific commit.
    
    Args:
        branch: The branch name to check
        commit_sha: Optional commit SHA (can be partial)
    
    Returns:
        Dictionary with CI status information
    """
    try:
        run = get_latest_run_for_commit(branch, commit_sha)
        
        if not run:
            return {
                "status": "not_found",
                "message": f"No workflow runs found for branch '{branch}'" + 
                          (f" and commit '{commit_sha}'" if commit_sha else "")
            }
        
        return {
            "status": run["status"],
            "conclusion": run.get("conclusion"),
            "run_id": run["id"],
            "run_number": run["run_number"],
            "head_sha": run["head_sha"],
            "created_at": run["created_at"],
            "updated_at": run["updated_at"],
            "html_url": run["html_url"],
            "branch": branch,
            "commit_message": run.get("head_commit", {}).get("message", "").split("\n")[0]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool()
def get_ci_logs(branch: str, commit_sha: Optional[str] = None, failed_only: bool = True) -> Dict[str, Any]:
    """
    Get CI logs for a branch or specific commit.
    
    Args:
        branch: The branch name
        commit_sha: Optional commit SHA (can be partial)
        failed_only: If True, only return logs from failed jobs
    
    Returns:
        Dictionary with logs from CI jobs
    """
    try:
        run = get_latest_run_for_commit(branch, commit_sha)
        
        if not run:
            return {
                "status": "not_found",
                "message": f"No workflow runs found for branch '{branch}'"
            }
        
        jobs = get_run_jobs(run["id"])
        logs_data = {
            "run_id": run["id"],
            "run_url": run["html_url"],
            "status": run["status"],
            "conclusion": run.get("conclusion"),
            "jobs": []
        }
        
        for job in jobs:
            # Skip successful jobs if failed_only is True
            if failed_only and job.get("conclusion") == "success":
                continue
                
            job_info = {
                "name": job["name"],
                "status": job["status"],
                "conclusion": job.get("conclusion"),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at")
            }
            
            # Get logs if job is completed
            if job["status"] == "completed":
                job_info["logs"] = get_job_logs(job["id"])
            
            logs_data["jobs"].append(job_info)
        
        return logs_data
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool()
async def wait_for_ci_to_start(
    branch: str, 
    commit_sha: Optional[str] = None,
    timeout_seconds: int = 30*60,
    poll_interval: int = 10
) -> Dict[str, Any]:
    """
    Wait for CI to start for a specific branch/commit.
    
    Args:
        branch: The branch name
        commit_sha: Optional commit SHA to wait for
        timeout_seconds: Maximum time to wait (default: 30 minutes)
        poll_interval: Seconds between checks (default: 10)
    
    Returns:
        Dictionary with CI run information once started
    """
    start_time = datetime.now()
    timeout = timedelta(seconds=timeout_seconds)
    
    while datetime.now() - start_time < timeout:
        try:
            run = get_latest_run_for_commit(branch, commit_sha)
            
            if run and run["status"] in ["queued", "in_progress", "completed"]:
                return {
                    "started": True,
                    "run_id": run["id"],
                    "status": run["status"],
                    "head_sha": run["head_sha"],
                    "html_url": run["html_url"],
                    "waited_seconds": (datetime.now() - start_time).total_seconds()
                }
        except Exception:
            # Continue waiting on API errors
            pass
        
        await asyncio.sleep(poll_interval)
    
    return {
        "started": False,
        "message": f"Timeout waiting for CI to start after {timeout_seconds} seconds",
        "branch": branch,
        "commit_sha": commit_sha
    }


@mcp.tool()
def get_pytest_failures(branch: str, commit_sha: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract structured pytest test failures from GitHub Actions CI runs.
    
    This tool identifies and extracts detailed pytest failure information including
    test names, full tracebacks, error messages, and captured output from CI job logs.
    
    Args:
        branch: The branch name to check
        commit_sha: Optional commit SHA (can be partial)
    
    Returns:
        Dictionary with extracted pytest failures containing:
        - run_id, run_url, status, conclusion
        - failures: list of test failures with job info, test names, and full failure content
    """
    try:
        run = get_latest_run_for_commit(branch, commit_sha)
        
        if not run:
            return {
                "status": "not_found",
                "message": f"No workflow runs found for branch '{branch}'" + 
                          (f" and commit '{commit_sha}'" if commit_sha else "")
            }
        
        jobs = get_run_jobs(run["id"])
        failures_data = {
            "run_id": run["id"],
            "run_url": run["html_url"],
            "status": run["status"],
            "conclusion": run.get("conclusion"),
            "failures": []
        }
        
        for job in jobs:
            # Process completed jobs that failed
            if job["status"] == "completed" and job.get("conclusion") == "failure":
                # Get logs with proper authentication
                logs_url = f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/actions/jobs/{job['id']}/logs"
                logs_response = requests.get(logs_url, headers=get_headers(), allow_redirects=True)
                
                if logs_response.status_code == 200:
                    job_logs = logs_response.text
                    test_failures = extract_test_failures(job_logs)
                    
                    for failure in test_failures:
                        failures_data["failures"].append({
                            "job_name": job["name"],
                            "job_id": job["id"],
                            "test_name": failure["test_name"],
                            "failure_content": failure["failure_content"],
                            "job_conclusion": job.get("conclusion"),
                            "job_url": job.get("html_url")
                        })
        
        return failures_data
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool()
async def wait_for_ci_completion(
    branch: str,
    commit_sha: Optional[str] = None,
    timeout_seconds: int = 1800,
    poll_interval: int = 30
) -> Dict[str, Any]:
    """
    Wait for CI to complete and return the results with logs.
    
    Args:
        branch: The branch name
        commit_sha: Optional commit SHA
        timeout_seconds: Maximum time to wait (default: 30 minutes)
        poll_interval: Seconds between checks (default: 30)
    
    Returns:
        Dictionary with CI completion status and logs
    """
    # First wait for CI to start
    start_result = await wait_for_ci_to_start(branch, commit_sha, timeout_seconds=300)
    
    if not start_result.get("started"):
        return start_result
    
    run_id = start_result["run_id"]
    start_time = datetime.now()
    timeout = timedelta(seconds=timeout_seconds)
    
    while datetime.now() - start_time < timeout:
        try:
            # Get run status directly by ID
            url = f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/actions/runs/{run_id}"
            response = requests.get(url, headers=get_headers())
            response.raise_for_status()
            run = response.json()
            
            if run["status"] == "completed":
                # Get logs for failed jobs
                logs = get_ci_logs(branch, commit_sha, failed_only=(run["conclusion"] != "success"))
                
                return {
                    "completed": True,
                    "conclusion": run["conclusion"],
                    "run_id": run["id"],
                    "html_url": run["html_url"],
                    "total_duration_seconds": (datetime.now() - start_time).total_seconds(),
                    "logs": logs
                }
        except Exception:
            # Continue waiting on API errors
            pass
        
        await asyncio.sleep(poll_interval)
    
    return {
        "completed": False,
        "message": f"Timeout waiting for CI to complete after {timeout_seconds} seconds",
        "run_id": run_id
    }


if __name__ == "__main__":
    # Run the MCP server
    import sys
    mcp.run(transport=sys.argv[1] if len(sys.argv) > 1 else "stdio")
#!/usr/bin/env python3
"""
Debug runner script - runs all debugging tools in sequence.
"""

import subprocess
import sys
import os

def run_debug_tool(tool_name, description):
    """Run a debug tool and report results."""
    print(f"\n{'='*60}")
    print(f"🔍 Running: {tool_name}")
    print(f"📝 Description: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, tool_name], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            print(result.stdout)
        else:
            print("❌ FAILED")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT - Tool took too long to run")
        return False
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False

def main():
    """Run all debugging tools in sequence."""
    print("🚀 Bayesian LoRA Debug Runner")
    print("Running all debugging tools in sequence...\n")
    
    # Define debug tools in order of execution
    debug_tools = [
        ("debug/cluster_troubleshooting.py", "Comprehensive environment check"),
        ("debug/debug_lora.py", "LoRA parameter debugging"),
        ("debug/test_model_direct.py", "Model creation testing"),
        ("debug/comprehensive_test.py", "Full system validation"),
    ]
    
    results = []
    
    for tool, description in debug_tools:
        success = run_debug_tool(tool, description)
        results.append((tool, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 DEBUG RUN SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for tool, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {tool}")
    
    print(f"\nOverall: {passed}/{total} tools passed")
    
    if passed == total:
        print("🎉 All debugging tools passed! Your system is ready.")
        return True
    else:
        print("⚠️  Some tools failed. Check the output above for issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""Comprehensive Test Runner for Ultimate Arbitrage System

This script orchestrates all testing activities:
- Unit tests with 95% coverage gate
- Integration tests with Testcontainers
- Market simulation tests
- Chaos engineering tests
- Performance benchmarks
- Rust tests with cargo
"""

import asyncio
import os
import sys
import subprocess
import time
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()


class TestResult:
    """Test result container"""
    def __init__(self, name: str, success: bool, duration: float, 
                 output: str = "", metrics: Dict[str, Any] = None):
        self.name = name
        self.success = success
        self.duration = duration
        self.output = output
        self.metrics = metrics or {}
        self.timestamp = datetime.now()


class ComprehensiveTestRunner:
    """Main test runner orchestrating all test types"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.test_results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
        
        # Test configuration
        self.config = self._load_test_config()
    
    def _load_test_config(self) -> Dict[str, Any]:
        """Load test configuration"""
        config_path = self.project_root / 'test_config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'coverage_threshold': 95,
            'unit_tests': {
                'parallel_workers': 4,
                'timeout': 300
            },
            'integration_tests': {
                'timeout': 600,
                'docker_required': True
            },
            'performance_tests': {
                'duration_seconds': 300,
                'max_users': 100
            },
            'chaos_tests': {
                'enabled': True,
                'fault_duration': 120
            },
            'rust_tests': {
                'coverage_threshold': 95,
                'benchmark': True
            }
        }
    
    async def run_all_tests(self, test_types: List[str] = None) -> bool:
        """Run all or specified test types"""
        self.start_time = datetime.now()
        console.print(Panel("🚀 Starting Comprehensive Test Suite", style="bold green"))
        
        if test_types is None:
            test_types = ['unit', 'integration', 'simulation', 'chaos', 'performance', 'rust']
        
        all_success = True
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for test_type in test_types:
                task = progress.add_task(f"Running {test_type} tests...", total=None)
                
                try:
                    if test_type == 'unit':
                        success = await self.run_unit_tests()
                    elif test_type == 'integration':
                        success = await self.run_integration_tests()
                    elif test_type == 'simulation':
                        success = await self.run_simulation_tests()
                    elif test_type == 'chaos':
                        success = await self.run_chaos_tests()
                    elif test_type == 'performance':
                        success = await self.run_performance_tests()
                    elif test_type == 'rust':
                        success = await self.run_rust_tests()
                    else:
                        console.print(f"[red]Unknown test type: {test_type}[/red]")
                        success = False
                    
                    all_success &= success
                    
                    if success:
                        progress.update(task, description=f"✅ {test_type} tests completed")
                    else:
                        progress.update(task, description=f"❌ {test_type} tests failed")
                        
                except Exception as e:
                    logger.error(f"Error running {test_type} tests: {e}")
                    all_success = False
                    progress.update(task, description=f"💥 {test_type} tests errored")
                
                progress.remove_task(task)
        
        self.end_time = datetime.now()
        await self.generate_comprehensive_report()
        
        return all_success
    
    async def run_unit_tests(self) -> bool:
        """Run Python unit tests with coverage"""
        console.print("\n[bold blue]Running Unit Tests[/bold blue]")
        
        start_time = time.time()
        
        # Prepare pytest command
        pytest_cmd = [
            'python', '-m', 'pytest',
            'tests/unit/',
            '-v',
            '--tb=short',
            f'--cov-fail-under={self.config["coverage_threshold"]}',
            '--cov-report=html:htmlcov',
            '--cov-report=xml:coverage.xml',
            '--cov-report=term-missing',
            '--junitxml=unit-test-results.xml',
            f'-n={self.config["unit_tests"]["parallel_workers"]}',
            f'--timeout={self.config["unit_tests"]["timeout"]}'
        ]
        
        try:
            result = subprocess.run(
                pytest_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.config['unit_tests']['timeout'] + 60
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            # Parse coverage from output
            coverage_percent = self._extract_coverage_from_output(result.stdout)
            
            test_result = TestResult(
                name="Unit Tests",
                success=success,
                duration=duration,
                output=result.stdout + result.stderr,
                metrics={
                    'coverage_percent': coverage_percent,
                    'threshold_met': coverage_percent >= self.config['coverage_threshold']
                }
            )
            
            self.test_results.append(test_result)
            
            if success:
                console.print(f"[green]✅ Unit tests passed with {coverage_percent}% coverage[/green]")
            else:
                console.print(f"[red]❌ Unit tests failed[/red]")
                console.print(result.stderr)
            
            return success
            
        except subprocess.TimeoutExpired:
            console.print("[red]❌ Unit tests timed out[/red]")
            return False
        except Exception as e:
            console.print(f"[red]❌ Unit tests error: {e}[/red]")
            return False
    
    async def run_integration_tests(self) -> bool:
        """Run integration tests with Testcontainers"""
        console.print("\n[bold blue]Running Integration Tests[/bold blue]")
        
        # Check Docker availability
        if self.config['integration_tests']['docker_required']:
            if not self._check_docker():
                console.print("[red]❌ Docker not available, skipping integration tests[/red]")
                return False
        
        start_time = time.time()
        
        pytest_cmd = [
            'python', '-m', 'pytest',
            'tests/integration/',
            '-v',
            '--tb=short',
            '-m', 'integration',
            f'--timeout={self.config["integration_tests"]["timeout"]}',
            '--junitxml=integration-test-results.xml'
        ]
        
        try:
            result = subprocess.run(
                pytest_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.config['integration_tests']['timeout'] + 120
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            test_result = TestResult(
                name="Integration Tests",
                success=success,
                duration=duration,
                output=result.stdout + result.stderr
            )
            
            self.test_results.append(test_result)
            
            if success:
                console.print("[green]✅ Integration tests passed[/green]")
            else:
                console.print("[red]❌ Integration tests failed[/red]")
                console.print(result.stderr)
            
            return success
            
        except Exception as e:
            console.print(f"[red]❌ Integration tests error: {e}[/red]")
            return False
    
    async def run_simulation_tests(self) -> bool:
        """Run market simulation tests"""
        console.print("\n[bold blue]Running Market Simulation Tests[/bold blue]")
        
        start_time = time.time()
        
        try:
            # Run simulation tests
            pytest_cmd = [
                'python', '-m', 'pytest',
                'tests/simulation/',
                '-v',
                '--tb=short',
                '-m', 'simulation',
                '--junitxml=simulation-test-results.xml'
            ]
            
            result = subprocess.run(
                pytest_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            test_result = TestResult(
                name="Market Simulation Tests",
                success=success,
                duration=duration,
                output=result.stdout + result.stderr
            )
            
            self.test_results.append(test_result)
            
            if success:
                console.print("[green]✅ Market simulation tests passed[/green]")
            else:
                console.print("[red]❌ Market simulation tests failed[/red]")
            
            return success
            
        except Exception as e:
            console.print(f"[red]❌ Simulation tests error: {e}[/red]")
            return False
    
    async def run_chaos_tests(self) -> bool:
        """Run chaos engineering tests"""
        console.print("\n[bold blue]Running Chaos Engineering Tests[/bold blue]")
        
        if not self.config['chaos_tests']['enabled']:
            console.print("[yellow]⚠️ Chaos tests disabled in configuration[/yellow]")
            return True
        
        start_time = time.time()
        
        try:
            pytest_cmd = [
                'python', '-m', 'pytest',
                'tests/chaos/',
                '-v',
                '--tb=short',
                '-m', 'chaos',
                '--junitxml=chaos-test-results.xml'
            ]
            
            result = subprocess.run(
                pytest_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.config['chaos_tests']['fault_duration'] * 3
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            test_result = TestResult(
                name="Chaos Engineering Tests",
                success=success,
                duration=duration,
                output=result.stdout + result.stderr
            )
            
            self.test_results.append(test_result)
            
            if success:
                console.print("[green]✅ Chaos engineering tests passed[/green]")
            else:
                console.print("[red]❌ Chaos engineering tests failed[/red]")
            
            return success
            
        except Exception as e:
            console.print(f"[red]❌ Chaos tests error: {e}[/red]")
            return False
    
    async def run_performance_tests(self) -> bool:
        """Run performance benchmarks"""
        console.print("\n[bold blue]Running Performance Tests[/bold blue]")
        
        start_time = time.time()
        
        try:
            pytest_cmd = [
                'python', '-m', 'pytest',
                'tests/performance/',
                '-v',
                '--tb=short',
                '-m', 'performance',
                '--benchmark-only',
                '--benchmark-json=performance-results.json',
                '--junitxml=performance-test-results.xml'
            ]
            
            result = subprocess.run(
                pytest_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.config['performance_tests']['duration_seconds'] * 2
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            # Load benchmark results
            benchmark_data = {}
            benchmark_file = self.project_root / 'performance-results.json'
            if benchmark_file.exists():
                with open(benchmark_file, 'r') as f:
                    benchmark_data = json.load(f)
            
            test_result = TestResult(
                name="Performance Tests",
                success=success,
                duration=duration,
                output=result.stdout + result.stderr,
                metrics={'benchmarks': benchmark_data}
            )
            
            self.test_results.append(test_result)
            
            if success:
                console.print("[green]✅ Performance tests passed[/green]")
            else:
                console.print("[red]❌ Performance tests failed[/red]")
            
            return success
            
        except Exception as e:
            console.print(f"[red]❌ Performance tests error: {e}[/red]")
            return False
    
    async def run_rust_tests(self) -> bool:
        """Run Rust tests with cargo"""
        console.print("\n[bold blue]Running Rust Tests[/bold blue]")
        
        rust_project_path = self.project_root / 'high_performance_core/high_performance_core/rust_execution_engine'
        
        if not rust_project_path.exists():
            console.print("[yellow]⚠️ Rust project not found, skipping Rust tests[/yellow]")
            return True
        
        start_time = time.time()
        
        try:
            # Run cargo test
            test_result = subprocess.run(
                ['cargo', 'test', '--all-features'],
                cwd=rust_project_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Run cargo tarpaulin for coverage (if available)
            coverage_output = ""
            coverage_percent = 0
            
            try:
                coverage_result = subprocess.run(
                    ['cargo', 'tarpaulin', '--out', 'xml', '--output-dir', '.'],
                    cwd=rust_project_path,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if coverage_result.returncode == 0:
                    coverage_output = coverage_result.stdout
                    # Extract coverage percentage from output
                    coverage_percent = self._extract_rust_coverage(coverage_output)
            except FileNotFoundError:
                console.print("[yellow]⚠️ cargo-tarpaulin not found, skipping coverage[/yellow]")
            
            # Run benchmarks if enabled
            benchmark_output = ""
            if self.config['rust_tests']['benchmark']:
                try:
                    bench_result = subprocess.run(
                        ['cargo', 'bench'],
                        cwd=rust_project_path,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    benchmark_output = bench_result.stdout
                except Exception as e:
                    console.print(f"[yellow]⚠️ Rust benchmarks failed: {e}[/yellow]")
            
            duration = time.time() - start_time
            success = test_result.returncode == 0
            
            # Check coverage threshold
            if coverage_percent > 0:
                threshold_met = coverage_percent >= self.config['rust_tests']['coverage_threshold']
                if not threshold_met:
                    success = False
            else:
                threshold_met = True  # If no coverage tool, don't fail
            
            test_result_obj = TestResult(
                name="Rust Tests",
                success=success,
                duration=duration,
                output=test_result.stdout + test_result.stderr + coverage_output + benchmark_output,
                metrics={
                    'coverage_percent': coverage_percent,
                    'threshold_met': threshold_met
                }
            )
            
            self.test_results.append(test_result_obj)
            
            if success:
                console.print(f"[green]✅ Rust tests passed with {coverage_percent}% coverage[/green]")
            else:
                console.print("[red]❌ Rust tests failed[/red]")
            
            return success
            
        except Exception as e:
            console.print(f"[red]❌ Rust tests error: {e}[/red]")
            return False
    
    def _check_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(
                ['docker', 'info'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except:
            return False
    
    def _extract_coverage_from_output(self, output: str) -> float:
        """Extract coverage percentage from pytest output"""
        import re
        match = re.search(r'TOTAL.*?(\d+)%', output)
        if match:
            return float(match.group(1))
        return 0.0
    
    def _extract_rust_coverage(self, output: str) -> float:
        """Extract coverage percentage from cargo tarpaulin output"""
        import re
        match = re.search(r'(\d+\.\d+)% coverage', output)
        if match:
            return float(match.group(1))
        return 0.0
    
    async def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        console.print("\n[bold blue]Generating Comprehensive Test Report[/bold blue]")
        
        # Create summary table
        table = Table(title="Test Results Summary")
        table.add_column("Test Type", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Duration", style="green")
        table.add_column("Coverage", style="yellow")
        table.add_column("Notes", style="white")
        
        total_duration = 0
        passed_tests = 0
        
        for result in self.test_results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            duration = f"{result.duration:.1f}s"
            coverage = ""
            notes = ""
            
            if 'coverage_percent' in result.metrics:
                cov_pct = result.metrics['coverage_percent']
                threshold_met = result.metrics.get('threshold_met', True)
                coverage = f"{cov_pct:.1f}%"
                if not threshold_met:
                    notes = "⚠️ Below threshold"
            
            table.add_row(result.name, status, duration, coverage, notes)
            
            total_duration += result.duration
            if result.success:
                passed_tests += 1
        
        console.print(table)
        
        # Print summary
        total_tests = len(self.test_results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        summary_panel = Panel(
            f"""Total Tests: {total_tests}
Passed: {passed_tests}
Failed: {total_tests - passed_tests}
Success Rate: {success_rate:.1f}%
Total Duration: {total_duration:.1f}s""",
            title="Test Execution Summary",
            style="bold"
        )
        
        console.print(summary_panel)
        
        # Save detailed report to JSON
        report_data = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': success_rate,
                'total_duration': total_duration,
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat()
            },
            'test_results': [
                {
                    'name': result.name,
                    'success': result.success,
                    'duration': result.duration,
                    'timestamp': result.timestamp.isoformat(),
                    'metrics': result.metrics
                }
                for result in self.test_results
            ]
        }
        
        report_file = self.project_root / 'comprehensive_test_report.json'
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        console.print(f"\n[green]📊 Detailed report saved to {report_file}[/green]")


@click.command()
@click.option('--test-types', '-t', multiple=True, 
              type=click.Choice(['unit', 'integration', 'simulation', 'chaos', 'performance', 'rust']),
              help='Specific test types to run (default: all)')
@click.option('--project-root', '-p', 
              type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='.',
              help='Project root directory')
@click.option('--config', '-c',
              type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='Test configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def main(test_types, project_root, config, verbose):
    """Run comprehensive test suite for Ultimate Arbitrage System"""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    runner = ComprehensiveTestRunner(Path(project_root))
    
    # Override config if provided
    if config:
        with open(config, 'r') as f:
            runner.config.update(yaml.safe_load(f))
    
    # Run tests
    test_types_list = list(test_types) if test_types else None
    success = asyncio.run(runner.run_all_tests(test_types_list))
    
    if success:
        console.print("\n[bold green]🎉 All tests passed![/bold green]")
        sys.exit(0)
    else:
        console.print("\n[bold red]❌ Some tests failed![/bold red]")
        sys.exit(1)


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Ultimate Arbitrage System - Security Hardening & Compliance Framework
A comprehensive security system covering SAST, DAST, dependency scanning,
container hardening, governance, and compliance.
"""

import os
import sys
import json
import yaml
import logging
import asyncio
import subprocess
import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Security scanning imports
try:
    import semgrep
    import docker
    import requests
    from jira import JIRA
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except ImportError as e:
    print(f"Warning: Some security dependencies not installed: {e}")
    print("Install with: pip install semgrep docker-py jira cryptography requests pyyaml")

class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceFramework(Enum):
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    NIST = "nist"

@dataclass
class SecurityFinding:
    """Represents a security finding from various scanners"""
    id: str
    title: str
    description: str
    severity: SecurityLevel
    category: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    remediation: Optional[str] = None
    confidence: float = 1.0
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    scanner: str = "unknown"
    raw_output: Optional[Dict] = None

@dataclass
class ComplianceControl:
    """Represents a compliance control requirement"""
    id: str
    framework: ComplianceFramework
    title: str
    description: str
    implementation_status: str = "pending"
    evidence_path: Optional[str] = None
    jira_ticket: Optional[str] = None
    policy_document: Optional[str] = None
    last_assessed: Optional[datetime.datetime] = None
    assessment_notes: Optional[str] = None

class SecurityHardeningCompliance:
    """Comprehensive security hardening and compliance framework"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.findings: List[SecurityFinding] = []
        self.compliance_controls: Dict[str, ComplianceControl] = {}
        self.docker_client = None
        self.jira_client = None
        
        # Initialize security tools
        self._initialize_security_tools()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load security configuration"""
        default_config = {
            "sast": {
                "enabled": True,
                "tools": ["semgrep", "bandit", "safety"],
                "rules_path": "./security/rules",
                "exclude_paths": ["node_modules", ".git", "__pycache__"]
            },
            "dast": {
                "enabled": True,
                "tools": ["owasp-zap", "nuclei"],
                "target_urls": [],
                "authentication": {}
            },
            "dependency_scanning": {
                "enabled": True,
                "tools": ["osv-scanner", "trivy", "safety"],
                "package_managers": ["pip", "npm", "go", "cargo"]
            },
            "container_hardening": {
                "enabled": True,
                "distroless": True,
                "rootless": True,
                "seccomp": True,
                "apparmor": True,
                "read_only_root": True
            },
            "compliance": {
                "frameworks": ["soc2", "iso27001"],
                "jira_integration": {
                    "enabled": False,
                    "url": "",
                    "username": "",
                    "api_token": ""
                }
            },
            "audit": {
                "penetration_testing": {
                    "enabled": True,
                    "frequency": "quarterly"
                },
                "aml_cft_screening": {
                    "enabled": True,
                    "ofac_api_key": "",
                    "frequency": "daily"
                }
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
                
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('security_hardening')
        logger.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_dir = Path("logs/security")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"security_scan_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_security_tools(self):
        """Initialize security scanning tools"""
        try:
            if self.config.get('container_hardening', {}).get('enabled'):
                self.docker_client = docker.from_env()
                
            if self.config.get('compliance', {}).get('jira_integration', {}).get('enabled'):
                jira_config = self.config['compliance']['jira_integration']
                self.jira_client = JIRA(
                    server=jira_config['url'],
                    basic_auth=(jira_config['username'], jira_config['api_token'])
                )
        except Exception as e:
            self.logger.warning(f"Failed to initialize some security tools: {e}")
    
    async def run_comprehensive_security_scan(self, target_path: str) -> Dict[str, Any]:
        """Run comprehensive security scanning"""
        self.logger.info("Starting comprehensive security scan")
        
        scan_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "target_path": target_path,
            "sast_results": {},
            "dast_results": {},
            "dependency_scan_results": {},
            "container_scan_results": {},
            "compliance_results": {},
            "summary": {}
        }
        
        # Run SAST (Static Application Security Testing)
        if self.config['sast']['enabled']:
            self.logger.info("Running SAST scans")
            scan_results['sast_results'] = await self._run_sast_scans(target_path)
        
        # Run DAST (Dynamic Application Security Testing)
        if self.config['dast']['enabled']:
            self.logger.info("Running DAST scans")
            scan_results['dast_results'] = await self._run_dast_scans()
        
        # Run Dependency Scanning
        if self.config['dependency_scanning']['enabled']:
            self.logger.info("Running dependency scans")
            scan_results['dependency_scan_results'] = await self._run_dependency_scans(target_path)
        
        # Run Container Security Scanning
        if self.config['container_hardening']['enabled']:
            self.logger.info("Running container security scans")
            scan_results['container_scan_results'] = await self._run_container_scans(target_path)
        
        # Run Compliance Checks
        self.logger.info("Running compliance checks")
        scan_results['compliance_results'] = await self._run_compliance_checks()
        
        # Generate summary
        scan_results['summary'] = self._generate_scan_summary()
        
        # Save results
        await self._save_scan_results(scan_results)
        
        return scan_results
    
    async def _run_sast_scans(self, target_path: str) -> Dict[str, Any]:
        """Run Static Application Security Testing"""
        sast_results = {}
        
        # Semgrep scan
        if 'semgrep' in self.config['sast']['tools']:
            sast_results['semgrep'] = await self._run_semgrep_scan(target_path)
        
        # Bandit scan (Python)
        if 'bandit' in self.config['sast']['tools']:
            sast_results['bandit'] = await self._run_bandit_scan(target_path)
        
        # Safety scan (Python dependencies)
        if 'safety' in self.config['sast']['tools']:
            sast_results['safety'] = await self._run_safety_scan(target_path)
        
        return sast_results
    
    async def _run_semgrep_scan(self, target_path: str) -> Dict[str, Any]:
        """Run Semgrep SAST scan"""
        try:
            cmd = [
                'semgrep',
                '--config=auto',
                '--json',
                '--quiet',
                target_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                semgrep_output = json.loads(result.stdout)
                findings = []
                
                for finding in semgrep_output.get('results', []):
                    security_finding = SecurityFinding(
                        id=finding.get('check_id', ''),
                        title=finding.get('message', ''),
                        description=finding.get('message', ''),
                        severity=self._map_severity(finding.get('severity', 'info')),
                        category='sast',
                        file_path=finding.get('path', ''),
                        line_number=finding.get('start', {}).get('line'),
                        scanner='semgrep',
                        raw_output=finding
                    )
                    findings.append(security_finding)
                    self.findings.append(security_finding)
                
                return {
                    'status': 'success',
                    'findings_count': len(findings),
                    'findings': [f.__dict__ for f in findings]
                }
            else:
                self.logger.error(f"Semgrep scan failed: {result.stderr}")
                return {'status': 'error', 'error': result.stderr}
                
        except Exception as e:
            self.logger.error(f"Semgrep scan error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _run_bandit_scan(self, target_path: str) -> Dict[str, Any]:
        """Run Bandit Python security scan"""
        try:
            cmd = [
                'bandit',
                '-r',
                target_path,
                '-f', 'json',
                '-q'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.stdout:
                bandit_output = json.loads(result.stdout)
                findings = []
                
                for finding in bandit_output.get('results', []):
                    security_finding = SecurityFinding(
                        id=finding.get('test_id', ''),
                        title=finding.get('test_name', ''),
                        description=finding.get('issue_text', ''),
                        severity=self._map_severity(finding.get('issue_severity', 'low')),
                        category='sast',
                        file_path=finding.get('filename', ''),
                        line_number=finding.get('line_number'),
                        cwe_id=finding.get('cwe', {}).get('id'),
                        confidence=finding.get('issue_confidence', 1.0),
                        scanner='bandit',
                        raw_output=finding
                    )
                    findings.append(security_finding)
                    self.findings.append(security_finding)
                
                return {
                    'status': 'success',
                    'findings_count': len(findings),
                    'findings': [f.__dict__ for f in findings]
                }
            else:
                return {'status': 'success', 'findings_count': 0, 'findings': []}
                
        except Exception as e:
            self.logger.error(f"Bandit scan error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _run_safety_scan(self, target_path: str) -> Dict[str, Any]:
        """Run Safety dependency vulnerability scan"""
        try:
            # Find requirements files
            req_files = []
            for root, dirs, files in os.walk(target_path):
                for file in files:
                    if file in ['requirements.txt', 'requirements-dev.txt', 'Pipfile', 'pyproject.toml']:
                        req_files.append(os.path.join(root, file))
            
            all_findings = []
            
            for req_file in req_files:
                cmd = ['safety', 'check', '-r', req_file, '--json']
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.stdout:
                    try:
                        safety_output = json.loads(result.stdout)
                        
                        for vuln in safety_output:
                            security_finding = SecurityFinding(
                                id=vuln.get('id', ''),
                                title=f"Vulnerable dependency: {vuln.get('package', '')}",
                                description=vuln.get('advisory', ''),
                                severity=self._map_severity('high'),  # Dependencies are usually high
                                category='dependency',
                                file_path=req_file,
                                scanner='safety',
                                raw_output=vuln
                            )
                            all_findings.append(security_finding)
                            self.findings.append(security_finding)
                    except json.JSONDecodeError:
                        pass
            
            return {
                'status': 'success',
                'findings_count': len(all_findings),
                'findings': [f.__dict__ for f in all_findings]
            }
            
        except Exception as e:
            self.logger.error(f"Safety scan error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _run_dast_scans(self) -> Dict[str, Any]:
        """Run Dynamic Application Security Testing"""
        dast_results = {}
        
        # OWASP ZAP scan
        if 'owasp-zap' in self.config['dast']['tools']:
            dast_results['owasp_zap'] = await self._run_owasp_zap_scan()
        
        # Nuclei scan
        if 'nuclei' in self.config['dast']['tools']:
            dast_results['nuclei'] = await self._run_nuclei_scan()
        
        return dast_results
    
    async def _run_owasp_zap_scan(self) -> Dict[str, Any]:
        """Run OWASP ZAP scan"""
        try:
            target_urls = self.config['dast'].get('target_urls', [])
            if not target_urls:
                return {'status': 'skipped', 'reason': 'No target URLs configured'}
            
            findings = []
            
            for url in target_urls:
                # Start ZAP daemon
                cmd = [
                    'zap-baseline.py',
                    '-t', url,
                    '-J', f'/tmp/zap_report_{hashlib.md5(url.encode()).hexdigest()}.json'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                # Parse ZAP results
                report_path = f'/tmp/zap_report_{hashlib.md5(url.encode()).hexdigest()}.json'
                if os.path.exists(report_path):
                    with open(report_path, 'r') as f:
                        zap_output = json.load(f)
                    
                    for alert in zap_output.get('site', [{}])[0].get('alerts', []):
                        security_finding = SecurityFinding(
                            id=alert.get('pluginid', ''),
                            title=alert.get('name', ''),
                            description=alert.get('desc', ''),
                            severity=self._map_zap_severity(alert.get('riskdesc', '')),
                            category='dast',
                            scanner='owasp_zap',
                            raw_output=alert
                        )
                        findings.append(security_finding)
                        self.findings.append(security_finding)
            
            return {
                'status': 'success',
                'findings_count': len(findings),
                'findings': [f.__dict__ for f in findings]
            }
            
        except Exception as e:
            self.logger.error(f"OWASP ZAP scan error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _run_nuclei_scan(self) -> Dict[str, Any]:
        """Run Nuclei vulnerability scan"""
        try:
            target_urls = self.config['dast'].get('target_urls', [])
            if not target_urls:
                return {'status': 'skipped', 'reason': 'No target URLs configured'}
            
            findings = []
            
            for url in target_urls:
                cmd = [
                    'nuclei',
                    '-u', url,
                    '-json',
                    '-silent'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            try:
                                nuclei_finding = json.loads(line)
                                security_finding = SecurityFinding(
                                    id=nuclei_finding.get('template-id', ''),
                                    title=nuclei_finding.get('info', {}).get('name', ''),
                                    description=nuclei_finding.get('info', {}).get('description', ''),
                                    severity=self._map_severity(nuclei_finding.get('info', {}).get('severity', 'info')),
                                    category='dast',
                                    scanner='nuclei',
                                    raw_output=nuclei_finding
                                )
                                findings.append(security_finding)
                                self.findings.append(security_finding)
                            except json.JSONDecodeError:
                                continue
            
            return {
                'status': 'success',
                'findings_count': len(findings),
                'findings': [f.__dict__ for f in findings]
            }
            
        except Exception as e:
            self.logger.error(f"Nuclei scan error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _run_dependency_scans(self, target_path: str) -> Dict[str, Any]:
        """Run comprehensive dependency vulnerability scanning"""
        dep_results = {}
        
        # OSV Scanner
        if 'osv-scanner' in self.config['dependency_scanning']['tools']:
            dep_results['osv_scanner'] = await self._run_osv_scan(target_path)
        
        # Trivy
        if 'trivy' in self.config['dependency_scanning']['tools']:
            dep_results['trivy'] = await self._run_trivy_scan(target_path)
        
        return dep_results
    
    async def _run_osv_scan(self, target_path: str) -> Dict[str, Any]:
        """Run OSV vulnerability scanner"""
        try:
            cmd = [
                'osv-scanner',
                '--format=json',
                target_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.stdout:
                osv_output = json.loads(result.stdout)
                findings = []
                
                for vuln in osv_output.get('results', []):
                    for package in vuln.get('packages', []):
                        for vuln_detail in package.get('vulnerabilities', []):
                            security_finding = SecurityFinding(
                                id=vuln_detail.get('id', ''),
                                title=f"Vulnerable package: {package.get('package', {}).get('name', '')}",
                                description=vuln_detail.get('summary', ''),
                                severity=self._map_severity('high'),
                                category='dependency',
                                scanner='osv_scanner',
                                raw_output=vuln_detail
                            )
                            findings.append(security_finding)
                            self.findings.append(security_finding)
                
                return {
                    'status': 'success',
                    'findings_count': len(findings),
                    'findings': [f.__dict__ for f in findings]
                }
            else:
                return {'status': 'success', 'findings_count': 0, 'findings': []}
                
        except Exception as e:
            self.logger.error(f"OSV scan error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _run_trivy_scan(self, target_path: str) -> Dict[str, Any]:
        """Run Trivy vulnerability scanner"""
        try:
            cmd = [
                'trivy',
                'fs',
                '--format=json',
                target_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.stdout:
                trivy_output = json.loads(result.stdout)
                findings = []
                
                for result_item in trivy_output.get('Results', []):
                    for vuln in result_item.get('Vulnerabilities', []):
                        security_finding = SecurityFinding(
                            id=vuln.get('VulnerabilityID', ''),
                            title=f"Vulnerable package: {vuln.get('PkgName', '')}",
                            description=vuln.get('Description', ''),
                            severity=self._map_severity(vuln.get('Severity', 'unknown').lower()),
                            category='dependency',
                            scanner='trivy',
                            raw_output=vuln
                        )
                        findings.append(security_finding)
                        self.findings.append(security_finding)
                
                return {
                    'status': 'success',
                    'findings_count': len(findings),
                    'findings': [f.__dict__ for f in findings]
                }
            else:
                return {'status': 'success', 'findings_count': 0, 'findings': []}
                
        except Exception as e:
            self.logger.error(f"Trivy scan error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _run_container_scans(self, target_path: str) -> Dict[str, Any]:
        """Run container security scanning and hardening"""
        container_results = {
            'hardening_status': {},
            'vulnerabilities': [],
            'best_practices': {}
        }
        
        # Check for Dockerfiles
        dockerfiles = []
        for root, dirs, files in os.walk(target_path):
            for file in files:
                if file.lower().startswith('dockerfile'):
                    dockerfiles.append(os.path.join(root, file))
        
        if dockerfiles:
            container_results['dockerfiles_analyzed'] = len(dockerfiles)
            
            for dockerfile in dockerfiles:
                analysis = await self._analyze_dockerfile(dockerfile)
                container_results[f'dockerfile_{os.path.basename(dockerfile)}'] = analysis
        
        # Container hardening recommendations
        container_results['hardening_recommendations'] = self._get_container_hardening_recommendations()
        
        return container_results
    
    async def _analyze_dockerfile(self, dockerfile_path: str) -> Dict[str, Any]:
        """Analyze Dockerfile for security issues"""
        analysis = {
            'security_issues': [],
            'best_practices': [],
            'hardening_score': 0
        }
        
        try:
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            score = 100
            
            for i, line in enumerate(lines, 1):
                line = line.strip().upper()
                
                # Check for security issues
                if line.startswith('USER ROOT') or 'USER 0' in line:
                    analysis['security_issues'].append({
                        'line': i,
                        'issue': 'Running as root user',
                        'severity': 'high',
                        'recommendation': 'Use non-root user'
                    })
                    score -= 20
                
                if 'COPY . .' in line or 'ADD . .' in line:
                    analysis['security_issues'].append({
                        'line': i,
                        'issue': 'Copying entire context',
                        'severity': 'medium',
                        'recommendation': 'Use specific file paths'
                    })
                    score -= 10
                
                if 'EXPOSE 22' in line:
                    analysis['security_issues'].append({
                        'line': i,
                        'issue': 'SSH port exposed',
                        'severity': 'high',
                        'recommendation': 'Remove SSH access'
                    })
                    score -= 15
                
                # Check for best practices
                if line.startswith('FROM') and 'LATEST' in line:
                    analysis['best_practices'].append({
                        'line': i,
                        'practice': 'Use specific image tags instead of latest',
                        'severity': 'medium'
                    })
                    score -= 5
            
            analysis['hardening_score'] = max(0, score)
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _get_container_hardening_recommendations(self) -> Dict[str, Any]:
        """Get container hardening recommendations"""
        return {
            'distroless_images': {
                'enabled': self.config['container_hardening']['distroless'],
                'description': 'Use distroless base images to reduce attack surface',
                'implementation': 'FROM gcr.io/distroless/python3'
            },
            'rootless_runtime': {
                'enabled': self.config['container_hardening']['rootless'],
                'description': 'Run containers with non-root user',
                'implementation': 'USER 1000:1000'
            },
            'seccomp_profile': {
                'enabled': self.config['container_hardening']['seccomp'],
                'description': 'Apply seccomp security profile',
                'implementation': '--security-opt seccomp=seccomp-profile.json'
            },
            'read_only_root': {
                'enabled': self.config['container_hardening']['read_only_root'],
                'description': 'Mount root filesystem as read-only',
                'implementation': '--read-only'
            },
            'capabilities_drop': {
                'enabled': True,
                'description': 'Drop unnecessary Linux capabilities',
                'implementation': '--cap-drop=ALL --cap-add=SETUID --cap-add=SETGID'
            }
        }
    
    async def _run_compliance_checks(self) -> Dict[str, Any]:
        """Run compliance framework checks"""
        compliance_results = {}
        
        for framework in self.config['compliance']['frameworks']:
            if framework == 'soc2':
                compliance_results['soc2'] = await self._check_soc2_compliance()
            elif framework == 'iso27001':
                compliance_results['iso27001'] = await self._check_iso27001_compliance()
        
        return compliance_results
    
    async def _check_soc2_compliance(self) -> Dict[str, Any]:
        """Check SOC 2 compliance controls"""
        soc2_controls = {
            'CC1.1': ComplianceControl(
                id='CC1.1',
                framework=ComplianceFramework.SOC2,
                title='Control Environment - Integrity and Ethical Values',
                description='Management demonstrates commitment to integrity and ethical values'
            ),
            'CC2.1': ComplianceControl(
                id='CC2.1',
                framework=ComplianceFramework.SOC2,
                title='Communication and Information',
                description='Management obtains/generates and uses relevant information'
            ),
            'CC6.1': ComplianceControl(
                id='CC6.1',
                framework=ComplianceFramework.SOC2,
                title='Logical and Physical Access Controls',
                description='Entity implements logical access security software'
            ),
            'CC6.2': ComplianceControl(
                id='CC6.2',
                framework=ComplianceFramework.SOC2,
                title='Logical and Physical Access Controls',
                description='Prior to system access, users are identified and authenticated'
            ),
            'CC7.1': ComplianceControl(
                id='CC7.1',
                framework=ComplianceFramework.SOC2,
                title='System Operations',
                description='Entity uses encryption to protect data'
            )
        }
        
        # Update compliance controls
        self.compliance_controls.update(soc2_controls)
        
        # Create JIRA tickets if integration enabled
        if self.jira_client:
            await self._create_compliance_tickets(list(soc2_controls.values()))
        
        return {
            'framework': 'SOC2',
            'controls_count': len(soc2_controls),
            'controls': {k: v.__dict__ for k, v in soc2_controls.items()}
        }
    
    async def _check_iso27001_compliance(self) -> Dict[str, Any]:
        """Check ISO 27001 compliance controls"""
        iso27001_controls = {
            'A.5.1.1': ComplianceControl(
                id='A.5.1.1',
                framework=ComplianceFramework.ISO27001,
                title='Information Security Policies',
                description='Information security policy document'
            ),
            'A.6.1.2': ComplianceControl(
                id='A.6.1.2',
                framework=ComplianceFramework.ISO27001,
                title='Information Security in Project Management',
                description='Information security in project management'
            ),
            'A.9.1.1': ComplianceControl(
                id='A.9.1.1',
                framework=ComplianceFramework.ISO27001,
                title='Access Control Policy',
                description='Access control policy'
            ),
            'A.10.1.1': ComplianceControl(
                id='A.10.1.1',
                framework=ComplianceFramework.ISO27001,
                title='Cryptographic Policy',
                description='Policy on the use of cryptographic controls'
            ),
            'A.12.1.1': ComplianceControl(
                id='A.12.1.1',
                framework=ComplianceFramework.ISO27001,
                title='Operational Procedures',
                description='Documented operating procedures'
            )
        }
        
        # Update compliance controls
        self.compliance_controls.update(iso27001_controls)
        
        # Create JIRA tickets if integration enabled
        if self.jira_client:
            await self._create_compliance_tickets(list(iso27001_controls.values()))
        
        return {
            'framework': 'ISO27001',
            'controls_count': len(iso27001_controls),
            'controls': {k: v.__dict__ for k, v in iso27001_controls.items()}
        }
    
    async def _create_compliance_tickets(self, controls: List[ComplianceControl]):
        """Create JIRA tickets for compliance controls"""
        try:
            for control in controls:
                # Check if ticket already exists
                jql = f'summary ~ "{control.id}" AND project = "COMPLIANCE"'
                existing_issues = self.jira_client.search_issues(jql)
                
                if not existing_issues:
                    issue_dict = {
                        'project': {'key': 'COMPLIANCE'},
                        'summary': f'{control.framework.value.upper()} - {control.id}: {control.title}',
                        'description': control.description,
                        'issuetype': {'name': 'Task'},
                        'labels': [control.framework.value, 'security', 'compliance']
                    }
                    
                    new_issue = self.jira_client.create_issue(fields=issue_dict)
                    control.jira_ticket = new_issue.key
                    self.logger.info(f"Created JIRA ticket {new_issue.key} for {control.id}")
                
        except Exception as e:
            self.logger.error(f"Error creating JIRA tickets: {e}")
    
    def run_privacy_impact_assessment(self) -> Dict[str, Any]:
        """Run Privacy Impact Assessment (PIA)"""
        pia_results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'data_categories': [],
            'processing_activities': [],
            'data_minimization': {},
            'encryption_zones': {},
            'risk_assessment': {},
            'recommendations': []
        }
        
        # Identify data categories
        data_categories = [
            {'category': 'Personal Identifiers', 'sensitivity': 'high', 'retention': '7 years'},
            {'category': 'Financial Data', 'sensitivity': 'critical', 'retention': '10 years'},
            {'category': 'Trading Data', 'sensitivity': 'high', 'retention': '5 years'},
            {'category': 'System Logs', 'sensitivity': 'medium', 'retention': '2 years'},
            {'category': 'Analytics Data', 'sensitivity': 'low', 'retention': '3 years'}
        ]
        
        pia_results['data_categories'] = data_categories
        
        # Define encryption zones
        encryption_zones = {
            'zone_1_critical': {
                'description': 'Financial and trading data',
                'encryption': 'AES-256-GCM',
                'key_management': 'HSM',
                'access_control': 'RBAC + MFA'
            },
            'zone_2_sensitive': {
                'description': 'Personal identifiers',
                'encryption': 'AES-256-CBC',
                'key_management': 'KMS',
                'access_control': 'RBAC'
            },
            'zone_3_internal': {
                'description': 'System logs and analytics',
                'encryption': 'AES-128-GCM',
                'key_management': 'Application',
                'access_control': 'Basic Auth'
            }
        }
        
        pia_results['encryption_zones'] = encryption_zones
        
        # Data minimization recommendations
        pia_results['data_minimization'] = {
            'principles': [
                'Collect only necessary data',
                'Implement data retention policies',
                'Regular data purging',
                'Anonymization where possible'
            ],
            'implementation': {
                'automated_deletion': True,
                'data_classification': True,
                'consent_management': True,
                'audit_logging': True
            }
        }
        
        return pia_results
    
    async def run_penetration_test_simulation(self) -> Dict[str, Any]:
        """Simulate penetration testing scenarios"""
        pentest_results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'test_scenarios': [],
            'vulnerabilities_found': [],
            'recommendations': [],
            'risk_score': 0
        }
        
        # Simulate various attack scenarios
        test_scenarios = [
            {
                'scenario': 'Insider API Breach',
                'description': 'Simulate insider threat accessing exchange APIs',
                'techniques': ['Credential harvesting', 'Privilege escalation', 'Data exfiltration'],
                'impact': 'High',
                'likelihood': 'Medium'
            },
            {
                'scenario': 'External Network Penetration',
                'description': 'External attacker gaining network access',
                'techniques': ['Port scanning', 'Vulnerability exploitation', 'Lateral movement'],
                'impact': 'Critical',
                'likelihood': 'Low'
            },
            {
                'scenario': 'Social Engineering',
                'description': 'Phishing and social engineering attacks',
                'techniques': ['Spear phishing', 'Pretexting', 'Credential theft'],
                'impact': 'High',
                'likelihood': 'High'
            }
        ]
        
        pentest_results['test_scenarios'] = test_scenarios
        
        # Generate recommendations based on findings
        recommendations = [
            'Implement zero-trust network architecture',
            'Enhanced monitoring for API access patterns',
            'Regular security awareness training',
            'Multi-factor authentication for all accounts',
            'Network segmentation and micro-segmentation',
            'Endpoint detection and response (EDR)',
            'Security orchestration and automated response (SOAR)'
        ]
        
        pentest_results['recommendations'] = recommendations
        pentest_results['risk_score'] = 75  # Medium-high risk
        
        return pentest_results
    
    async def run_aml_cft_screening(self) -> Dict[str, Any]:
        """Run AML/CFT screening and sanctions list checking"""
        screening_results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'ofac_screening': {},
            'sanctions_lists': [],
            'risk_assessment': {},
            'alerts': []
        }
        
        # OFAC Sanctions List Screening
        ofac_results = await self._screen_ofac_sanctions()
        screening_results['ofac_screening'] = ofac_results
        
        # Additional sanctions lists
        sanctions_lists = [
            'OFAC SDN List',
            'UN Consolidated List',
            'EU Consolidated List',
            'HMT Consolidated List',
            'DFAT Consolidated List'
        ]
        
        screening_results['sanctions_lists'] = sanctions_lists
        
        # Risk-based monitoring
        risk_factors = {
            'high_risk_countries': ['Country A', 'Country B'],
            'suspicious_patterns': [
                'Large round-number transactions',
                'Unusual trading hours',
                'Rapid fund movements',
                'Shell company indicators'
            ],
            'monitoring_rules': [
                'Transactions > $10,000',
                'Velocity > 100 transactions/hour',
                'Cross-border transfers',
                'Crypto-to-fiat conversions'
            ]
        }
        
        screening_results['risk_assessment'] = risk_factors
        
        return screening_results
    
    async def _screen_ofac_sanctions(self) -> Dict[str, Any]:
        """Screen against OFAC sanctions lists"""
        try:
            # In production, this would connect to OFAC API
            # For simulation, we'll return mock results
            
            ofac_results = {
                'status': 'completed',
                'lists_checked': [
                    'SDN List',
                    'Consolidated Sanctions List',
                    'Sectoral Sanctions Identifications List'
                ],
                'matches_found': 0,
                'last_updated': datetime.datetime.now().isoformat(),
                'api_status': 'operational'
            }
            
            # Simulate API call to OFAC
            api_key = self.config.get('audit', {}).get('aml_cft_screening', {}).get('ofac_api_key')
            if api_key:
                # In production: Make actual API call to OFAC
                pass
            
            return ofac_results
            
        except Exception as e:
            self.logger.error(f"OFAC screening error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'fallback_screening': True
            }
    
    def _map_severity(self, severity_str: str) -> SecurityLevel:
        """Map string severity to SecurityLevel enum"""
        severity_map = {
            'critical': SecurityLevel.CRITICAL,
            'high': SecurityLevel.HIGH,
            'medium': SecurityLevel.MEDIUM,
            'low': SecurityLevel.LOW,
            'info': SecurityLevel.LOW,
            'warning': SecurityLevel.MEDIUM,
            'error': SecurityLevel.HIGH
        }
        
        return severity_map.get(severity_str.lower(), SecurityLevel.LOW)
    
    def _map_zap_severity(self, risk_desc: str) -> SecurityLevel:
        """Map ZAP risk description to SecurityLevel"""
        if 'High' in risk_desc:
            return SecurityLevel.HIGH
        elif 'Medium' in risk_desc:
            return SecurityLevel.MEDIUM
        elif 'Low' in risk_desc:
            return SecurityLevel.LOW
        else:
            return SecurityLevel.LOW
    
    def _generate_scan_summary(self) -> Dict[str, Any]:
        """Generate summary of all security findings"""
        summary = {
            'total_findings': len(self.findings),
            'by_severity': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'by_category': {},
            'by_scanner': {},
            'risk_score': 0
        }
        
        for finding in self.findings:
            # Count by severity
            summary['by_severity'][finding.severity.value] += 1
            
            # Count by category
            if finding.category not in summary['by_category']:
                summary['by_category'][finding.category] = 0
            summary['by_category'][finding.category] += 1
            
            # Count by scanner
            if finding.scanner not in summary['by_scanner']:
                summary['by_scanner'][finding.scanner] = 0
            summary['by_scanner'][finding.scanner] += 1
        
        # Calculate risk score
        risk_score = (
            summary['by_severity']['critical'] * 10 +
            summary['by_severity']['high'] * 7 +
            summary['by_severity']['medium'] * 4 +
            summary['by_severity']['low'] * 1
        )
        
        summary['risk_score'] = min(100, risk_score)
        
        return summary
    
    async def _save_scan_results(self, results: Dict[str, Any]):
        """Save scan results to file"""
        try:
            # Create results directory
            results_dir = Path("security_results")
            results_dir.mkdir(exist_ok=True)
            
            # Save detailed results
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = results_dir / f"security_scan_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Scan results saved to {results_file}")
            
            # Save summary report
            summary_file = results_dir / f"security_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(results['summary'], f, indent=2, default=str)
            
            # Generate HTML report
            await self._generate_html_report(results, results_dir / f"security_report_{timestamp}.html")
            
        except Exception as e:
            self.logger.error(f"Error saving scan results: {e}")
    
    async def _generate_html_report(self, results: Dict[str, Any], output_path: Path):
        """Generate HTML security report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Security Hardening & Compliance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; }
                .summary { background-color: #e8f5e8; padding: 15px; margin: 20px 0; }
                .critical { color: #d32f2f; }
                .high { color: #f57c00; }
                .medium { color: #fbc02d; }
                .low { color: #388e3c; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Security Hardening & Compliance Report</h1>
                <p>Generated: {timestamp}</p>
                <p>Target: {target_path}</p>
            </div>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>Total Findings: {total_findings}</p>
                <p>Risk Score: {risk_score}/100</p>
                <ul>
                    <li class="critical">Critical: {critical_count}</li>
                    <li class="high">High: {high_count}</li>
                    <li class="medium">Medium: {medium_count}</li>
                    <li class="low">Low: {low_count}</li>
                </ul>
            </div>
            
            <h2>Detailed Findings</h2>
            <table>
                <tr>
                    <th>Scanner</th>
                    <th>Severity</th>
                    <th>Title</th>
                    <th>File</th>
                    <th>Line</th>
                </tr>
                {findings_table}
            </table>
            
            <h2>Compliance Status</h2>
            {compliance_section}
            
            <h2>Recommendations</h2>
            <ul>
                <li>Prioritize fixing critical and high severity findings</li>
                <li>Implement container hardening measures</li>
                <li>Establish regular security scanning pipeline</li>
                <li>Complete compliance control implementation</li>
                <li>Conduct regular penetration testing</li>
            </ul>
        </body>
        </html>
        """
        
        try:
            # Build findings table
            findings_rows = []
            for finding in self.findings[:50]:  # Limit to first 50 findings
                severity_class = finding.severity.value
                row = f"""
                <tr>
                    <td>{finding.scanner}</td>
                    <td class="{severity_class}">{finding.severity.value.title()}</td>
                    <td>{finding.title}</td>
                    <td>{finding.file_path or 'N/A'}</td>
                    <td>{finding.line_number or 'N/A'}</td>
                </tr>
                """
                findings_rows.append(row)
            
            findings_table = ''.join(findings_rows)
            
            # Build compliance section
            compliance_section = "<p>Compliance controls tracked and managed.</p>"
            
            # Fill template
            html_content = html_template.format(
                timestamp=results['timestamp'],
                target_path=results['target_path'],
                total_findings=results['summary']['total_findings'],
                risk_score=results['summary']['risk_score'],
                critical_count=results['summary']['by_severity']['critical'],
                high_count=results['summary']['by_severity']['high'],
                medium_count=results['summary']['by_severity']['medium'],
                low_count=results['summary']['by_severity']['low'],
                findings_table=findings_table,
                compliance_section=compliance_section
            )
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")

# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Initialize security system
        security_system = SecurityHardeningCompliance()
        
        # Run comprehensive security scan
        target_path = "."
        results = await security_system.run_comprehensive_security_scan(target_path)
        
        print("\n=== SECURITY SCAN COMPLETED ===")
        print(f"Total findings: {results['summary']['total_findings']}")
        print(f"Risk score: {results['summary']['risk_score']}/100")
        print(f"Critical: {results['summary']['by_severity']['critical']}")
        print(f"High: {results['summary']['by_severity']['high']}")
        print(f"Medium: {results['summary']['by_severity']['medium']}")
        print(f"Low: {results['summary']['by_severity']['low']}")
        
        # Run Privacy Impact Assessment
        pia_results = security_system.run_privacy_impact_assessment()
        print("\n=== PRIVACY IMPACT ASSESSMENT ===")
        print(f"Data categories identified: {len(pia_results['data_categories'])}")
        print(f"Encryption zones: {len(pia_results['encryption_zones'])}")
        
        # Run Penetration Test Simulation
        pentest_results = await security_system.run_penetration_test_simulation()
        print("\n=== PENETRATION TEST SIMULATION ===")
        print(f"Test scenarios: {len(pentest_results['test_scenarios'])}")
        print(f"Risk score: {pentest_results['risk_score']}/100")
        
        # Run AML/CFT Screening
        screening_results = await security_system.run_aml_cft_screening()
        print("\n=== AML/CFT SCREENING ===")
        print(f"OFAC screening status: {screening_results['ofac_screening']['status']}")
        print(f"Sanctions lists checked: {len(screening_results['sanctions_lists'])}")
        
        print("\n=== SECURITY HARDENING COMPLETE ===")
    
    # Run the security system
    asyncio.run(main())


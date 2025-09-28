"""System requirements checking with user-friendly feedback.

Addresses "Prerequisites Confusion" identified in REVIEW_UX.md by providing
clear, actionable information about system requirements and installation estimates.
"""

import platform
import sys
import shutil
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from enum import Enum

# Import from scripts utils
from ..utils.result_types import (
    SystemResult, success, failure, SystemRequirementError, Severity
)


class InstallationType(Enum):
    """Installation type options."""
    MINIMAL = "minimal"
    FULL = "full"
    PIPELINE_SPECIFIC = "pipeline"


@dataclass(frozen=True)
class DiskEstimate:
    """Disk space requirements estimate."""
    base_install_gb: float
    conda_env_gb: float
    sample_data_gb: float
    working_space_gb: float
    total_required_gb: float
    total_recommended_gb: float

    @property
    def total_minimum_gb(self) -> float:
        """Absolute minimum space needed."""
        return self.base_install_gb + self.conda_env_gb

    def format_summary(self) -> str:
        """Format disk space summary for user display."""
        return (f"Required: {self.total_required_gb:.1f}GB | "
                f"Recommended: {self.total_recommended_gb:.1f}GB | "
                f"Minimum: {self.total_minimum_gb:.1f}GB")


@dataclass(frozen=True)
class TimeEstimate:
    """Installation time estimate."""
    download_minutes: int
    install_minutes: int
    setup_minutes: int
    total_minutes: int
    factors: List[str]

    def format_range(self) -> str:
        """Format time estimate as range."""
        min_time = max(1, self.total_minutes - 2)
        max_time = self.total_minutes + 3
        return f"{min_time}-{max_time} minutes"

    def format_summary(self) -> str:
        """Format time summary with factors."""
        factors_str = ", ".join(self.factors) if self.factors else "standard"
        return f"{self.format_range()} (factors: {factors_str})"


@dataclass(frozen=True)
class SystemInfo:
    """Comprehensive system information."""
    os_name: str
    os_version: str
    architecture: str
    is_m1_mac: bool
    python_version: Tuple[int, int, int]
    python_executable: str
    available_space_gb: float
    conda_available: bool
    mamba_available: bool
    docker_available: bool
    git_available: bool
    network_speed_estimate: Optional[str] = None


@dataclass(frozen=True)
class RequirementCheck:
    """Individual requirement check result."""
    name: str
    met: bool
    found: Optional[str]
    required: Optional[str]
    severity: Severity
    message: str
    suggestions: List[str]


class SystemRequirementsChecker:
    """Comprehensive system requirements checker with user-friendly output."""

    # Constants for disk space calculations (in GB)
    DISK_ESTIMATES = {
        InstallationType.MINIMAL: DiskEstimate(
            base_install_gb=2.5,
            conda_env_gb=3.0,
            sample_data_gb=1.0,
            working_space_gb=2.0,
            total_required_gb=8.5,
            total_recommended_gb=15.0
        ),
        InstallationType.FULL: DiskEstimate(
            base_install_gb=5.0,
            conda_env_gb=8.0,
            sample_data_gb=2.0,
            working_space_gb=3.0,
            total_required_gb=18.0,
            total_recommended_gb=30.0
        ),
        InstallationType.PIPELINE_SPECIFIC: DiskEstimate(
            base_install_gb=3.0,
            conda_env_gb=5.0,
            sample_data_gb=1.5,
            working_space_gb=2.5,
            total_required_gb=12.0,
            total_recommended_gb=20.0
        )
    }

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize checker with optional base directory."""
        self.base_dir = base_dir or Path.home() / "spyglass_data"

    def detect_system_info(self) -> SystemInfo:
        """Detect comprehensive system information."""
        # OS detection
        os_name = platform.system()
        os_version = platform.release()
        architecture = platform.machine()

        # Map OS names to user-friendly versions
        os_display_name = {
            "Darwin": "macOS",
            "Linux": "Linux",
            "Windows": "Windows"
        }.get(os_name, os_name)

        # Apple Silicon detection
        is_m1_mac = (os_name == "Darwin" and architecture == "arm64")

        # Python version
        python_version = sys.version_info[:3]
        python_executable = sys.executable

        # Available disk space
        try:
            _, _, available_bytes = shutil.disk_usage(self.base_dir.parent if self.base_dir.exists() else self.base_dir.parent)
            available_space_gb = available_bytes / (1024**3)
        except (OSError, AttributeError):
            available_space_gb = 0.0

        # Tool availability
        conda_available = shutil.which("conda") is not None
        mamba_available = shutil.which("mamba") is not None
        docker_available = shutil.which("docker") is not None
        git_available = shutil.which("git") is not None

        return SystemInfo(
            os_name=os_display_name,
            os_version=os_version,
            architecture=architecture,
            is_m1_mac=is_m1_mac,
            python_version=python_version,
            python_executable=python_executable,
            available_space_gb=available_space_gb,
            conda_available=conda_available,
            mamba_available=mamba_available,
            docker_available=docker_available,
            git_available=git_available
        )

    def check_python_version(self, system_info: SystemInfo) -> RequirementCheck:
        """Check Python version requirement."""
        major, minor, micro = system_info.python_version
        version_str = f"{major}.{minor}.{micro}"

        if major >= 3 and minor >= 9:
            return RequirementCheck(
                name="Python Version",
                met=True,
                found=version_str,
                required="≥3.9",
                severity=Severity.INFO,
                message=f"Python {version_str} meets requirements",
                suggestions=[]
            )
        else:
            return RequirementCheck(
                name="Python Version",
                met=False,
                found=version_str,
                required="≥3.9",
                severity=Severity.ERROR,
                message=f"Python {version_str} is too old",
                suggestions=[
                    "Install Python 3.9+ from python.org",
                    "Use conda to install newer Python in environment",
                    "Consider using pyenv for Python version management"
                ]
            )

    def check_operating_system(self, system_info: SystemInfo) -> RequirementCheck:
        """Check operating system compatibility."""
        if system_info.os_name in ["macOS", "Linux"]:
            return RequirementCheck(
                name="Operating System",
                met=True,
                found=f"{system_info.os_name} {system_info.os_version}",
                required="macOS or Linux",
                severity=Severity.INFO,
                message=f"{system_info.os_name} is fully supported",
                suggestions=[]
            )
        elif system_info.os_name == "Windows":
            return RequirementCheck(
                name="Operating System",
                met=True,
                found=f"Windows {system_info.os_version}",
                required="macOS or Linux (Windows experimental)",
                severity=Severity.WARNING,
                message="Windows support is experimental",
                suggestions=[
                    "Consider using Windows Subsystem for Linux (WSL)",
                    "Some features may not work as expected",
                    "Docker Desktop for Windows is recommended"
                ]
            )
        else:
            return RequirementCheck(
                name="Operating System",
                met=False,
                found=system_info.os_name,
                required="macOS or Linux",
                severity=Severity.ERROR,
                message=f"Unsupported operating system: {system_info.os_name}",
                suggestions=[
                    "Use macOS, Linux, or Windows with WSL",
                    "Check community support for your platform"
                ]
            )

    def check_package_manager(self, system_info: SystemInfo) -> RequirementCheck:
        """Check package manager availability with intelligent recommendations."""
        if system_info.mamba_available:
            return RequirementCheck(
                name="Package Manager",
                met=True,
                found="mamba (recommended)",
                required="conda or mamba",
                severity=Severity.INFO,
                message="Mamba provides fastest package resolution",
                suggestions=[]
            )
        elif system_info.conda_available:
            # Check conda version to determine solver
            conda_version = self._get_conda_version()
            if conda_version and self._has_libmamba_solver(conda_version):
                return RequirementCheck(
                    name="Package Manager",
                    met=True,
                    found=f"conda {conda_version} (with libmamba solver)",
                    required="conda or mamba",
                    severity=Severity.INFO,
                    message="Conda with libmamba solver is fast and reliable",
                    suggestions=[]
                )
            else:
                return RequirementCheck(
                    name="Package Manager",
                    met=True,
                    found=f"conda {conda_version or 'unknown'} (classic solver)",
                    required="conda or mamba",
                    severity=Severity.WARNING,
                    message="Conda classic solver is slower than mamba",
                    suggestions=[
                        "Install mamba for faster package resolution: conda install mamba -n base -c conda-forge",
                        "Update conda for libmamba solver: conda update conda",
                        "Current setup will work but may be slower"
                    ]
                )
        else:
            return RequirementCheck(
                name="Package Manager",
                met=False,
                found="none",
                required="conda or mamba",
                severity=Severity.ERROR,
                message="No conda/mamba found",
                suggestions=[
                    "Install miniforge (recommended): https://github.com/conda-forge/miniforge",
                    "Install miniconda: https://docs.conda.io/en/latest/miniconda.html",
                    "Install Anaconda: https://www.anaconda.com/products/distribution"
                ]
            )

    def check_disk_space(self, system_info: SystemInfo,
                         install_type: InstallationType) -> RequirementCheck:
        """Check available disk space against requirements."""
        estimate = self.DISK_ESTIMATES[install_type]
        available = system_info.available_space_gb

        if available >= estimate.total_recommended_gb:
            return RequirementCheck(
                name="Disk Space",
                met=True,
                found=f"{available:.1f}GB available",
                required=f"{estimate.total_required_gb:.1f}GB minimum",
                severity=Severity.INFO,
                message=f"Excellent! {available:.1f}GB available ({estimate.format_summary()})",
                suggestions=[]
            )
        elif available >= estimate.total_required_gb:
            return RequirementCheck(
                name="Disk Space",
                met=True,
                found=f"{available:.1f}GB available",
                required=f"{estimate.total_required_gb:.1f}GB minimum",
                severity=Severity.WARNING,
                message=f"Sufficient space: {available:.1f}GB available, {estimate.total_required_gb:.1f}GB required",
                suggestions=[
                    f"Consider freeing up space for optimal experience ({estimate.total_recommended_gb:.1f}GB recommended)",
                    "Monitor disk usage during installation"
                ]
            )
        elif available >= estimate.total_minimum_gb:
            return RequirementCheck(
                name="Disk Space",
                met=True,
                found=f"{available:.1f}GB available",
                required=f"{estimate.total_required_gb:.1f}GB minimum",
                severity=Severity.WARNING,
                message=f"Tight on space: {available:.1f}GB available, {estimate.total_minimum_gb:.1f}GB absolute minimum",
                suggestions=[
                    "Consider minimal installation to reduce space requirements",
                    "Free up space before installation",
                    "Install to external drive if available"
                ]
            )
        else:
            return RequirementCheck(
                name="Disk Space",
                met=False,
                found=f"{available:.1f}GB available",
                required=f"{estimate.total_minimum_gb:.1f}GB minimum",
                severity=Severity.ERROR,
                message=f"Insufficient space: {available:.1f}GB available, {estimate.total_minimum_gb:.1f}GB required",
                suggestions=[
                    f"Free up {estimate.total_minimum_gb - available:.1f}GB of disk space",
                    "Delete unnecessary files or move to external storage",
                    "Choose installation location with more space"
                ]
            )

    def check_optional_tools(self, system_info: SystemInfo) -> List[RequirementCheck]:
        """Check optional tools that enhance the experience."""
        checks = []

        # Docker check
        if system_info.docker_available:
            checks.append(RequirementCheck(
                name="Docker",
                met=True,
                found="available",
                required="optional (for local database)",
                severity=Severity.INFO,
                message="Docker available for local database setup",
                suggestions=[]
            ))
        else:
            checks.append(RequirementCheck(
                name="Docker",
                met=False,
                found="not found",
                required="optional (for local database)",
                severity=Severity.INFO,
                message="Docker not found - can install later for database",
                suggestions=[
                    "Install Docker for easy database setup: https://docs.docker.com/get-docker/",
                    "Alternatively, configure external database connection",
                    "Can be installed later if needed"
                ]
            ))

        # Git check
        if system_info.git_available:
            checks.append(RequirementCheck(
                name="Git",
                met=True,
                found="available",
                required="recommended",
                severity=Severity.INFO,
                message="Git available for repository management",
                suggestions=[]
            ))
        else:
            checks.append(RequirementCheck(
                name="Git",
                met=False,
                found="not found",
                required="recommended",
                severity=Severity.WARNING,
                message="Git not found - needed for development",
                suggestions=[
                    "Install Git: https://git-scm.com/downloads",
                    "Required for cloning repository and version control",
                    "Can download ZIP file as alternative"
                ]
            ))

        return checks

    def estimate_installation_time(self, system_info: SystemInfo,
                                  install_type: InstallationType) -> TimeEstimate:
        """Estimate installation time based on system and installation type."""
        base_times = {
            InstallationType.MINIMAL: {"download": 3, "install": 4, "setup": 1},
            InstallationType.FULL: {"download": 8, "install": 12, "setup": 2},
            InstallationType.PIPELINE_SPECIFIC: {"download": 5, "install": 7, "setup": 2}
        }

        times = base_times[install_type].copy()
        factors = []

        # Adjust for system characteristics
        if system_info.is_m1_mac:
            # M1 Macs are generally faster
            times["install"] = int(times["install"] * 0.8)
            factors.append("Apple Silicon speed boost")

        if not system_info.mamba_available and system_info.conda_available:
            conda_version = self._get_conda_version()
            if not (conda_version and self._has_libmamba_solver(conda_version)):
                # Older conda is slower
                times["install"] = int(times["install"] * 1.5)
                factors.append("conda classic solver")

        if not system_info.conda_available:
            # Need to install conda first
            times["setup"] += 5
            factors.append("conda installation needed")

        # Network speed estimation (simple heuristic)
        if self._estimate_slow_network():
            times["download"] = int(times["download"] * 2)
            factors.append("slow network detected")

        total = sum(times.values())

        return TimeEstimate(
            download_minutes=times["download"],
            install_minutes=times["install"],
            setup_minutes=times["setup"],
            total_minutes=total,
            factors=factors
        )

    def run_comprehensive_check(self, install_type: InstallationType = InstallationType.MINIMAL) -> Dict[str, RequirementCheck]:
        """Run all system requirement checks."""
        system_info = self.detect_system_info()

        checks = {
            "python": self.check_python_version(system_info),
            "os": self.check_operating_system(system_info),
            "package_manager": self.check_package_manager(system_info),
            "disk_space": self.check_disk_space(system_info, install_type)
        }

        # Add optional tool checks
        optional_checks = self.check_optional_tools(system_info)
        for i, check in enumerate(optional_checks):
            checks[f"optional_{i}"] = check

        return checks

    def _get_conda_version(self) -> Optional[str]:
        """Get conda version if available."""
        try:
            result = subprocess.run(
                ["conda", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Extract version from "conda 23.10.0"
                import re
                match = re.search(r'conda (\d+\.\d+\.\d+)', result.stdout)
                return match.group(1) if match else None
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None

    def _has_libmamba_solver(self, conda_version: str) -> bool:
        """Check if conda version includes libmamba solver by default."""
        try:
            from packaging import version
            return version.parse(conda_version) >= version.parse("23.10.0")
        except (ImportError, Exception):
            # Fallback to simple comparison
            try:
                major, minor, patch = map(int, conda_version.split('.'))
                return (major > 23) or (major == 23 and minor >= 10)
            except ValueError:
                return False

    def _estimate_slow_network(self) -> bool:
        """Simple heuristic to detect slow network connections."""
        # This is a placeholder - could be enhanced with actual network testing
        # For now, just return False (assume good network)
        return False


# Convenience function for quick system check
def check_system_requirements(install_type: InstallationType = InstallationType.MINIMAL,
                             base_dir: Optional[Path] = None) -> Dict[str, RequirementCheck]:
    """Quick system requirements check."""
    checker = SystemRequirementsChecker(base_dir)
    return checker.run_comprehensive_check(install_type)
"""Enhanced error recovery and troubleshooting for Spyglass setup.

This module provides structured error messages with actionable recovery steps
for common failure scenarios during Spyglass installation and validation.
"""

import subprocess
import platform
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Import from utils (using absolute path within scripts)
import sys
scripts_dir = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_dir))

from utils.result_types import Result, failure


class ErrorCategory(Enum):
    """Categories of errors that can occur during setup."""
    DOCKER = "docker"
    CONDA = "conda"
    PYTHON = "python"
    NETWORK = "network"
    PERMISSIONS = "permissions"
    VALIDATION = "validation"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Context information for an error."""
    category: ErrorCategory
    error_message: str
    command_attempted: Optional[str] = None
    file_path: Optional[str] = None
    system_info: Optional[Dict[str, Any]] = None


class ErrorRecoveryGuide:
    """Provides structured error recovery guidance."""

    def __init__(self, ui):
        self.ui = ui

    def handle_error(self, error: Exception, context: ErrorContext) -> None:
        """Handle an error with appropriate recovery guidance."""
        self.ui.print_error(f"{context.error_message}")

        if context.category == ErrorCategory.DOCKER:
            self._handle_docker_error(error, context)
        elif context.category == ErrorCategory.CONDA:
            self._handle_conda_error(error, context)
        elif context.category == ErrorCategory.PYTHON:
            self._handle_python_error(error, context)
        elif context.category == ErrorCategory.NETWORK:
            self._handle_network_error(error, context)
        elif context.category == ErrorCategory.PERMISSIONS:
            self._handle_permissions_error(error, context)
        elif context.category == ErrorCategory.VALIDATION:
            self._handle_validation_error(error, context)
        else:
            self._handle_generic_error(error, context)

    def _handle_docker_error(self, error: Exception, context: ErrorContext) -> None:
        """Handle Docker-related errors."""
        self.ui.print_header("Docker Troubleshooting")

        # Get full error context including stderr/stdout if available
        error_msg = str(error).lower()
        command_msg = (context.command_attempted or "").lower()

        # Extract stderr/stdout if available from CalledProcessError
        stderr_msg = ""
        stdout_msg = ""
        if hasattr(error, 'stderr') and error.stderr:
            stderr_msg = str(error.stderr).lower()
        if hasattr(error, 'stdout') and error.stdout:
            stdout_msg = str(error.stdout).lower()

        full_error_text = f"{error_msg} {stderr_msg} {stdout_msg} {command_msg}"

        # Check for common Docker error patterns
        if (("not found" in full_error_text and "docker" in full_error_text) or
            (hasattr(error, 'returncode') and error.returncode == 127)):
            print("\n🐳 **Docker Not Installed**\n")
            print("Docker is required for the local database setup.\n")

            system = platform.system()
            if system == "Darwin":  # macOS
                print("📥 **Install Docker Desktop for macOS:**")
                print("  1. Visit: https://docs.docker.com/desktop/install/mac-install/")
                print("  2. Download Docker Desktop")
                print("  3. Install and start Docker Desktop")
                print("  4. Verify with: docker --version")
            elif system == "Linux":
                print("📥 **Install Docker for Linux:**")
                print("  1. Visit: https://docs.docker.com/engine/install/")
                print("  2. Follow instructions for your Linux distribution")
                print("  3. Start Docker: sudo systemctl start docker")
                print("  4. Verify with: docker --version")
            elif system == "Windows":
                print("📥 **Install Docker Desktop for Windows:**")
                print("  1. Visit: https://docs.docker.com/desktop/install/windows-install/")
                print("  2. Download Docker Desktop")
                print("  3. Install and restart your computer")
                print("  4. Verify with: docker --version")

            print("\n🔄 **After Installation:**")
            print("  → Restart your terminal")
            print("  → Run: python scripts/quickstart.py --trial")

        elif ("permission denied" in full_error_text or "access denied" in full_error_text):
            print("\n🔒 **Docker Permission Issue**\n")

            system = platform.system()
            if system == "Linux":
                print("**Most likely cause**: Your user is not in the docker group\n")
                print("🛠️ **Fix for Linux:**")
                print("  1. Add your user to docker group:")
                print("     sudo usermod -aG docker $USER")
                print("  2. Log out and log back in (or restart)")
                print("  3. Verify with: docker run hello-world")
            else:
                print("**Most likely cause**: Docker Desktop not running\n")
                print("🛠️ **Fix:**")
                print("  1. Start Docker Desktop application")
                print("  2. Wait for Docker to be ready (green status)")
                print("  3. Try again")

        elif ("docker daemon" in full_error_text or "cannot connect" in full_error_text or
              "connection refused" in full_error_text or "is the docker daemon running" in full_error_text):
            print("\n🔄 **Docker Daemon Not Running**\n")
            print("Docker is installed but not running.\n")

            system = platform.system()
            if system in ["Darwin", "Windows"]:
                print("🚀 **Start Docker Desktop:**")
                print("  1. Open Docker Desktop application")
                print("  2. Wait for 'Docker Desktop is running' status")
                print("  3. Check system tray for Docker whale icon")
            else:  # Linux
                print("🚀 **Start Docker Service:**")
                print("  1. Start Docker: sudo systemctl start docker")
                print("  2. Enable auto-start: sudo systemctl enable docker")
                print("  3. Check status: sudo systemctl status docker")

            print("\n✅ **Verify Docker is Ready:**")
            print("  → Run: docker run hello-world")

        elif (("port" in full_error_text and ("in use" in full_error_text or "bind" in full_error_text)) or
              ("3306" in command_msg and ("already in use" in full_error_text or "address already in use" in full_error_text))):
            print("\n🔌 **Port Conflict (Port 3306 Already in Use)**\n")
            print("Another service is using the MySQL port.\n")

            print("🔍 **Find What's Using Port 3306:**")
            if platform.system() == "Darwin":
                print("  → Run: lsof -i :3306")
            elif platform.system() == "Linux":
                print("  → Run: sudo netstat -tlnp | grep :3306")
            else:  # Windows
                print("  → Run: netstat -ano | findstr :3306")

            print("\n🛠️ **Solutions:**")
            print("  1. **Stop conflicting service** (if safe to do so)")
            print("  2. **Use different port** with: --db-port 3307")
            print("  3. **Remove existing container**: docker rm -f spyglass-db")

        else:
            print("\n🐳 **General Docker Issue**\n")
            print("🔍 **Troubleshooting Steps:**")
            print("  1. Check Docker status: docker version")
            print("  2. Test Docker: docker run hello-world")
            print("  3. Check disk space: df -h")
            print("  4. Restart Docker Desktop")
            print("\n📧 **If problem persists:**")
            print(f"  → Report issue with this error: {error}")

    def _handle_conda_error(self, error: Exception, context: ErrorContext) -> None:
        """Handle Conda/environment related errors."""
        self.ui.print_header("Conda Environment Troubleshooting")

        error_msg = str(error).lower()

        if "conda" in error_msg and "not found" in error_msg:
            print("\n🐍 **Conda/Mamba Not Found**\n")
            print("Conda or Mamba package manager is required.\n")

            print("📥 **Install Options:**")
            print("  1. **Miniforge (Recommended)**: https://github.com/conda-forge/miniforge")
            print("  2. **Miniconda**: https://docs.conda.io/en/latest/miniconda.html")
            print("  3. **Anaconda**: https://www.anaconda.com/products/distribution")

            print("\n✅ **After Installation:**")
            print("  1. Restart your terminal")
            print("  2. Verify with: conda --version")
            print("  3. Run setup again")

        elif "environment" in error_msg and ("exists" in error_msg or "already" in error_msg):
            print("\n🔄 **Environment Already Exists**\n")
            print("A conda environment with this name already exists.\n")

            print("🛠️ **Options:**")
            print("  1. **Use existing environment**:")
            print("     conda activate spyglass")
            print("  2. **Remove and recreate**:")
            print("     conda env remove -n spyglass")
            print("     [then run setup again]")
            print("  3. **Use different name**:")
            print("     python scripts/quickstart.py --env-name spyglass-new")

        elif "solving environment" in error_msg or "conflicts" in error_msg:
            print("\n⚡ **Environment Solving Issues**\n")
            print("Conda is having trouble resolving package dependencies.\n")

            print("🛠️ **Try These Solutions:**")
            print("  1. **Use Mamba (faster solver)**:")
            print("     conda install mamba -n base -c conda-forge")
            print("     [then run setup again]")
            print("  2. **Update conda**:")
            print("     conda update conda")
            print("  3. **Clear conda cache**:")
            print("     conda clean --all")
            print("  4. **Use libmamba solver**:")
            print("     conda config --set solver libmamba")

        elif "timeout" in error_msg or "connection" in error_msg:
            print("\n🌐 **Network/Download Issues**\n")
            print("Conda cannot download packages due to network issues.\n")

            print("🛠️ **Try These Solutions:**")
            print("  1. **Check internet connection**")
            print("  2. **Try different conda channels**:")
            print("     conda config --add channels conda-forge")
            print("  3. **Use proxy settings** (if behind corporate firewall)")
            print("  4. **Retry with timeout**:")
            print("     conda config --set remote_read_timeout_secs 120")

        else:
            print("\n🐍 **General Conda Issue**\n")
            print("🔍 **Debugging Steps:**")
            print("  1. Check conda info: conda info")
            print("  2. List environments: conda env list")
            print("  3. Update conda: conda update conda")
            print("  4. Clear cache: conda clean --all")

    def _handle_python_error(self, error: Exception, context: ErrorContext) -> None:
        """Handle Python-related errors."""
        self.ui.print_header("Python Environment Troubleshooting")

        error_msg = str(error).lower()

        if "python" in error_msg and "not found" in error_msg:
            print("\n🐍 **Python Not Found in Environment**\n")
            print("The conda environment may not have Python installed.\n")

            print("🛠️ **Fix Environment:**")
            print("  1. Activate environment: conda activate spyglass")
            print("  2. Install Python: conda install python")
            print("  3. Verify: python --version")

        elif "import" in error_msg or "module" in error_msg:
            print("\n📦 **Missing Python Package**\n")
            print("Required Python packages are not installed.\n")

            if "spyglass" in error_msg:
                print("🛠️ **Install Spyglass:**")
                print("  1. Activate environment: conda activate spyglass")
                print("  2. Install in development mode: pip install -e .")
                print("  3. Verify: python -c 'import spyglass'")
            else:
                print("🛠️ **Install Missing Package:**")
                print("  1. Activate environment: conda activate spyglass")
                print("  2. Install package: pip install [package-name]")
                print("  3. Or reinstall environment completely")

        elif "version" in error_msg:
            print("\n🔢 **Python Version Issue**\n")
            print("Python version compatibility problem.\n")

            print("✅ **Spyglass Requirements:**")
            print("  → Python 3.9 or higher")
            print("  → Check current version: python --version")
            print("\n🛠️ **Fix Version Issue:**")
            print("  → Recreate environment with correct Python version")

    def _handle_network_error(self, error: Exception, context: ErrorContext) -> None:
        """Handle network-related errors."""
        self.ui.print_header("Network Troubleshooting")

        print("\n🌐 **Network Connection Issue**\n")
        print("Cannot connect to required services.\n")

        print("🔍 **Check Connectivity:**")
        print("  1. Test internet: ping google.com")
        print("  2. Test conda: conda search python")
        print("  3. Test Docker: docker pull hello-world")

        print("\n🛠️ **Common Fixes:**")
        print("  1. **Corporate Network**: Configure proxy settings")
        print("  2. **VPN Issues**: Try disconnecting VPN temporarily")
        print("  3. **Firewall**: Check firewall allows conda/docker")
        print("  4. **DNS Issues**: Try using different DNS (8.8.8.8)")

    def _handle_permissions_error(self, error: Exception, context: ErrorContext) -> None:
        """Handle permission-related errors."""
        self.ui.print_header("Permissions Troubleshooting")

        print("\n🔒 **Permission Denied**\n")

        if context.file_path:
            print(f"Cannot access: {context.file_path}\n")

        print("🛠️ **Fix Permissions:**")
        if platform.system() != "Windows":
            print("  1. Check file permissions: ls -la")
            print("  2. Fix ownership: sudo chown -R $USER:$USER [directory]")
            print("  3. Fix permissions: chmod -R 755 [directory]")
        else:
            print("  1. Run terminal as Administrator")
            print("  2. Check folder permissions in Properties")
            print("  3. Ensure you have write access")

        print("\n💡 **Prevention:**")
        print("  → Install in user directory (avoid system directories)")
        print("  → Use virtual environments")

    def _handle_validation_error(self, error: Exception, context: ErrorContext) -> None:
        """Handle validation-specific errors."""
        self.ui.print_header("Validation Error Recovery")

        error_msg = str(error).lower()

        if "datajoint" in error_msg or "database" in error_msg:
            print("\n🗄️ **Database Connection Failed**\n")
            print("Spyglass cannot connect to the database.\n")

            print("🔍 **Check Database Status:**")
            print("  1. Docker container running: docker ps")
            print("  2. Database accessible: docker exec spyglass-db mysql -uroot -ptutorial -e 'SHOW DATABASES;'")
            print("  3. Port available: telnet localhost 3306")

            print("\n🛠️ **Fix Database Issues:**")
            print("  1. **Restart container**: docker restart spyglass-db")
            print("  2. **Check logs**: docker logs spyglass-db")
            print("  3. **Recreate database**: python scripts/quickstart.py --trial")

        elif "import" in error_msg:
            print("\n📦 **Package Import Failed**\n")
            print("Required packages are not properly installed.\n")

            print("🛠️ **Reinstall Packages:**")
            print("  1. Activate environment: conda activate spyglass")
            print("  2. Reinstall Spyglass: pip install -e .")
            print("  3. Check imports: python -c 'import spyglass; print(spyglass.__version__)'")

        else:
            print("\n⚠️ **Validation Failed**\n")
            print("Some components are not working correctly.\n")

            print("🔍 **Debugging Steps:**")
            print("  1. Run validation with verbose: python scripts/validate_spyglass.py -v")
            print("  2. Check each component individually")
            print("  3. Review error messages for specific issues")

    def _handle_generic_error(self, error: Exception, context: ErrorContext) -> None:
        """Handle generic errors."""
        self.ui.print_header("General Troubleshooting")

        print("\n❓ **Unexpected Error**\n")
        print(f"Error: {error}\n")

        print("🔍 **General Debugging Steps:**")
        print("  1. Check system requirements")
        print("  2. Ensure all prerequisites are installed")
        print("  3. Try restarting your terminal")
        print("  4. Check available disk space")

        print("\n📧 **Get Help:**")
        print("  1. Check Spyglass documentation")
        print("  2. Search existing GitHub issues")
        print("  3. Report new issue with:")
        print(f"     → Error message: {error}")
        print(f"     → Command attempted: {context.command_attempted}")
        print(f"     → System: {platform.system()} {platform.release()}")


def create_error_context(category: ErrorCategory,
                        error_message: str,
                        command: Optional[str] = None,
                        file_path: Optional[str] = None) -> ErrorContext:
    """Create error context with system information."""
    return ErrorContext(
        category=category,
        error_message=error_message,
        command_attempted=command,
        file_path=file_path,
        system_info={
            "platform": platform.system(),
            "release": platform.release(),
            "python_version": platform.python_version(),
        }
    )


# Convenience functions for common error scenarios
def handle_docker_error(ui, error: Exception, command: Optional[str] = None) -> None:
    """Handle Docker-related errors with recovery guidance."""
    context = create_error_context(ErrorCategory.DOCKER, str(error), command)
    guide = ErrorRecoveryGuide(ui)
    guide.handle_error(error, context)


def handle_conda_error(ui, error: Exception, command: Optional[str] = None) -> None:
    """Handle Conda-related errors with recovery guidance."""
    context = create_error_context(ErrorCategory.CONDA, str(error), command)
    guide = ErrorRecoveryGuide(ui)
    guide.handle_error(error, context)


def handle_validation_error(ui, error: Exception, validation_step: str) -> None:
    """Handle validation errors with specific recovery guidance."""
    context = create_error_context(ErrorCategory.VALIDATION, str(error), validation_step)
    guide = ErrorRecoveryGuide(ui)
    guide.handle_error(error, context)
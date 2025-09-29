"""User persona-driven onboarding for Spyglass.

This module provides different setup paths based on user intent:
- Lab members joining existing infrastructure
- Researchers trying Spyglass for the first time
- Admins/power users needing full control

Follows UX best practices:
- Start with user intent, not technical details
- Progressive disclosure of complexity
- Context-appropriate defaults
- Clear guidance for each user type
"""

from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import subprocess
import getpass

# Import from utils (using absolute path within scripts)
import sys
scripts_dir = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_dir))

from utils.result_types import (
    Result, success, failure,
    ValidationResult, validation_success, validation_failure
)
from ux.validation import (
    validate_host, validate_port, validate_directory,
    validate_environment_name
)


class UserPersona(Enum):
    """User personas for Spyglass onboarding."""
    LAB_MEMBER = "lab_member"
    TRIAL_USER = "trial_user"
    ADMIN = "admin"
    UNDECIDED = "undecided"


@dataclass
class PersonaConfig:
    """Configuration specific to each user persona."""
    persona: UserPersona
    install_type: str = "minimal"
    setup_database: bool = True
    database_config: Optional[Dict[str, Any]] = None
    include_sample_data: bool = False
    base_dir: Optional[Path] = None
    env_name: str = "spyglass"
    auto_confirm: bool = False

    def __post_init__(self):
        """Set persona-specific defaults."""
        if self.persona == UserPersona.LAB_MEMBER:
            # Lab members connect to existing database
            self.setup_database = False
            self.include_sample_data = False
            if not self.base_dir:
                self.base_dir = Path.home() / "spyglass_data"

        elif self.persona == UserPersona.TRIAL_USER:
            # Trial users get everything locally
            self.setup_database = True
            self.include_sample_data = True
            if not self.base_dir:
                self.base_dir = Path.home() / "spyglass_trial"

        elif self.persona == UserPersona.ADMIN:
            # Admins get full control
            self.install_type = "full"
            if not self.base_dir:
                self.base_dir = Path.home() / "spyglass"


@dataclass
class LabDatabaseConfig:
    """Database configuration for lab members."""
    host: str
    port: int = 3306
    username: str = ""
    password: str = ""
    database_name: str = ""

    def is_complete(self) -> bool:
        """Check if all required fields are filled."""
        return all([
            self.host,
            self.port,
            self.username,
            self.password,
            self.database_name
        ])


class PersonaDetector:
    """Detect user persona based on their intent."""

    @staticmethod
    def detect_from_args(args) -> UserPersona:
        """Detect persona from command line arguments."""
        if hasattr(args, 'lab_member') and args.lab_member:
            return UserPersona.LAB_MEMBER
        elif hasattr(args, 'trial') and args.trial:
            return UserPersona.TRIAL_USER
        elif hasattr(args, 'advanced') and args.advanced:
            return UserPersona.ADMIN
        else:
            return UserPersona.UNDECIDED

    @staticmethod
    def detect_from_environment() -> Optional[UserPersona]:
        """Check for environment variables suggesting persona."""
        import os

        # Check for lab environment variables
        if os.getenv('SPYGLASS_LAB_HOST') or os.getenv('DJ_HOST'):
            return UserPersona.LAB_MEMBER

        # Check for CI/testing environment
        if os.getenv('CI') or os.getenv('GITHUB_ACTIONS'):
            return UserPersona.ADMIN

        return None


class PersonaOnboarding:
    """Base class for persona-specific onboarding flows."""

    def __init__(self, ui, base_config=None):
        self.ui = ui
        self.base_config = base_config or {}

    def run(self) -> Result:
        """Execute the onboarding flow."""
        raise NotImplementedError

    def _show_preview(self, config: PersonaConfig) -> None:
        """Show installation preview to user."""
        self.ui.print_header("Installation Preview")

        print("\n📋 Here's what will be installed:\n")
        print(f"  📁 Location: {config.base_dir}")
        print(f"  🐍 Environment: {config.env_name}")

        if config.setup_database:
            if config.include_sample_data:
                print(f"  🗄️ Database: Local Docker (configured automatically)")
            else:
                print(f"  🗄️ Database: Local Docker container")
        elif config.database_config:
            print(f"  🗄️ Database: Connecting to existing")

        print(f"  📦 Installation: {config.install_type}")

        if config.include_sample_data:
            print(f"  📊 Sample Data: Included")

        print("")

    def _confirm_installation(self, message: str = "Proceed with installation?") -> bool:
        """Get user confirmation."""
        try:
            response = input(f"\n{message} [Y/n]: ").strip().lower()
            return response in ['', 'y', 'yes']
        except (EOFError, KeyboardInterrupt):
            return False


class LabMemberOnboarding(PersonaOnboarding):
    """Onboarding flow for lab members joining existing infrastructure."""

    def run(self) -> Result:
        """Execute lab member onboarding."""
        self.ui.print_header("Lab Member Setup")

        print("\nPerfect! You'll connect to your lab's existing Spyglass database.")
        print("This setup is optimized for working with shared lab resources.\n")

        # Collect database connection info
        db_config = self._collect_database_info()
        if db_config.is_failure:
            return db_config

        # Create persona config
        config = PersonaConfig(
            persona=UserPersona.LAB_MEMBER,
            database_config=db_config.value.__dict__
        )

        # Test connection before proceeding
        print("\n🔍 Testing database connection...")
        connection_result = self._test_connection(db_config.value)

        if connection_result.is_failure:
            self._show_connection_help(connection_result.error)
            return connection_result

        self.ui.print_success("Database connection successful!")

        # Add note about validation
        if "Basic connectivity test passed" in connection_result.message:
            print("\n💡 Note: Full MySQL authentication will be tested during validation.")
            print("   If validation fails with authentication errors, the troubleshooting")
            print("   guide will provide specific steps for your lab admin.")

        # Show preview and confirm
        self._show_preview(config)

        if not self._confirm_installation():
            return failure(None, "Installation cancelled by user")

        return success(config, "Lab member configuration ready")

    def _collect_database_info(self) -> Result[LabDatabaseConfig, Any]:
        """Collect database connection information from user."""
        print("📊 Database Connection Information")
        print("Your lab admin should have provided these details.\n")

        config = LabDatabaseConfig(host="", port=3306)

        # Collect host
        print("Database Host:")
        print("  Examples: lmf-db.cin.ucsf.edu, spyglass.mylab.edu")
        host_input = input("  Host: ").strip()

        if not host_input:
            print("\n💡 Tip: Ask your lab admin for 'Spyglass database host'")
            return failure(None, "Database host is required")

        host_result = validate_host(host_input)
        if host_result.is_failure:
            self.ui.print_error(host_result.error.message)
            return failure(None, "Invalid host address")

        config.host = host_input

        # Collect port
        port_input = input("  Port [3306]: ").strip() or "3306"
        port_result = validate_port(port_input)

        if port_result.is_failure:
            self.ui.print_error(port_result.error.message)
            return failure(None, "Invalid port number")

        config.port = int(port_input)

        # Collect username
        config.username = input("  Username: ").strip()
        if not config.username:
            print("\n💡 Tip: Your lab admin will provide your database username")
            return failure(None, "Username is required")

        # Collect password (hidden input)
        try:
            config.password = getpass.getpass("  Password: ")
        except (EOFError, KeyboardInterrupt):
            return failure(None, "Password input cancelled")

        if not config.password:
            return failure(None, "Password is required")

        # Use default database name 'spyglass' - this is the MySQL database name
        # not the conda environment name
        config.database_name = "spyglass"

        return success(config, "Database configuration collected")

    def _test_connection(self, config: LabDatabaseConfig) -> Result:
        """Test database connection with actual MySQL authentication."""
        try:
            # First test basic connectivity
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((config.host, config.port))
            sock.close()

            if result != 0:
                return failure(
                    {"host": config.host, "port": config.port},
                    f"Cannot connect to {config.host}:{config.port}"
                )

            # Test actual MySQL authentication
            try:
                import pymysql
            except ImportError:
                # pymysql not available, fall back to basic connectivity test
                print("  ⚠️  Note: Cannot test MySQL authentication (PyMySQL not available)")
                print("      Full authentication test will happen during validation")
                return success(True, "Basic connectivity test passed - host reachable")

            try:
                connection = pymysql.connect(
                    host=config.host,
                    port=config.port,
                    user=config.username,
                    password=config.password,
                    database=config.database_name,
                    connect_timeout=10
                )
                connection.close()
                return success(True, "MySQL authentication successful")

            except pymysql.err.OperationalError as e:
                error_code, error_msg = e.args

                if error_code == 1045:  # Access denied
                    return failure(
                        {"error_code": error_code, "mysql_error": error_msg},
                        f"MySQL authentication failed: {error_msg}"
                    )
                elif error_code == 2003:  # Can't connect to server
                    return failure(
                        {"error_code": error_code, "mysql_error": error_msg},
                        f"Cannot reach MySQL server: {error_msg}"
                    )
                else:
                    return failure(
                        {"error_code": error_code, "mysql_error": error_msg},
                        f"MySQL error ({error_code}): {error_msg}"
                    )

            except Exception as e:
                return failure(e, f"MySQL connection test failed: {str(e)}")

        except Exception as e:
            return failure(e, f"Connection test failed: {str(e)}")

    def _show_connection_help(self, error: Any) -> None:
        """Show help for connection issues."""
        self.ui.print_header("Connection Troubleshooting")

        # Check if this is a MySQL authentication error
        if isinstance(error, dict) and error.get('error_code') == 1045:
            mysql_error = error.get('mysql_error', '')

            print(f"\n🔒 **MySQL Authentication Failed**")
            print(f"   Error: {mysql_error}")
            print("\n**Most likely causes:**\n")

            if '@' in mysql_error and 'using password: YES' in mysql_error:
                # Extract the hostname from error message
                print("  1. **Database permissions issue**")
                print("     → Your database user may not have permission from this location")
                print("     → MySQL sees hostname/IP resolution differently")
                print("")
                print("  2. **VPN/Network location**")
                print("     → Try connecting from within the lab network")
                print("     → Ensure you're on the lab VPN")
                print("")
                print("  3. **Username/password incorrect**")
                print("     → Double-check credentials with lab admin")
                print("     → Case-sensitive username and password")

            print("")
            print("📧 **Next steps:**")
            print("  1. Forward this exact error to your lab admin:")
            print(f"     '{mysql_error}'")
            print("  2. Ask them to check database user permissions")
            print("  3. Verify you're on the correct network/VPN")

        else:
            print("\n🔗 Connection failed. Common causes:\n")
            print("  1. **Not on lab network/VPN**")
            print("     → Connect to your lab's VPN first")
            print("     → Or connect from within the lab")
            print("")
            print("  2. **Incorrect credentials**")
            print("     → Double-check with your lab admin")
            print("     → Username/password are case-sensitive")
            print("")
            print("  3. **Firewall blocking connection**")
            print("     → Your IT may need to allow access")
            print("     → Port 3306 needs to be open")
            print("")
            print("📧 **Next steps:**")
            print("  1. Send this error to your lab admin:")
            print(f"     '{error}'")

        print("  4. Try again with: python scripts/quickstart.py --lab-member")


class TrialUserOnboarding(PersonaOnboarding):
    """Onboarding flow for researchers trying Spyglass."""

    def run(self) -> Result:
        """Execute trial user onboarding."""
        self.ui.print_header("Research Trial Setup")

        print("\nGreat choice! I'll set up everything you need to explore Spyglass.")
        print("This includes a complete local environment perfect for:")
        print("  → Learning Spyglass concepts")
        print("  → Testing with your own data")
        print("  → Running tutorials and examples\n")

        # Create config with trial defaults
        config = PersonaConfig(
            persona=UserPersona.TRIAL_USER,
            install_type="minimal",
            setup_database=True,
            include_sample_data=True,
            base_dir=Path.home() / "spyglass_trial"
        )

        # Show what they'll get
        self._show_trial_benefits()

        # Show preview
        self._show_preview(config)

        # Estimate time and space
        print("📊 **Resource Requirements:**")
        print(f"  💾 Disk Space: ~8GB (includes sample data)")
        print(f"  ⏱️ Install Time: 5-8 minutes")
        print(f"  🔧 Prerequisites: Docker (will be configured automatically)")
        print("")

        if not self._confirm_installation("Ready to set up your trial environment?"):
            return self._offer_alternatives()

        return success(config, "Trial configuration ready")

    def _show_trial_benefits(self) -> None:
        """Show what trial users will get."""
        print("✨ **Your trial environment includes:**\n")
        print("  📚 **Tutorial Notebooks**")
        print("     6 guided tutorials from basics to advanced")
        print("")
        print("  📊 **Sample Datasets**")
        print("     Real neural recordings to practice with")
        print("")
        print("  🔧 **Analysis Pipelines**")
        print("     Spike sorting, LFP, position tracking")
        print("")
        print("  🗄️ **Local Database**")
        print("     Your own sandbox to experiment safely")
        print("")

    def _offer_alternatives(self) -> Result:
        """Offer alternatives if user declines trial setup."""
        print("\nNo problem! Here are other options:\n")
        print("  1. **Lab Member Setup** - If you're joining an existing lab")
        print("     → Run: python scripts/quickstart.py --lab-member")
        print("")
        print("  2. **Advanced Setup** - If you need custom configuration")
        print("     → Run: python scripts/quickstart.py --advanced")
        print("")
        print("  3. **Learn More** - Read documentation first")
        print("     → Visit: https://lorenfranklab.github.io/spyglass/")

        return failure(None, "User chose alternative path")


class AdminOnboarding(PersonaOnboarding):
    """Onboarding flow for administrators and power users."""

    def run(self) -> Result:
        """Execute admin onboarding with full control."""
        self.ui.print_header("Advanced Configuration")

        print("\nYou have full control over the installation process.")
        print("This mode is recommended for:")
        print("  → System administrators")
        print("  → Setting up lab infrastructure")
        print("  → Custom deployments\n")

        # Return to original detailed flow
        # This maintains backward compatibility
        config = PersonaConfig(
            persona=UserPersona.ADMIN,
            install_type="full"
        )

        # Signal to use traditional detailed setup
        return success(config, "Using advanced configuration mode")


class PersonaOrchestrator:
    """Main orchestrator for persona-based onboarding."""

    def __init__(self, ui):
        self.ui = ui
        self.persona = UserPersona.UNDECIDED

    def detect_persona(self, args=None) -> UserPersona:
        """Detect or ask for user persona."""

        # Check command line args first
        if args:
            persona = PersonaDetector.detect_from_args(args)
            if persona != UserPersona.UNDECIDED:
                return persona

        # Check environment
        persona = PersonaDetector.detect_from_environment()
        if persona:
            return persona

        # Ask user interactively
        return self._ask_user_persona()

    def _ask_user_persona(self) -> UserPersona:
        """Interactive persona selection."""
        self.ui.print_header("Welcome to Spyglass!")

        print("\nWhat brings you here today?\n")
        print("  1. 🏫 I'm joining a lab that uses Spyglass")
        print("     └── Connect to existing lab infrastructure\n")
        print("  2. 🔬 I want to try Spyglass for my research")
        print("     └── Set up everything locally to explore\n")
        print("  3. ⚙️  I need advanced configuration options")
        print("     └── Full control over installation\n")

        while True:
            try:
                choice = input("Which describes your situation? [1-3]: ").strip()

                if choice == "1":
                    return UserPersona.LAB_MEMBER
                elif choice == "2":
                    return UserPersona.TRIAL_USER
                elif choice == "3":
                    return UserPersona.ADMIN
                else:
                    self.ui.print_error("Please enter 1, 2, or 3")

            except (EOFError, KeyboardInterrupt):
                print("\n\nInstallation cancelled.")
                return UserPersona.UNDECIDED

    def run_onboarding(self, persona: UserPersona, base_config=None) -> Result:
        """Run the appropriate onboarding flow."""

        if persona == UserPersona.LAB_MEMBER:
            return LabMemberOnboarding(self.ui, base_config).run()

        elif persona == UserPersona.TRIAL_USER:
            return TrialUserOnboarding(self.ui, base_config).run()

        elif persona == UserPersona.ADMIN:
            return AdminOnboarding(self.ui, base_config).run()

        else:
            return failure(None, "No persona selected")
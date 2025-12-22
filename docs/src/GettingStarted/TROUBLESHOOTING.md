# Spyglass Installation Troubleshooting

This guide helps resolve common installation issues with Spyglass.

## Quick Diagnosis

Run the validation script to identify issues:

```bash
python scripts/validate.py
```

The validator will check:

- ✓ Python version compatibility
- ✓ Conda/Mamba availability
- ✓ Spyglass import
- ⚠ SpyglassConfig (optional)
- ⚠ Database connection (optional)

## Common Issues

### Environment Creation Fails

**Symptoms:**

- `conda env create` hangs or fails
- Package conflict errors
- Timeout during solving environment

**Solutions:**

1. **Update conda/mamba:**

   ```bash
   conda update conda
   # or
   mamba update mamba
   ```

2. **Clear package cache:**

   ```bash
   conda clean --all
   ```

3. **Try mamba (faster, better at resolving conflicts):**

   ```bash
   conda install mamba -c conda-forge
   mamba env create -f environment.yml
   ```

4. **Use minimal installation first:**

   ```bash
   python scripts/install.py --minimal
   ```

5. **Check disk space:**
   - Minimal: ~10 GB required
   - Full: ~25 GB required

   ```bash
   df -h
   ```

### Docker Database Issues

**Symptoms:**

- "Docker not available"
- Container fails to start
- MySQL timeout waiting for readiness

**Solutions:**

1. **Verify Docker is installed and running:**

   ```bash
   docker --version
   docker ps
   ```

2. **Start Docker Desktop** (macOS/Windows)
   - Check system tray for Docker icon
   - Ensure Docker Desktop is running

3. **Check Docker permissions** (Linux):

   ```bash
   sudo usermod -aG docker $USER
   # Then log out and back in
   ```

4. **Container already exists:**

   ```bash
   # Check if container exists
   docker ps -a | grep spyglass-db

   # Remove old container
   docker rm -f spyglass-db # This will delete all data in the container!

   # Try installation again
   python scripts/install.py --docker
   ```

5. **Port 3306 already in use:**

   ```bash
   # Check what's using port 3306
   lsof -i :3306
   # or
   netstat -an | grep 3306

   # Stop conflicting service or use different port
   ```

6. **Container starts but MySQL times out:**

   ```bash
   # Check container logs
   docker logs spyglass-db

   # Wait longer and check again
   docker exec spyglass-db mysqladmin -uroot -ptutorial ping
   ```

### Remote Database Connection Fails

**Symptoms:**

- "Connection refused"
- "Access denied for user"
- TLS/SSL errors

**Solutions:**

1. **Verify credentials:**
   - Double-check host, port, username, password
   - Try connecting with mysql CLI:

   ```bash
   mysql -h HOST -P PORT -u USER -p
   ```

2. **Check network/firewall:**

   ```bash
   # Test if port is open
   telnet HOST PORT
   # or
   nc -zv HOST PORT
   ```

3. **TLS configuration:**
   - For `localhost`, TLS should be disabled
   - For remote hosts, TLS should be enabled
   - If TLS errors occur, verify server certificate

4. **Database permissions:**

   ```sql
   -- Run on MySQL server
   GRANT ALL PRIVILEGES ON *.* TO 'user'@'%' IDENTIFIED BY 'password';
   FLUSH PRIVILEGES;
   ```

### Python Version Issues

**Symptoms:**

- "Python 3.9+ required, found 3.8"
- Import errors for newer Python features

**Solutions:**

1. **Check Python version:**

   ```bash
   python --version
   ```

2. **Install correct Python version:**

   ```bash
   # Using conda
   conda install python=3.10

   # Or create new environment
   conda create -n spyglass python=3.10
   ```

3. **Verify conda environment:**

   ```bash
   # Check active environment
   conda info --envs

   # Activate spyglass environment
   conda activate spyglass
   ```

### Spyglass Import Fails

**Symptoms:**

- `ModuleNotFoundError: No module named 'spyglass'`
- Import errors for spyglass submodules

**Solutions:**

1. **Verify installation:**

   ```bash
   conda activate spyglass
   pip show spyglass
   ```

2. **Reinstall in development mode:**

   ```bash
   cd /path/to/spyglass
   pip install -e .
   pip show spyglass-neuro # confirm installation
   ```

3. **Check sys.path:**

   ```python
   import sys
   print(sys.path)
   # Should include spyglass source directory
   ```

### SpyglassConfig Issues

**Symptoms:**

- "Cannot find configuration file"
- Base directory errors

**Solutions:**

1. **Check config file location:**

   ```bash
   ls -la ~/.datajoint_config.json
   # or
   ls -la ./dj_local_conf.json
   ```

2. **Set base directory:**

   ```bash
   export SPYGLASS_BASE_DIR=/path/to/data
   ```

3. **Create default config:**

   ```python
   from spyglass.settings import SpyglassConfig
   config = SpyglassConfig()  # Auto-creates if missing
   ```

### DataJoint Configuration Issues

**Symptoms:**

- "Could not connect to database"
- Configuration file not found

**Solutions:**

1. **Check DataJoint config:**

   ```bash
   cat ~/.datajoint_config.json
   ```

2. **Manually create config** (`~/.datajoint_config.json`):

   ```json
   {
     "database.host": "localhost",
     "database.port": 3306,
     "database.user": "root",
     "database.password": "tutorial",
     "database.use_tls": false
   }
   ```

3. **Test connection:**

   ```python
   import datajoint as dj
   dj.conn().ping()
   ```

### M1/M2 Mac Issues

**Symptoms:**

- Architecture mismatch errors
- Rosetta warnings
- Package installation failures

**Solutions:**

1. **Use native ARM environment:**

   ```bash
   # Ensure using ARM conda
   conda config --env --set subdir osx-arm64
   ```

2. **Some packages may require Rosetta:**

   ```bash
   # Install Rosetta 2 if needed
   softwareupdate --install-rosetta
   ```

3. **Use mamba for better ARM support:**

   ```bash
   conda install mamba -c conda-forge
   mamba env create -f environment.yml
   ```

### Insufficient Disk Space

**Symptoms:**

- Installation fails partway through
- "No space left on device"

**Solutions:**

1. **Check available space:**

   ```bash
   df -h
   ```

2. **Clean conda cache:**

   ```bash
   conda clean --all
   ```

3. **Choose different installation directory:**

   ```bash
   python scripts/install.py --base-dir /path/with/more/space
   ```

4. **Use minimal installation:**

   ```bash
   python scripts/install.py --minimal
   ```

### Permission Errors

**Symptoms:**

- "Permission denied" during installation
- Cannot write to directory

**Solutions:**

1. **Check directory permissions:**

   ```bash
   ls -la /path/to/directory
   ```

2. **Create directory with correct permissions:**

   ```bash
   mkdir -p ~/spyglass_data
   chmod 755 ~/spyglass_data
   ```

3. **Don't use sudo with conda:**
   - Conda environments should be user-owned
   - Never run `sudo conda` or `sudo pip`

### Git Issues

**Symptoms:**

- Cannot clone repository
- Git not found

**Solutions:**

1. **Install git:**

    === "maxOS"
        ```bash
       xcode-select --install
       ```

    === "Windows"
        ```powershell
       choco install git
       ```

    === "Linux - Debian/Ubuntu"
       ```bash
       sudo apt-get update -y
       sudo apt-get install git -y
       ```

    === "Linux - CentOS/RHEL"
       ```bash
       sudo yum install git -y
       ```

2. **Clone with HTTPS instead of SSH:**

   ```bash
   git clone https://github.com/LorenFrankLab/spyglass.git
   ```

## Platform-Specific Issues

=== "maxOS"

    **Issue: Xcode Command Line Tools missing**

    ```bash
    xcode-select --install
    ```

    **Issue: Homebrew conflicts**

    ```bash
    # Use conda-installed tools instead of homebrew
    conda activate spyglass
    which python  # Should show conda path
    ```

=== "Linux"

    **Issue: Missing system libraries**

    ```bash
    # Ubuntu/Debian
    sudo apt-get install build-essential libhdf5-dev

    # CentOS/RHEL
    sudo yum groupinstall "Development Tools"
    sudo yum install hdf5-devel
    ```

    **Issue: Docker permissions**

    ```bash
    sudo usermod -aG docker $USER
    # Log out and back in
    ```

=== "Windows (WSL)"

    **Issue: WSL not set up**

    ```bash
    # Install WSL 2 from PowerShell (admin):
    wsl --install
    ```

    **Issue: Docker Desktop integration**

    - Enable WSL 2 integration in Docker Desktop settings
    - Ensure Docker is running before installation

## Still Having Issues?

1. **Check [GitHub Issues:](https://github.com/LorenFrankLab/spyglass/issues)**

2. **Ask for Help:**
   - Include output from `python scripts/validate.py`
   - Include relevant error messages
   - Mention your OS and Python version

3. **Manual Installation:**
   See `docs/DATABASE.md` and main documentation for manual setup steps

## Reset and Start Fresh

If all else fails, completely reset your installation:

```bash
# Remove conda environment
conda env remove -n spyglass

# Remove configuration files
rm ~/.datajoint_config.json
rm ./dj_local_conf.json
rm -rf ~/spyglass_data # Delete all Spyglass data!

# Remove Docker container
docker rm -f spyglass-db # This will delete all data in the container!

# Start fresh
git clone https://github.com/LorenFrankLab/spyglass.git
cd spyglass
python scripts/install.py
```

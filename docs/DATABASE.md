# Spyglass Database Setup Guide

Spyglass requires a MySQL database backend for storing experimental data and analysis results. This guide covers all setup options from quick local development to production deployments.

## Quick Start (Recommended)

The easiest way to set up a database is using the installer with Docker Compose:

```bash
cd spyglass
python scripts/install.py
# Choose option 1 (Docker Compose) when prompted
```

This automatically:
- Pulls the MySQL 8.0 Docker image
- Creates and starts a container named `spyglass-db`
- Waits for MySQL to be ready
- Creates configuration file with credentials

**Or use Docker Compose directly:**
```bash
cd spyglass
docker compose up -d
```

## Setup Options

### Option 1: Docker Compose (Recommended for Local Development)

**Pros:**
- One-command setup (~2 minutes)
- Infrastructure as code (version controlled)
- Easy to customize via .env file
- Industry-standard tool
- Persistent data storage
- Health checks built-in

**Cons:**
- Requires Docker Desktop with Compose plugin
- Uses system resources when running

#### Prerequisites

1. **Install Docker Desktop:**
   - macOS: https://docs.docker.com/desktop/install/mac-install/
   - Windows: https://docs.docker.com/desktop/install/windows-install/
   - Linux: https://docs.docker.com/desktop/install/linux-install/

2. **Start Docker Desktop** and ensure it's running

3. **Verify Compose is available:**
   ```bash
   docker compose version
   # Should show: Docker Compose version v2.x.x
   ```

#### Setup

**Using installer (recommended):**
```bash
python scripts/install.py --docker  # Will auto-detect and use Compose
```

**Using Docker Compose directly:**
```bash
# From spyglass repository root
docker compose up -d
```

The default configuration uses:
- Port: 3306
- Password: tutorial
- Container name: spyglass-db
- Persistent storage: spyglass-db-data volume

#### Customization (Optional)

Create a `.env` file to customize settings:

```bash
# Copy example
cp .env.example .env

# Edit settings
nano .env
```

Available options:
```bash
# Change port if 3306 is in use
MYSQL_PORT=3307

# Change root password (for production)
MYSQL_ROOT_PASSWORD=your-secure-password

# Use different MySQL version
MYSQL_IMAGE=datajoint/mysql:8.4
```

**Important:** If you change port or password, update your DataJoint config accordingly.

#### Management

**Start/stop services:**
```bash
# Start
docker compose up -d

# Stop (keeps data)
docker compose stop

# Stop and remove containers (keeps data)
docker compose down

# Stop and remove everything including data
docker compose down -v  # WARNING: Deletes all data!
```

**View logs:**
```bash
docker compose logs mysql
docker compose logs -f mysql  # Follow mode
```

**Check status:**
```bash
docker compose ps
```

**Access MySQL shell:**
```bash
docker compose exec mysql mysql -uroot -ptutorial
```

**Restart services:**
```bash
docker compose restart
```

### Option 2: Remote Database (Lab/Cloud Setup)

**Pros:**
- Shared across team members
- Production-ready
- Professional backup/monitoring
- Persistent storage

**Cons:**
- Requires existing MySQL server
- Network configuration needed
- May need VPN/SSH tunnel

#### Prerequisites

- MySQL 8.0+ server accessible over network
- Database credentials (host, port, user, password)
- Firewall rules allowing connection

#### Setup

**Using installer (interactive):**
```bash
python scripts/install.py --remote
# Enter connection details when prompted
```

**Using installer (non-interactive for automation):**
```bash
# Using CLI arguments
python scripts/install.py --remote \
  --db-host db.mylab.edu \
  --db-user myusername \
  --db-password mypassword

# Using environment variables (recommended for CI/CD)
export SPYGLASS_DB_PASSWORD=mypassword
python scripts/install.py --remote \
  --db-host db.mylab.edu \
  --db-user myusername
```

**Manual configuration:**

Create `~/.datajoint_config.json`:
```json
{
  "database.host": "db.mylab.edu",
  "database.port": 3306,
  "database.user": "myusername",
  "database.password": "mypassword",
  "database.use_tls": true
}
```

**Test connection:**
```python
import datajoint as dj
dj.conn().ping()  # Should succeed
```

#### SSH Tunnel (For Remote Access)

If database is behind firewall, use SSH tunnel:

```bash
# Create tunnel (keep running in terminal)
ssh -L 3306:localhost:3306 user@remote-server

# In separate terminal, configure as localhost
cat > ~/.datajoint_config.json << EOF
{
  "database.host": "localhost",
  "database.port": 3306,
  "database.user": "root",
  "database.password": "password",
  "database.use_tls": false
}
EOF
```

Or use autossh for persistent tunnel:
```bash
autossh -M 0 -L 3306:localhost:3306 user@remote-server
```

### Option 3: Local MySQL Installation

**Pros:**
- No Docker required
- Direct system integration
- Full control over configuration

**Cons:**
- More complex setup
- Platform-specific installation
- Harder to reset/clean

#### macOS (Homebrew)

```bash
# Install MySQL
brew install mysql

# Start MySQL service
brew services start mysql

# Secure installation
mysql_secure_installation

# Create user
mysql -uroot -p
```

In MySQL shell:
```sql
CREATE USER 'spyglass'@'localhost' IDENTIFIED BY 'spyglass_password';
GRANT ALL PRIVILEGES ON *.* TO 'spyglass'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

Configure DataJoint:
```json
{
  "database.host": "localhost",
  "database.port": 3306,
  "database.user": "spyglass",
  "database.password": "spyglass_password",
  "database.use_tls": false
}
```

#### Linux (Ubuntu/Debian)

```bash
# Install MySQL
sudo apt-get update
sudo apt-get install mysql-server

# Start service
sudo systemctl start mysql
sudo systemctl enable mysql

# Secure installation
sudo mysql_secure_installation

# Create user
sudo mysql
```

In MySQL shell:
```sql
CREATE USER 'spyglass'@'localhost' IDENTIFIED BY 'spyglass_password';
GRANT ALL PRIVILEGES ON *.* TO 'spyglass'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

#### Windows

1. Download MySQL Installer: https://dev.mysql.com/downloads/installer/
2. Run installer and select "Developer Default"
3. Follow setup wizard
4. Create spyglass user with full privileges

## Configuration Reference

### DataJoint Configuration File

Location: `~/.datajoint_config.json`

**Full configuration example:**
```json
{
  "database.host": "localhost",
  "database.port": 3306,
  "database.user": "root",
  "database.password": "tutorial",
  "database.use_tls": false,
  "database.charset": "utf8mb4",
  "connection.init_function": null,
  "loglevel": "INFO",
  "safemode": true,
  "fetch_format": "array"
}
```

**Key settings:**

- `database.host`: MySQL server hostname or IP
- `database.port`: MySQL port (default: 3306)
- `database.user`: MySQL username
- `database.password`: MySQL password
- `database.use_tls`: Use TLS/SSL encryption (recommended for remote)

### TLS/SSL Configuration

**When to use TLS:**
- ✅ Remote database connections
- ✅ Production environments
- ✅ When connecting over untrusted networks
- ❌ localhost connections
- ❌ Docker containers on same machine

**Enable TLS:**
```json
{
  "database.use_tls": true
}
```

**Custom certificate:**
```json
{
  "database.use_tls": {
    "ssl": {
      "ca": "/path/to/ca-cert.pem",
      "cert": "/path/to/client-cert.pem",
      "key": "/path/to/client-key.pem"
    }
  }
}
```

## Security Best Practices

### Development

For local development, simple credentials are acceptable:
- User: `root` or dedicated user
- Password: Simple but unique
- TLS: Disabled for localhost

### Production

For shared/production databases:

1. **Strong passwords:**
   ```bash
   # Generate secure password
   openssl rand -base64 32
   ```

2. **User permissions:**
   ```sql
   -- Create user with specific database access
   CREATE USER 'spyglass'@'%' IDENTIFIED BY 'strong_password';
   GRANT ALL PRIVILEGES ON spyglass_*.* TO 'spyglass'@'%';
   FLUSH PRIVILEGES;
   ```

3. **Enable TLS:**
   ```json
   {
     "database.use_tls": true
   }
   ```

4. **Network security:**
   - Use firewall rules
   - Consider VPN for remote access
   - Use SSH tunnels when appropriate

5. **Credential management:**
   - Never commit config files to git
   - Use environment variables for CI/CD
   - Consider secrets management tools

### File Permissions

Protect configuration file:
```bash
chmod 600 ~/.datajoint_config.json
```

## Multi-User Setup

For lab environments with shared database:

### Server-Side Setup

```sql
-- Create database prefix for lab
CREATE DATABASE spyglass_common;

-- Create users
CREATE USER 'alice'@'%' IDENTIFIED BY 'alice_password';
CREATE USER 'bob'@'%' IDENTIFIED BY 'bob_password';

-- Grant permissions
GRANT ALL PRIVILEGES ON spyglass_*.* TO 'alice'@'%';
GRANT ALL PRIVILEGES ON spyglass_*.* TO 'bob'@'%';
FLUSH PRIVILEGES;
```

### Client-Side Setup

Each user creates their own config:

**Alice's config:**
```json
{
  "database.host": "lab-db.university.edu",
  "database.user": "alice",
  "database.password": "alice_password",
  "database.use_tls": true
}
```

**Bob's config:**
```json
{
  "database.host": "lab-db.university.edu",
  "database.user": "bob",
  "database.password": "bob_password",
  "database.use_tls": true
}
```

## Troubleshooting

### Cannot Connect

**Check MySQL is running:**
```bash
# Docker
docker ps | grep spyglass-db

# System service (Linux)
systemctl status mysql

# Homebrew (macOS)
brew services list | grep mysql
```

**Test connection:**
```bash
# With mysql client
mysql -h HOST -P PORT -u USER -p

# With Python
python -c "import datajoint as dj; dj.conn().ping()"
```

### Permission Denied

```sql
-- Grant missing privileges
GRANT ALL PRIVILEGES ON *.* TO 'user'@'host';
FLUSH PRIVILEGES;
```

### Port Already in Use

```bash
# Find what's using port 3306
lsof -i :3306
netstat -an | grep 3306

# Use different port
docker run -p 3307:3306 ...
# Update config with port 3307
```

### TLS Errors

```python
# Disable TLS for localhost
config = {
    "database.host": "localhost",
    "database.use_tls": False
}
```

For more troubleshooting help, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Advanced Topics

### Database Backup

**Docker database:**
```bash
# Backup
docker exec spyglass-db mysqldump -uroot -ptutorial --all-databases > backup.sql

# Restore
docker exec -i spyglass-db mysql -uroot -ptutorial < backup.sql
```

**System MySQL:**
```bash
# Backup
mysqldump -u USER -p --all-databases > backup.sql

# Restore
mysql -u USER -p < backup.sql
```

### Performance Tuning

**Increase buffer pool (Docker):**
```bash
docker run -d \
  --name spyglass-db \
  -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=tutorial \
  datajoint/mysql:8.0 \
  --innodb-buffer-pool-size=2G
```

**Optimize tables:**
```sql
OPTIMIZE TABLE tablename;
```

### Migration

**Moving from Docker to Remote:**
1. Backup Docker database
2. Restore to remote server
3. Update config to point to remote
4. Test connection

**Example:**
```bash
# Backup from Docker
docker exec spyglass-db mysqldump -uroot -ptutorial --all-databases > backup.sql

# Restore to remote
mysql -h remote-host -u user -p < backup.sql

# Update config
cat > ~/.datajoint_config.json << EOF
{
  "database.host": "remote-host",
  "database.user": "user",
  "database.password": "password",
  "database.use_tls": true
}
EOF
```

## Getting Help

- **Issues:** https://github.com/LorenFrankLab/spyglass/issues
- **Docs:** See main Spyglass documentation
- **DataJoint:** https://docs.datajoint.org/

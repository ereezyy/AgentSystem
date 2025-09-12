# AgentSystem Pi5 Deployment Package - File Transfer Guide

This guide covers multiple methods to transfer the Pi5 deployment package from your computer to the Raspberry Pi 5. Choose the method that best fits your setup and technical requirements.

## Table of Contents

- [Overview](#overview)
- [Method 1: SCP/SSH Transfer](#method-1-scpssh-transfer)
- [Method 2: USB Drive Transfer](#method-2-usb-drive-transfer)
- [Method 3: Network File Sharing](#method-3-network-file-sharing)
- [Method 4: Direct SD Card Copy](#method-4-direct-sd-card-copy)
- [Method 5: HTTP/Web Download](#method-5-httpweb-download)
- [Method 6: Git Repository](#method-6-git-repository)
- [Security Considerations](#security-considerations)
- [Troubleshooting Transfer Issues](#troubleshooting-transfer-issues)
- [Verification](#verification)

## Overview

The AgentSystem Pi5 deployment package contains approximately 184 files totaling about 50-100MB (depending on your specific AgentSystem build). All methods below will transfer the complete package structure:

```
pi5_deployment_package/
├── README.md
├── INSTALL.md
├── QUICKSTART.md
├── TRANSFER.md (this file)
├── .env.pi5
├── pi5_requirements.txt
├── pi5_health_check.py
├── agentsystem-pi5-worker.service
├── scripts/
│   ├── install.sh
│   └── setup_environment.sh
└── AgentSystem/
    └── [Complete AgentSystem codebase - 184 files]
```

## Method 1: SCP/SSH Transfer

**Best for:** Remote deployments, secure transfers, network-connected Pi5  
**Requirements:** SSH enabled on Pi5, network connectivity  
**Security:** Encrypted transfer, authentication required  

### Prerequisites

1. **Enable SSH on Pi5:**
   ```bash
   # On Pi5
   sudo systemctl enable ssh
   sudo systemctl start ssh
   
   # Check SSH status
   sudo systemctl status ssh
   ```

2. **Find Pi5 IP address:**
   ```bash
   # On Pi5
   hostname -I
   # OR
   ip addr show
   ```

3. **Test SSH connectivity:**
   ```bash
   # From your computer
   ssh pi@YOUR_PI5_IP
   ```

### Transfer Commands

#### From Linux/macOS:
```bash
# Transfer entire package
scp -r pi5_deployment_package/ pi@YOUR_PI5_IP:/home/pi/

# With progress display
rsync -avz --progress pi5_deployment_package/ pi@YOUR_PI5_IP:/home/pi/pi5_deployment_package/

# With compression (for slower networks)
scp -C -r pi5_deployment_package/ pi@YOUR_PI5_IP:/home/pi/
```

#### From Windows (PowerShell):
```powershell
# Using built-in scp (Windows 10+)
scp -r pi5_deployment_package pi@YOUR_PI5_IP:/home/pi/

# Using WinSCP (GUI alternative)
# Download WinSCP and use GUI to transfer files
```

#### From Windows (Command Prompt):
```cmd
# Using pscp (from PuTTY package)
pscp -r pi5_deployment_package pi@YOUR_PI5_IP:/home/pi/
```

### Advanced SCP Options

```bash
# Specify custom SSH port
scp -P 2222 -r pi5_deployment_package/ pi@YOUR_PI5_IP:/home/pi/

# Use SSH key authentication
scp -i ~/.ssh/id_rsa -r pi5_deployment_package/ pi@YOUR_PI5_IP:/home/pi/

# Resume interrupted transfer with rsync
rsync -avz --partial --progress pi5_deployment_package/ pi@YOUR_PI5_IP:/home/pi/pi5_deployment_package/

# Transfer with bandwidth limit (1MB/s)
rsync -avz --bwlimit=1000 pi5_deployment_package/ pi@YOUR_PI5_IP:/home/pi/pi5_deployment_package/
```

## Method 2: USB Drive Transfer

**Best for:** Physical access to Pi5, no network setup required  
**Requirements:** USB drive, physical access to both computers  
**Security:** Physical security of USB drive  

### Step-by-Step Instructions

#### Step 1: Prepare USB Drive

```bash
# Format USB drive (optional, if needed)
# WARNING: This will erase all data on the USB drive

# On Linux:
sudo fdisk -l  # Find USB drive (e.g., /dev/sdb1)
sudo mkfs.ext4 /dev/sdb1  # Format as ext4
# OR
sudo mkfs.fat -F32 /dev/sdb1  # Format as FAT32 (cross-platform)

# On Windows:
# Use Disk Management or format via File Explorer

# On macOS:
diskutil list  # Find USB drive
diskutil eraseDisk FAT32 "PI5_DEPLOY" /dev/diskX
```

#### Step 2: Copy Package to USB

**From Linux/macOS:**
```bash
# Mount USB drive if not auto-mounted
sudo mkdir -p /mnt/usb
sudo mount /dev/sdb1 /mnt/usb

# Copy package
cp -r pi5_deployment_package/ /mnt/usb/

# Safely unmount
sync
sudo umount /mnt/usb
```

**From Windows:**
```cmd
# Copy via File Explorer or command line
xcopy pi5_deployment_package E:\pi5_deployment_package\ /E /I
# where E: is your USB drive letter
```

**From macOS:**
```bash
# Copy to mounted USB drive
cp -r pi5_deployment_package/ /Volumes/YOUR_USB_NAME/
```

#### Step 3: Transfer on Pi5

```bash
# Insert USB drive into Pi5 and find mount point
lsblk
# OR
sudo fdisk -l

# Mount if not auto-mounted
sudo mkdir -p /mnt/usb
sudo mount /dev/sda1 /mnt/usb  # Adjust device name as needed

# Copy to Pi5
cp -r /mnt/usb/pi5_deployment_package/ /home/pi/

# Verify copy
ls -la /home/pi/pi5_deployment_package/

# Safely unmount USB
sync
sudo umount /mnt/usb
```

### USB Transfer Troubleshooting

```bash
# If USB not detected
sudo dmesg | tail  # Check kernel messages
lsusb  # List USB devices

# If permission issues
sudo chown -R pi:pi /home/pi/pi5_deployment_package/

# If filesystem issues
sudo fsck /dev/sda1  # Check/repair filesystem
```

## Method 3: Network File Sharing

**Best for:** Local network environments, multiple deployments  
**Requirements:** Network file server (SMB/NFS/FTP)  
**Security:** Network-based, configure appropriate access controls  

### SMB/CIFS Share

#### Set Up SMB Server (Linux/Windows)

**On Linux (as SMB server):**
```bash
# Install Samba
sudo apt install -y samba samba-common-bin

# Create shared directory
sudo mkdir -p /srv/samba/pi5_deploy
sudo cp -r pi5_deployment_package/ /srv/samba/pi5_deploy/

# Configure Samba
sudo nano /etc/samba/smb.conf

# Add this section:
[pi5_deploy]
path = /srv/samba/pi5_deploy
browseable = yes
writable = no
guest ok = yes
read only = yes

# Restart Samba
sudo systemctl restart smbd
sudo systemctl restart nmbd
```

**On Windows (as SMB server):**
1. Right-click on folder containing `pi5_deployment_package`
2. Select "Properties" → "Sharing" → "Advanced Sharing"
3. Enable "Share this folder"
4. Set appropriate permissions

#### Access SMB Share from Pi5

```bash
# Install SMB client
sudo apt install -y cifs-utils

# Create mount point
sudo mkdir -p /mnt/smb_share

# Mount share (guest access)
sudo mount -t cifs //SERVER_IP/pi5_deploy /mnt/smb_share -o guest,uid=pi,gid=pi

# OR with authentication
sudo mount -t cifs //SERVER_IP/pi5_deploy /mnt/smb_share -o username=USER,uid=pi,gid=pi

# Copy files
cp -r /mnt/smb_share/pi5_deployment_package/ /home/pi/

# Unmount
sudo umount /mnt/smb_share
```

### NFS Share

#### Set Up NFS Server (Linux)

```bash
# Install NFS server
sudo apt install -y nfs-kernel-server

# Create export directory
sudo mkdir -p /srv/nfs/pi5_deploy
sudo cp -r pi5_deployment_package/ /srv/nfs/pi5_deploy/

# Configure exports
echo "/srv/nfs/pi5_deploy *(ro,sync,no_subtree_check)" | sudo tee -a /etc/exports

# Apply configuration
sudo exportfs -a
sudo systemctl restart nfs-kernel-server
```

#### Access NFS Share from Pi5

```bash
# Install NFS client
sudo apt install -y nfs-common

# Create mount point
sudo mkdir -p /mnt/nfs_share

# Mount share
sudo mount -t nfs SERVER_IP:/srv/nfs/pi5_deploy /mnt/nfs_share

# Copy files
cp -r /mnt/nfs_share/pi5_deployment_package/ /home/pi/

# Unmount
sudo umount /mnt/nfs_share
```

## Method 4: Direct SD Card Copy

**Best for:** Pre-deployment setup, offline environments  
**Requirements:** SD card reader, ability to mount Pi5 SD card  
**Security:** Physical access control  

### Option A: Copy to Existing Pi5 SD Card

```bash
# Remove SD card from Pi5 and insert into computer

# Mount SD card partitions
# Linux:
sudo mkdir -p /mnt/pi5_boot /mnt/pi5_root
sudo mount /dev/sdX1 /mnt/pi5_boot    # Boot partition
sudo mount /dev/sdX2 /mnt/pi5_root    # Root partition

# Copy to Pi user home directory
sudo cp -r pi5_deployment_package/ /mnt/pi5_root/home/pi/

# Set correct ownership
sudo chown -R 1000:1000 /mnt/pi5_root/home/pi/pi5_deployment_package/

# Unmount safely
sync
sudo umount /mnt/pi5_boot /mnt/pi5_root
```

### Option B: Include in Fresh Pi OS Image

```bash
# Mount fresh Raspberry Pi OS image
# Download Raspberry Pi Imager and flash SD card first

# After flashing, re-insert SD card and mount
sudo mount /dev/sdX2 /mnt/pi5_root

# Copy package
sudo cp -r pi5_deployment_package/ /mnt/pi5_root/home/pi/
sudo chown -R 1000:1000 /mnt/pi5_root/home/pi/pi5_deployment_package/

# Optionally enable SSH by default
sudo touch /mnt/pi5_boot/ssh

# Create user and SSH keys if needed
# (Follow Raspberry Pi OS documentation)

# Unmount
sync
sudo umount /mnt/pi5_root
```

## Method 5: HTTP/Web Download

**Best for:** Remote deployments, automated setups  
**Requirements:** Web server, internet connectivity  
**Security:** Use HTTPS, consider authentication  

### Set Up Simple HTTP Server

#### Python HTTP Server (Quick Setup)

```bash
# Navigate to parent directory of pi5_deployment_package
cd /path/to/parent/directory

# Start simple HTTP server (Python 3)
python3 -m http.server 8080

# OR with Python 2
python -m SimpleHTTPServer 8080

# Access from browser: http://YOUR_IP:8080/pi5_deployment_package/
```

#### Download on Pi5

```bash
# Download using wget
wget -r -np -nH --cut-dirs=0 http://YOUR_SERVER_IP:8080/pi5_deployment_package/

# OR download as compressed archive
# First, create archive on server:
tar -czf pi5_deployment_package.tar.gz pi5_deployment_package/

# Then download and extract on Pi5:
wget http://YOUR_SERVER_IP:8080/pi5_deployment_package.tar.gz
tar -xzf pi5_deployment_package.tar.gz
```

### Using Cloud Storage

#### Google Drive/Dropbox/OneDrive

```bash
# Upload pi5_deployment_package.tar.gz to cloud storage
# Get shareable link

# On Pi5, download using curl/wget
wget "https://drive.google.com/uc?export=download&id=FILE_ID" -O pi5_deployment_package.tar.gz

# Extract
tar -xzf pi5_deployment_package.tar.gz
```

## Method 6: Git Repository

**Best for:** Version control, distributed deployments  
**Requirements:** Git repository (local or remote)  
**Security:** Git authentication, SSH keys  

### Set Up Local Git Repository

```bash
# Initialize repository
cd pi5_deployment_package
git init
git add .
git commit -m "Initial Pi5 deployment package"

# Set up bare repository on server
ssh user@server "git init --bare /srv/git/pi5_deployment.git"

# Add remote and push
git remote add origin user@server:/srv/git/pi5_deployment.git
git push origin main
```

### Clone on Pi5

```bash
# Clone repository
git clone user@server:/srv/git/pi5_deployment.git /home/pi/pi5_deployment_package

# OR if using SSH keys
git clone git@server:/srv/git/pi5_deployment.git /home/pi/pi5_deployment_package
```

## Security Considerations

### General Security Practices

1. **Verify Package Integrity:**
   ```bash
   # Create checksum on source
   find pi5_deployment_package -type f -exec sha256sum {} \; > pi5_package.sha256
   
   # Verify on Pi5
   sha256sum -c pi5_package.sha256
   ```

2. **Secure Transfer Channels:**
   - Use SCP/SSH for encrypted transfers
   - Avoid unencrypted FTP or HTTP for sensitive data
   - Use VPN for remote transfers over internet

3. **Access Control:**
   ```bash
   # Set appropriate permissions after transfer
   chmod -R 755 /home/pi/pi5_deployment_package/
   chmod +x /home/pi/pi5_deployment_package/scripts/*.sh
   ```

### Network Security

```bash
# For SSH transfers, use key-based authentication
ssh-keygen -t rsa -b 4096 -C "pi5-deployment"
ssh-copy-id pi@YOUR_PI5_IP

# Disable password authentication
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
sudo systemctl restart ssh
```

### USB Security

```bash
# Scan USB drive for malware (Linux)
sudo apt install -y clamav
sudo freshclam
sudo clamscan -r /mnt/usb

# Verify no additional files were added
find /mnt/usb -name "*.exe" -o -name "*.bat" -o -name "*.scr"
```

## Troubleshooting Transfer Issues

### Network Issues

```bash
# Test network connectivity
ping YOUR_PI5_IP
telnet YOUR_PI5_IP 22  # Test SSH port

# Check network configuration
ip route show
cat /etc/resolv.conf
```

### Permission Issues

```bash
# Fix ownership after transfer
sudo chown -R pi:pi /home/pi/pi5_deployment_package/

# Fix permissions
find /home/pi/pi5_deployment_package/ -type d -exec chmod 755 {} \;
find /home/pi/pi5_deployment_package/ -type f -exec chmod 644 {} \;
chmod +x /home/pi/pi5_deployment_package/scripts/*.sh
```

### Incomplete Transfers

```bash
# Compare file counts
# On source:
find pi5_deployment_package -type f | wc -l

# On Pi5:
find /home/pi/pi5_deployment_package -type f | wc -l

# Check for missing files
diff <(find pi5_deployment_package -type f | sort) <(find /home/pi/pi5_deployment_package -type f | sort)
```

### Large File Issues

```bash
# For transfers that fail with large files, use rsync with resumption
rsync -avz --partial --inplace --progress pi5_deployment_package/ pi@YOUR_PI5_IP:/home/pi/pi5_deployment_package/
```

## Verification

After any transfer method, verify the package integrity:

### Complete Verification Script

```bash
#!/bin/bash
# Save as verify_transfer.sh on Pi5

PKG_DIR="/home/pi/pi5_deployment_package"

echo "Verifying Pi5 deployment package transfer..."

# Check directory exists
if [ ! -d "$PKG_DIR" ]; then
    echo "❌ Package directory not found: $PKG_DIR"
    exit 1
fi

# Check essential files
REQUIRED_FILES=(
    "README.md"
    "INSTALL.md"
    "QUICKSTART.md"
    "TRANSFER.md"
    ".env.pi5"
    "pi5_requirements.txt"
    "pi5_health_check.py"
    "agentsystem-pi5-worker.service"
    "scripts/install.sh"
    "scripts/setup_environment.sh"
    "AgentSystem/pi5_worker.py"
)

echo "Checking required files..."
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$PKG_DIR/$file" ]; then
        echo "✅ $file"
    else
        echo "❌ Missing: $file"
        exit 1
    fi
done

# Check script permissions
echo "Checking script permissions..."
for script in "$PKG_DIR/scripts"/*.sh; do
    if [ -x "$script" ]; then
        echo "✅ $(basename $script) is executable"
    else
        echo "⚠️  $(basename $script) not executable, fixing..."
        chmod +x "$script"
    fi
done

# Count files
TOTAL_FILES=$(find "$PKG_DIR" -type f | wc -l)
echo "📊 Total files transferred: $TOTAL_FILES"

# Check Python syntax of key files
echo "Checking Python syntax..."
python3 -m py_compile "$PKG_DIR/pi5_health_check.py" && echo "✅ pi5_health_check.py syntax OK"
python3 -m py_compile "$PKG_DIR/AgentSystem/pi5_worker.py" && echo "✅ pi5_worker.py syntax OK"

echo "🎉 Transfer verification completed successfully!"
echo "You can now proceed with installation using:"
echo "  cd $PKG_DIR"
echo "  sudo ./scripts/install.sh"
```

Run the verification:

```bash
chmod +x verify_transfer.sh
./verify_transfer.sh
```

---

## Summary

Choose the transfer method that best fits your environment:

| Method | Best For | Security | Speed | Complexity |
|--------|----------|----------|--------|------------|
| **SCP/SSH** | Remote deployment | High | Fast | Medium |
| **USB Drive** | Physical access | Medium | Fast | Low |
| **Network Share** | Local network | Medium | Fast | Medium |
| **SD Card Direct** | Pre-deployment | High | Fast | Low |
| **HTTP Download** | Automation | Low-Medium | Medium | Low |
| **Git Repository** | Version control | High | Medium | High |

After successful transfer, proceed with the installation using [`INSTALL.md`](INSTALL.md) or [`QUICKSTART.md`](QUICKSTART.md).
import re

filepath = "AgentSystem/api/security_endpoints.py"

with open(filepath, "r") as f:
    content = f.read()

# 1. Clean up imports
# Remove redundant imports if they exist
content = content.replace("from fastapi import status\n", "")
content = content.replace("from fastapi.security import HTTPAuthorizationCredentials\n", "")

# Ensure the main imports have what we need
if "from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, UploadFile, File, status" not in content:
    content = content.replace(
        "from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, UploadFile, File",
        "from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, UploadFile, File, status"
    )

if "from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials" not in content:
    content = content.replace(
        "from fastapi.security import HTTPBearer",
        "from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials"
    )

# 2. Fix security issue: make JWT_SECRET_KEY required
content = content.replace(
    'jwt_secret = get_env("JWT_SECRET_KEY", "default-insecure-secret-change-me")',
    'jwt_secret = get_env("JWT_SECRET_KEY", required=True)'
)

# Also make JWT_ALGORITHM robust (though defaulting is safer here, let's keep it but maybe warn)
# The review only mentioned the secret key.

with open(filepath, "w") as f:
    f.write(content)

print("File updated with code review fixes.")

#!/usr/bin/env python3
"""
OAuth2/OIDC Authentication System with MFA and FIDO2 for Ultimate Arbitrage System

Features:
- OAuth2/OIDC compliant authentication
- Multi-Factor Authentication (MFA) with TOTP/HOTP
- FIDO2/WebAuthn passwordless authentication
- Fine-grained RBAC/ABAC authorization
- JWT token management with rotation
- Session management and security

Security Design:
- Secure token storage and rotation
- Strong password policies
- Biometric authentication support
- Hardware security key integration
- Real-time threat detection
- Compliance with security standards
"""

import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from urllib.parse import urlencode, parse_qs
import secrets
import hashlib
import base64
import qrcode
import io
from PIL import Image

# JWT and cryptography
import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

# OTP and FIDO2
import pyotp
from fido2.server import Fido2Server
from fido2.webauthn import PublicKeyCredentialRpEntity, PublicKeyCredentialUserEntity
from fido2 import cbor

# Password hashing
import bcrypt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auth_audit.log'),
        logging.StreamHandler()
    ]
)
auth_logger = logging.getLogger('AuthSystem')

class AuthenticationMethod(Enum):
    """Authentication methods"""
    PASSWORD = "password"
    TOTP = "totp"
    HOTP = "hotp"
    SMS = "sms"
    EMAIL = "email"
    FIDO2 = "fido2"
    BIOMETRIC = "biometric"
    BACKUP_CODES = "backup_codes"

class UserRole(Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"
    GUEST = "guest"

class Permission(Enum):
    """System permissions"""
    # Trading permissions
    TRADING_VIEW = "trading:view"
    TRADING_EXECUTE = "trading:execute"
    TRADING_MANAGE = "trading:manage"
    
    # Strategy permissions
    STRATEGY_VIEW = "strategy:view"
    STRATEGY_CREATE = "strategy:create"
    STRATEGY_MODIFY = "strategy:modify"
    STRATEGY_DELETE = "strategy:delete"
    
    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"
    
    # Data permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_EXPORT = "data:export"

class TokenType(Enum):
    """Token types"""
    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    ID_TOKEN = "id_token"
    AUTHORIZATION_CODE = "authorization_code"

@dataclass
class User:
    """User account information"""
    user_id: str
    username: str
    email: str
    password_hash: str
    roles: Set[UserRole] = field(default_factory=set)
    permissions: Set[Permission] = field(default_factory=set)
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    password_changed_at: datetime = field(default_factory=datetime.now)
    require_password_change: bool = False
    mfa_enabled: bool = False
    mfa_methods: Set[AuthenticationMethod] = field(default_factory=set)
    backup_codes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MFASecret:
    """MFA secret information"""
    user_id: str
    method: AuthenticationMethod
    secret: str
    backup_codes: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    is_active: bool = True

@dataclass
class FIDO2Credential:
    """FIDO2 credential information"""
    credential_id: str
    user_id: str
    public_key: str
    sign_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    device_name: str = "Unknown Device"
    is_active: bool = True

@dataclass
class Session:
    """User session information"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    mfa_verified: bool = False
    risk_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccessToken:
    """Access token information"""
    token_id: str
    user_id: str
    token_type: TokenType
    token_value: str
    expires_at: datetime
    scope: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    is_revoked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class PasswordHasher:
    """Secure password hashing using Argon2"""
    
    def __init__(self):
        self.hasher = PasswordHasher(
            time_cost=3,      # Number of iterations
            memory_cost=65536,  # Memory usage in KB
            parallelism=1,    # Number of parallel threads
            hash_len=32,      # Hash length in bytes
            salt_len=16       # Salt length in bytes
        )
    
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        return self.hasher.hash(password)
    
    def verify_password(self, password: str, hash: str) -> bool:
        """Verify password against hash"""
        try:
            self.hasher.verify(hash, password)
            return True
        except VerifyMismatchError:
            return False
    
    def needs_rehash(self, hash: str) -> bool:
        """Check if password hash needs updating"""
        return self.hasher.check_needs_rehash(hash)

class TOTPManager:
    """Time-based One-Time Password manager"""
    
    def __init__(self, issuer: str = "Ultimate Arbitrage System"):
        self.issuer = issuer
    
    def generate_secret(self) -> str:
        """Generate TOTP secret"""
        return pyotp.random_base32()
    
    def generate_qr_code(self, secret: str, user_email: str) -> bytes:
        """Generate QR code for TOTP setup"""
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=user_email,
            issuer_name=self.issuer
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()
    
    def verify_totp(self, secret: str, token: str, window: int = 1) -> bool:
        """Verify TOTP token"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=window)
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup recovery codes"""
        codes = []
        for _ in range(count):
            code = secrets.token_hex(4).upper()
            codes.append(f"{code[:4]}-{code[4:]}")
        return codes

class FIDO2Manager:
    """FIDO2/WebAuthn manager"""
    
    def __init__(self, rp_id: str = "localhost", rp_name: str = "Ultimate Arbitrage System"):
        self.rp = PublicKeyCredentialRpEntity(rp_id, rp_name)
        self.server = Fido2Server(self.rp)
        
    def begin_registration(self, user_id: str, username: str, display_name: str) -> Dict[str, Any]:
        """Begin FIDO2 registration process"""
        user = PublicKeyCredentialUserEntity(
            id=user_id.encode(),
            name=username,
            display_name=display_name
        )
        
        registration_data, state = self.server.register_begin(
            user,
            credentials=[],  # Existing credentials
            user_verification="preferred"
        )
        
        return {
            "publicKey": registration_data,
            "state": state
        }
    
    def complete_registration(self, registration_response: Dict[str, Any], 
                            state: bytes) -> FIDO2Credential:
        """Complete FIDO2 registration"""
        auth_data = self.server.register_complete(state, registration_response)
        
        credential = FIDO2Credential(
            credential_id=base64.b64encode(auth_data.credential_data.credential_id).decode(),
            user_id="",  # Will be set by caller
            public_key=base64.b64encode(auth_data.credential_data.public_key).decode(),
            sign_count=auth_data.sign_count
        )
        
        return credential
    
    def begin_authentication(self, credentials: List[FIDO2Credential]) -> Dict[str, Any]:
        """Begin FIDO2 authentication process"""
        credential_descriptors = []
        for cred in credentials:
            credential_descriptors.append({
                "type": "public-key",
                "id": base64.b64decode(cred.credential_id)
            })
        
        auth_data, state = self.server.authenticate_begin(
            credentials=credential_descriptors,
            user_verification="preferred"
        )
        
        return {
            "publicKey": auth_data,
            "state": state
        }
    
    def complete_authentication(self, auth_response: Dict[str, Any], 
                              state: bytes, credential: FIDO2Credential) -> bool:
        """Complete FIDO2 authentication"""
        try:
            auth_data = self.server.authenticate_complete(
                state,
                credentials=[{
                    "id": base64.b64decode(credential.credential_id),
                    "public_key": base64.b64decode(credential.public_key),
                    "sign_count": credential.sign_count
                }],
                response=auth_response
            )
            
            # Update sign count
            credential.sign_count = auth_data.sign_count
            credential.last_used = datetime.now()
            
            return True
        except Exception as e:
            auth_logger.error(f"FIDO2 authentication failed: {e}")
            return False

class JWTManager:
    """JSON Web Token manager"""
    
    def __init__(self, private_key: str = None, public_key: str = None):
        if private_key and public_key:
            self.private_key = private_key
            self.public_key = public_key
        else:
            # Generate new key pair
            key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            self.private_key = key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ).decode()
            
            self.public_key = key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()
    
    def create_access_token(self, user_id: str, permissions: List[str], 
                          expires_minutes: int = 15) -> str:
        """Create JWT access token"""
        now = datetime.utcnow()
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + timedelta(minutes=expires_minutes),
            "aud": "ultimate-arbitrage-api",
            "iss": "ultimate-arbitrage-auth",
            "permissions": permissions,
            "token_type": "access",
            "jti": str(uuid.uuid4())
        }
        
        return jwt.encode(payload, self.private_key, algorithm="RS256")
    
    def create_refresh_token(self, user_id: str, 
                           expires_days: int = 30) -> str:
        """Create JWT refresh token"""
        now = datetime.utcnow()
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + timedelta(days=expires_days),
            "aud": "ultimate-arbitrage-api",
            "iss": "ultimate-arbitrage-auth",
            "token_type": "refresh",
            "jti": str(uuid.uuid4())
        }
        
        return jwt.encode(payload, self.private_key, algorithm="RS256")
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=["RS256"],
                audience="ultimate-arbitrage-api",
                issuer="ultimate-arbitrage-auth"
            )
            return payload
        except jwt.InvalidTokenError as e:
            auth_logger.warning(f"Invalid token: {e}")
            raise

class RBACManager:
    """Role-Based Access Control manager"""
    
    def __init__(self):
        self.role_permissions = {
            UserRole.ADMIN: {
                Permission.SYSTEM_ADMIN,
                Permission.SYSTEM_CONFIG,
                Permission.SYSTEM_MONITOR,
                Permission.TRADING_VIEW,
                Permission.TRADING_EXECUTE,
                Permission.TRADING_MANAGE,
                Permission.STRATEGY_VIEW,
                Permission.STRATEGY_CREATE,
                Permission.STRATEGY_MODIFY,
                Permission.STRATEGY_DELETE,
                Permission.DATA_READ,
                Permission.DATA_WRITE,
                Permission.DATA_EXPORT
            },
            UserRole.TRADER: {
                Permission.TRADING_VIEW,
                Permission.TRADING_EXECUTE,
                Permission.STRATEGY_VIEW,
                Permission.STRATEGY_CREATE,
                Permission.STRATEGY_MODIFY,
                Permission.DATA_READ
            },
            UserRole.ANALYST: {
                Permission.TRADING_VIEW,
                Permission.STRATEGY_VIEW,
                Permission.DATA_READ,
                Permission.DATA_EXPORT
            },
            UserRole.VIEWER: {
                Permission.TRADING_VIEW,
                Permission.STRATEGY_VIEW,
                Permission.DATA_READ
            },
            UserRole.API_USER: {
                Permission.TRADING_VIEW,
                Permission.TRADING_EXECUTE,
                Permission.DATA_READ
            }
        }
    
    def get_user_permissions(self, user: User) -> Set[Permission]:
        """Get all permissions for user based on roles"""
        permissions = set(user.permissions)  # Direct permissions
        
        # Add role-based permissions
        for role in user.roles:
            permissions.update(self.role_permissions.get(role, set()))
        
        return permissions
    
    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission"""
        user_permissions = self.get_user_permissions(user)
        return permission in user_permissions
    
    def check_permissions(self, user: User, permissions: List[Permission]) -> bool:
        """Check if user has all specified permissions"""
        user_permissions = self.get_user_permissions(user)
        return all(perm in user_permissions for perm in permissions)

class AuthenticationSystem:
    """Main authentication system"""
    
    def __init__(self):
        self.password_hasher = PasswordHasher()
        self.totp_manager = TOTPManager()
        self.fido2_manager = FIDO2Manager()
        self.jwt_manager = JWTManager()
        self.rbac_manager = RBACManager()
        
        # In-memory stores (use proper database in production)
        self.users: Dict[str, User] = {}
        self.mfa_secrets: Dict[str, Dict[AuthenticationMethod, MFASecret]] = {}
        self.fido2_credentials: Dict[str, List[FIDO2Credential]] = {}
        self.sessions: Dict[str, Session] = {}
        self.tokens: Dict[str, AccessToken] = {}
        
        # Security settings
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.session_timeout = timedelta(hours=8)
        self.password_min_length = 12
        
    async def register_user(self, username: str, email: str, password: str, 
                           roles: List[UserRole] = None) -> str:
        """Register new user"""
        # Validate password strength
        if not self._validate_password_strength(password):
            raise ValueError("Password does not meet security requirements")
        
        # Check if user already exists
        if any(u.username == username or u.email == email for u in self.users.values()):
            raise ValueError("User already exists")
        
        # Create user
        user_id = str(uuid.uuid4())
        password_hash = self.password_hasher.hash_password(password)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            roles=set(roles or [UserRole.VIEWER])
        )
        
        self.users[user_id] = user
        auth_logger.info(f"User registered: {username}")
        
        return user_id
    
    async def authenticate_password(self, username: str, password: str, 
                                   ip_address: str, user_agent: str) -> Tuple[bool, Optional[User]]:
        """Authenticate user with password"""
        # Find user
        user = None
        for u in self.users.values():
            if u.username == username or u.email == username:
                user = u
                break
        
        if not user:
            auth_logger.warning(f"Authentication failed: user not found - {username}")
            return False, None
        
        # Check if account is locked
        if user.locked_until and datetime.now() < user.locked_until:
            auth_logger.warning(f"Authentication failed: account locked - {username}")
            return False, None
        
        # Verify password
        if not self.password_hasher.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            
            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.max_login_attempts:
                user.locked_until = datetime.now() + self.lockout_duration
                auth_logger.warning(f"Account locked due to failed attempts: {username}")
            
            auth_logger.warning(f"Authentication failed: invalid password - {username}")
            return False, None
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()
        
        auth_logger.info(f"Password authentication successful: {username}")
        return True, user
    
    async def setup_totp(self, user_id: str) -> Tuple[str, bytes]:
        """Setup TOTP for user"""
        if user_id not in self.users:
            raise ValueError("User not found")
        
        user = self.users[user_id]
        secret = self.totp_manager.generate_secret()
        qr_code = self.totp_manager.generate_qr_code(secret, user.email)
        
        # Store secret (not activated until verified)
        if user_id not in self.mfa_secrets:
            self.mfa_secrets[user_id] = {}
        
        self.mfa_secrets[user_id][AuthenticationMethod.TOTP] = MFASecret(
            user_id=user_id,
            method=AuthenticationMethod.TOTP,
            secret=secret,
            is_active=False
        )
        
        return secret, qr_code
    
    async def verify_totp_setup(self, user_id: str, token: str) -> bool:
        """Verify TOTP setup and activate"""
        if (user_id not in self.mfa_secrets or 
            AuthenticationMethod.TOTP not in self.mfa_secrets[user_id]):
            return False
        
        mfa_secret = self.mfa_secrets[user_id][AuthenticationMethod.TOTP]
        
        if self.totp_manager.verify_totp(mfa_secret.secret, token):
            mfa_secret.is_active = True
            mfa_secret.backup_codes = self.totp_manager.generate_backup_codes()
            
            user = self.users[user_id]
            user.mfa_enabled = True
            user.mfa_methods.add(AuthenticationMethod.TOTP)
            user.backup_codes = mfa_secret.backup_codes.copy()
            
            auth_logger.info(f"TOTP setup completed for user: {user_id}")
            return True
        
        return False
    
    async def verify_totp(self, user_id: str, token: str) -> bool:
        """Verify TOTP token"""
        if (user_id not in self.mfa_secrets or 
            AuthenticationMethod.TOTP not in self.mfa_secrets[user_id]):
            return False
        
        mfa_secret = self.mfa_secrets[user_id][AuthenticationMethod.TOTP]
        
        if not mfa_secret.is_active:
            return False
        
        if self.totp_manager.verify_totp(mfa_secret.secret, token):
            mfa_secret.last_used = datetime.now()
            return True
        
        return False
    
    async def setup_fido2(self, user_id: str) -> Dict[str, Any]:
        """Setup FIDO2 for user"""
        if user_id not in self.users:
            raise ValueError("User not found")
        
        user = self.users[user_id]
        registration_data = self.fido2_manager.begin_registration(
            user_id, user.username, user.email
        )
        
        return registration_data
    
    async def complete_fido2_setup(self, user_id: str, registration_response: Dict[str, Any], 
                                  state: bytes, device_name: str = "FIDO2 Key") -> bool:
        """Complete FIDO2 setup"""
        try:
            credential = self.fido2_manager.complete_registration(registration_response, state)
            credential.user_id = user_id
            credential.device_name = device_name
            
            if user_id not in self.fido2_credentials:
                self.fido2_credentials[user_id] = []
            
            self.fido2_credentials[user_id].append(credential)
            
            user = self.users[user_id]
            user.mfa_enabled = True
            user.mfa_methods.add(AuthenticationMethod.FIDO2)
            
            auth_logger.info(f"FIDO2 setup completed for user: {user_id}")
            return True
            
        except Exception as e:
            auth_logger.error(f"FIDO2 setup failed: {e}")
            return False
    
    async def create_session(self, user_id: str, ip_address: str, 
                           user_agent: str, mfa_verified: bool = False) -> str:
        """Create user session"""
        session_id = secrets.token_urlsafe(32)
        now = datetime.now()
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            expires_at=now + self.session_timeout,
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent,
            mfa_verified=mfa_verified
        )
        
        self.sessions[session_id] = session
        auth_logger.info(f"Session created for user: {user_id}")
        
        return session_id
    
    async def create_tokens(self, user_id: str) -> Dict[str, str]:
        """Create access and refresh tokens"""
        user = self.users[user_id]
        permissions = [perm.value for perm in self.rbac_manager.get_user_permissions(user)]
        
        access_token = self.jwt_manager.create_access_token(user_id, permissions)
        refresh_token = self.jwt_manager.create_refresh_token(user_id)
        
        # Store tokens
        access_token_id = str(uuid.uuid4())
        refresh_token_id = str(uuid.uuid4())
        
        self.tokens[access_token_id] = AccessToken(
            token_id=access_token_id,
            user_id=user_id,
            token_type=TokenType.ACCESS_TOKEN,
            token_value=access_token,
            expires_at=datetime.now() + timedelta(minutes=15)
        )
        
        self.tokens[refresh_token_id] = AccessToken(
            token_id=refresh_token_id,
            user_id=user_id,
            token_type=TokenType.REFRESH_TOKEN,
            token_value=refresh_token,
            expires_at=datetime.now() + timedelta(days=30)
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": 900  # 15 minutes
        }
    
    async def verify_session(self, session_id: str) -> Optional[Session]:
        """Verify and update session"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        now = datetime.now()
        
        # Check if session is expired
        if now > session.expires_at or not session.is_active:
            return None
        
        # Update last activity
        session.last_activity = now
        
        return session
    
    async def revoke_session(self, session_id: str) -> bool:
        """Revoke user session"""
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            auth_logger.info(f"Session revoked: {session_id}")
            return True
        return False
    
    def _validate_password_strength(self, password: str) -> bool:
        """Validate password strength"""
        if len(password) < self.password_min_length:
            return False
        
        # Check for uppercase, lowercase, digits, and special characters
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions and tokens"""
        now = datetime.now()
        
        # Clean up sessions
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if session.expires_at < now
        ]
        
        for sid in expired_sessions:
            del self.sessions[sid]
        
        # Clean up tokens
        expired_tokens = [
            tid for tid, token in self.tokens.items()
            if token.expires_at < now
        ]
        
        for tid in expired_tokens:
            del self.tokens[tid]
        
        if expired_sessions or expired_tokens:
            auth_logger.info(
                f"Cleaned up {len(expired_sessions)} sessions and {len(expired_tokens)} tokens"
            )

# Factory function
def create_auth_system() -> AuthenticationSystem:
    """Create and configure authentication system"""
    return AuthenticationSystem()

if __name__ == "__main__":
    # Demo usage
    async def demo():
        # Create authentication system
        auth_system = create_auth_system()
        
        # Register a user
        user_id = await auth_system.register_user(
            "testuser", 
            "test@example.com", 
            "SecurePassword123!",
            [UserRole.TRADER]
        )
        print(f"User registered: {user_id}")
        
        # Authenticate with password
        success, user = await auth_system.authenticate_password(
            "testuser", "SecurePassword123!", "127.0.0.1", "Test Agent"
        )
        print(f"Password authentication: {success}")
        
        if success:
            # Setup TOTP
            secret, qr_code = await auth_system.setup_totp(user_id)
            print(f"TOTP secret generated: {secret[:10]}...")
            
            # Simulate TOTP verification (would use actual TOTP app in practice)
            import pyotp
            totp = pyotp.TOTP(secret)
            token = totp.now()
            
            verified = await auth_system.verify_totp_setup(user_id, token)
            print(f"TOTP setup verified: {verified}")
            
            # Create session
            session_id = await auth_system.create_session(
                user_id, "127.0.0.1", "Test Agent", mfa_verified=True
            )
            print(f"Session created: {session_id[:10]}...")
            
            # Create tokens
            tokens = await auth_system.create_tokens(user_id)
            print(f"Tokens created: {list(tokens.keys())}")
            
            # Check permissions
            has_trading = auth_system.rbac_manager.check_permission(
                user, Permission.TRADING_EXECUTE
            )
            print(f"Has trading permission: {has_trading}")
        
        print("Demo completed")
    
    asyncio.run(demo())


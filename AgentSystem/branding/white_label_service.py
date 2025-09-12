
"""
🎨 AgentSystem White-Label Branding Service
Complete branding customization system for enterprise clients
"""

import asyncio
import os
import json
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import hashlib
import secrets

import asyncpg
import aioredis
import aiofiles
from fastapi import HTTPException, UploadFile
from pydantic import BaseModel, Field, validator
from PIL import Image, ImageOps
import boto3
from jinja2 import Environment, DictLoader
import cssutils
from colorthief import ColorThief

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrandingTier(str, Enum):
    BASIC = "basic"          # Logo + primary color
    STANDARD = "standard"    # Logo, colors, fonts
    PREMIUM = "premium"      # Full customization
    ENTERPRISE = "enterprise" # Everything + custom domain

class AssetType(str, Enum):
    LOGO_PRIMARY = "logo_primary"
    LOGO_SECONDARY = "logo_secondary"
    LOGO_ICON = "logo_icon"
    FAVICON = "favicon"
    BACKGROUND_IMAGE = "background_image"
    EMAIL_HEADER = "email_header"
    LOADING_ANIMATION = "loading_animation"
    CUSTOM_CSS = "custom_css"

@dataclass
class BrandColors:
    """Brand color palette"""
    primary: str = "#007bff"
    secondary: str = "#6c757d"
    success: str = "#28a745"
    warning: str = "#ffc107"
    danger: str = "#dc3545"
    info: str = "#17a2b8"
    light: str = "#f8f9fa"
    dark: str = "#343a40"

    # Extended palette
    accent: str = "#6f42c1"
    muted: str = "#6c757d"

    def to_css_variables(self) -> str:
        """Convert to CSS custom properties"""
        css_vars = []
        for key, value in asdict(self).items():
            css_vars.append(f"--color-{key.replace('_', '-')}: {value};")
        return "\n".join(css_vars)

@dataclass
class BrandTypography:
    """Typography settings"""
    primary_font: str = "Inter, system-ui, sans-serif"
    secondary_font: str = "Inter, system-ui, sans-serif"
    monospace_font: str = "JetBrains Mono, Consolas, monospace"

    # Font sizes
    base_size: str = "16px"
    scale_ratio: float = 1.25

    # Font weights
    light_weight: int = 300
    normal_weight: int = 400
    medium_weight: int = 500
    semibold_weight: int = 600
    bold_weight: int = 700

    def to_css_variables(self) -> str:
        """Convert to CSS custom properties"""
        css_vars = [
            f"--font-primary: {self.primary_font};",
            f"--font-secondary: {self.secondary_font};",
            f"--font-monospace: {self.monospace_font};",
            f"--font-size-base: {self.base_size};",
            f"--font-scale-ratio: {self.scale_ratio};",
            f"--font-weight-light: {self.light_weight};",
            f"--font-weight-normal: {self.normal_weight};",
            f"--font-weight-medium: {self.medium_weight};",
            f"--font-weight-semibold: {self.semibold_weight};",
            f"--font-weight-bold: {self.bold_weight};"
        ]
        return "\n".join(css_vars)

@dataclass
class BrandSpacing:
    """Spacing and layout settings"""
    base_unit: str = "8px"
    border_radius: str = "8px"
    border_radius_small: str = "4px"
    border_radius_large: str = "12px"

    # Shadows
    shadow_small: str = "0 1px 3px rgba(0, 0, 0, 0.1)"
    shadow_medium: str = "0 4px 6px rgba(0, 0, 0, 0.1)"
    shadow_large: str = "0 10px 25px rgba(0, 0, 0, 0.15)"

    def to_css_variables(self) -> str:
        css_vars = [
            f"--spacing-unit: {self.base_unit};",
            f"--border-radius: {self.border_radius};",
            f"--border-radius-sm: {self.border_radius_small};",
            f"--border-radius-lg: {self.border_radius_large};",
            f"--shadow-sm: {self.shadow_small};",
            f"--shadow-md: {self.shadow_medium};",
            f"--shadow-lg: {self.shadow_large};"
        ]
        return "\n".join(css_vars)

@dataclass
class BrandConfiguration:
    """Complete brand configuration"""
    tenant_id: str
    tier: BrandingTier

    # Basic info
    brand_name: str
    tagline: str = ""
    description: str = ""

    # Visual identity
    colors: BrandColors = None
    typography: BrandTypography = None
    spacing: BrandSpacing = None

    # Assets
    logo_urls: Dict[AssetType, str] = None

    # Customization
    custom_css: str = ""
    custom_js: str = ""

    # Domain settings
    custom_domain: str = ""
    custom_subdomain: str = ""

    # Email branding
    email_templates: Dict[str, str] = None
    email_signature: str = ""

    # UI customization
    custom_labels: Dict[str, str] = None
    hide_agentsystem_branding: bool = False

    # Legal
    privacy_policy_url: str = ""
    terms_of_service_url: str = ""
    support_email: str = ""

    # Advanced
    custom_head_html: str = ""
    custom_footer_html: str = ""

    def __post_init__(self):
        if self.colors is None:
            self.colors = BrandColors()
        if self.typography is None:
            self.typography = BrandTypography()
        if self.spacing is None:
            self.spacing = BrandSpacing()
        if self.logo_urls is None:
            self.logo_urls = {}
        if self.email_templates is None:
            self.email_templates = {}
        if self.custom_labels is None:
            self.custom_labels = {}

class WhiteLabelService:
    """White-label branding management service"""

    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis,
                 s3_client: boto3.client, cdn_base_url: str):
        self.db_pool = db_pool
        self.redis = redis_client
        self.s3_client = s3_client
        self.cdn_base_url = cdn_base_url
        self.bucket_name = "agentsystem-assets"

        # Asset optimization settings
        self.logo_sizes = {
            'small': (32, 32),
            'medium': (128, 128),
            'large': (256, 256),
            'banner': (400, 100)
        }

        # Default email templates
        self.default_templates = {
            'welcome': '''
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    .brand-header { background: {{brand.colors.primary}}; color: white; padding: 20px; }
                    .logo { max-height: 60px; }
                    .content { padding: 30px 20px; font-family: {{brand.typography.primary_font}}; }
                </style>
            </head>
            <body>
                <div class="brand-header">
                    {% if brand.logo_urls.email_header %}
                    <img src="{{brand.logo_urls.email_header}}" alt="{{brand.brand_name}}" class="logo">
                    {% else %}
                    <h1>{{brand.brand_name}}</h1>
                    {% endif %}
                </div>
                <div class="content">
                    <h2>Welcome to {{brand.brand_name}}!</h2>
                    <p>{{content}}</p>
                    {% if brand.email_signature %}
                    <div style="margin-top: 30px; border-top: 1px solid #eee; padding-top: 20px;">
                        {{brand.email_signature | safe}}
                    </div>
                    {% endif %}
                </div>
            </body>
            </html>
            ''',
            'notification': '''
            <div style="font-family: {{brand.typography.primary_font}}; color: {{brand.colors.dark}};">
                <h3 style="color: {{brand.colors.primary}};">{{title}}</h3>
                <p>{{content}}</p>
            </div>
            '''
        }

    async def get_brand_configuration(self, tenant_id: str) -> Optional[BrandConfiguration]:
        """Get brand configuration for a tenant"""

        # Try Redis cache first
        cached = await self.redis.get(f"brand_config:{tenant_id}")
        if cached:
            data = json.loads(cached)
            return BrandConfiguration(**data)

        # Fetch from database
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM tenant_management.brand_configurations
                WHERE tenant_id = $1
            """, tenant_id)

            if not row:
                return None

            # Convert database row to configuration
            config = BrandConfiguration(
                tenant_id=row['tenant_id'],
                tier=BrandingTier(row['tier']),
                brand_name=row['brand_name'],
                tagline=row['tagline'] or "",
                description=row['description'] or "",
                colors=BrandColors(**json.loads(row['colors'] or '{}')),
                typography=BrandTypography(**json.loads(row['typography'] or '{}')),
                spacing=BrandSpacing(**json.loads(row['spacing'] or '{}')),
                logo_urls=json.loads(row['logo_urls'] or '{}'),
                custom_css=row['custom_css'] or "",
                custom_js=row['custom_js'] or "",
                custom_domain=row['custom_domain'] or "",
                custom_subdomain=row['custom_subdomain'] or "",
                email_templates=json.loads(row['email_templates'] or '{}'),
                email_signature=row['email_signature'] or "",
                custom_labels=json.loads(row['custom_labels'] or '{}'),
                hide_agentsystem_branding=row['hide_agentsystem_branding'] or False,
                privacy_policy_url=row['privacy_policy_url'] or "",
                terms_of_service_url=row['terms_of_service_url'] or "",
                support_email=row['support_email'] or "",
                custom_head_html=row['custom_head_html'] or "",
                custom_footer_html=row['custom_footer_html'] or ""
            )

            # Cache for 1 hour
            await self.redis.setex(
                f"brand_config:{tenant_id}",
                3600,
                json.dumps(asdict(config))
            )

            return config

    async def update_brand_configuration(self, tenant_id: str, updates: Dict[str, Any]) -> BrandConfiguration:
        """Update brand configuration"""

        # Check tenant's branding tier permissions
        await self._check_branding_permissions(tenant_id, updates)

        async with self.db_pool.acquire() as conn:
            # Get existing configuration or create default
            existing = await self.get_brand_configuration(tenant_id)
            if not existing:
                existing = BrandConfiguration(
                    tenant_id=tenant_id,
                    tier=BrandingTier.BASIC,
                    brand_name=updates.get('brand_name', 'Your Brand')
                )

            # Apply updates
            for key, value in updates.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)

            # Save to database
            await conn.execute("""
                INSERT INTO tenant_management.brand_configurations (
                    tenant_id, tier, brand_name, tagline, description,
                    colors, typography, spacing, logo_urls, custom_css, custom_js,
                    custom_domain, custom_subdomain, email_templates, email_signature,
                    custom_labels, hide_agentsystem_branding, privacy_policy_url,
                    terms_of_service_url, support_email, custom_head_html, custom_footer_html,
                    updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, NOW())
                ON CONFLICT (tenant_id) DO UPDATE SET
                    tier = EXCLUDED.tier,
                    brand_name = EXCLUDED.brand_name,
                    tagline = EXCLUDED.tagline,
                    description = EXCLUDED.description,
                    colors = EXCLUDED.colors,
                    typography = EXCLUDED.typography,
                    spacing = EXCLUDED.spacing,
                    logo_urls = EXCLUDED.logo_urls,
                    custom_css = EXCLUDED.custom_css,
                    custom_js = EXCLUDED.custom_js,
                    custom_domain = EXCLUDED.custom_domain,
                    custom_subdomain = EXCLUDED.custom_subdomain,
                    email_templates = EXCLUDED.email_templates,
                    email_signature = EXCLUDED.email_signature,
                    custom_labels = EXCLUDED.custom_labels,
                    hide_agentsystem_branding = EXCLUDED.hide_agentsystem_branding,
                    privacy_policy_url = EXCLUDED.privacy_policy_url,
                    terms_of_service_url = EXCLUDED.terms_of_service_url,
                    support_email = EXCLUDED.support_email,
                    custom_head_html = EXCLUDED.custom_head_html,
                    custom_footer_html = EXCLUDED.custom_footer_html,
                    updated_at = NOW()
            """,
                tenant_id, existing.tier.value, existing.brand_name, existing.tagline,
                existing.description, json.dumps(asdict(existing.colors)),
                json.dumps(asdict(existing.typography)), json.dumps(asdict(existing.spacing)),
                json.dumps(existing.logo_urls), existing.custom_css, existing.custom_js,
                existing.custom_domain, existing.custom_subdomain,
                json.dumps(existing.email_templates), existing.email_signature,
                json.dumps(existing.custom_labels), existing.hide_agentsystem_branding,
                existing.privacy_policy_url, existing.terms_of_service_url,
                existing.support_email, existing.custom_head_html, existing.custom_footer_html
            )

        # Clear cache
        await self.redis.delete(f"brand_config:{tenant_id}")
        await self.redis.delete(f"generated_css:{tenant_id}")

        # Regenerate CSS
        await self._generate_custom_css(existing)

        logger.info(f"Updated brand configuration for tenant {tenant_id}")
        return existing

    async def upload_brand_asset(self, tenant_id: str, asset_type: AssetType,
                               file: UploadFile, optimize: bool = True) -> Dict[str, str]:
        """Upload and process brand assets"""

        # Validate file type
        allowed_types = {
            AssetType.LOGO_PRIMARY: ['image/png', 'image/jpeg', 'image/svg+xml'],
            AssetType.LOGO_SECONDARY: ['image/png', 'image/jpeg', 'image/svg+xml'],
            AssetType.LOGO_ICON: ['image/png', 'image/x-icon', 'image/vnd.microsoft.icon'],
            AssetType.FAVICON: ['image/png', 'image/x-icon', 'image/vnd.microsoft.icon'],
            AssetType.BACKGROUND_IMAGE: ['image/png', 'image/jpeg', 'image/webp'],
            AssetType.EMAIL_HEADER: ['image/png', 'image/jpeg'],
            AssetType.LOADING_ANIMATION: ['image/gif', 'image/svg+xml'],
            AssetType.CUSTOM_CSS: ['text/css']
        }

        if file.content_type not in allowed_types.get(asset_type, []):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type for {asset_type}. Allowed: {allowed_types.get(asset_type)}"
            )

        # Check file size (max 5MB for images, 1MB for CSS)
        max_size = 1024 * 1024 * (5 if 'image' in file.content_type else 1)
        content = await file.read()

        if len(content) > max_size:
            raise HTTPException(status_code=400, detail="File too large")

        # Process and optimize asset
        processed_assets = {}

        if asset_type in [AssetType.LOGO_PRIMARY, AssetType.LOGO_SECONDARY, AssetType.EMAIL_HEADER]:
            processed_assets = await self._process_logo_asset(tenant_id, asset_type, content, file.content_type)
        elif asset_type == AssetType.FAVICON:
            processed_assets = await self._process_favicon(tenant_id, content)
        elif asset_type == AssetType.CUSTOM_CSS:
            processed_assets = await self._process_custom_css(tenant_id, content.decode('utf-8'))
        else:
            processed_assets = await self._upload_generic_asset(tenant_id, asset_type, content, file.content_type)

        # Update brand configuration with new asset URLs
        brand_config = await self.get_brand_configuration(tenant_id)
        if brand_config:
            brand_config.logo_urls.update(processed_assets)
            await self.update_brand_configuration(tenant_id, {'logo_urls': brand_config.logo_urls})

        return processed_assets

    async def generate_theme_css(self, tenant_id: str) -> str:
        """Generate complete CSS theme for tenant"""

        # Check cache first
        cached_css = await self.redis.get(f"generated_css:{tenant_id}")
        if cached_css:
            return cached_css.decode()

        brand_config = await self.get_brand_configuration(tenant_id)
        if not brand_config:
            return self._get_default_css()

        # Generate CSS from brand configuration
        css_parts = [
            ":root {",
            brand_config.colors.to_css_variables(),
            brand_config.typography.to_css_variables(),
            brand_config.spacing.to_css_variables(),
            "}"
        ]

        # Add logo CSS if logos exist
        if brand_config.logo_urls:
            css_parts.extend(self._generate_logo_css(brand_config))

        # Add custom CSS
        if brand_config.custom_css:
            css_parts.append(brand_config.custom_css)

        # Add component-specific styles
        css_parts.extend(self._generate_component_styles(brand_config))

        final_css = "\n\n".join(css_parts)

        # Cache for 1 hour
        await self.redis.setex(f"generated_css:{tenant_id}", 3600, final_css)

        return final_css

    async def generate_email_template(self, tenant_id: str, template_name: str,
                                    context: Dict[str, Any]) -> str:
        """Generate branded email template"""

        brand_config = await self.get_brand_configuration(tenant_id)
        if not brand_config:
            # Return default template
            return self._render_default_email_template(template_name, context)

        # Get template
        template_html = brand_config.email_templates.get(
            template_name,
            self.default_templates.get(template_name, self.default_templates['notification'])
        )

        # Render with Jinja2
        env = Environment(loader=DictLoader({template_name: template_html}))
        template = env.get_template(template_name)

        return template.render(
            brand=brand_config,
            **context
        )

    async def extract_colors_from_logo(self, image_content: bytes) -> BrandColors:
        """Extract brand colors from logo using AI color analysis"""

        # Save temporary file for ColorThief
        temp_path = f"/tmp/logo_{secrets.token_hex(8)}.png"

        try:
            # Convert to RGB if needed and save
            with Image.open(io.BytesIO(image_content)) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(temp_path, 'PNG')

            # Extract dominant colors
            color_thief = ColorThief(temp_path)
            dominant_color = color_thief.get_color(quality=1)
            palette = color_thief.get_palette(color_count=6, quality=10)

            # Convert to hex colors
            def rgb_to_hex(rgb):
                return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

            primary = rgb_to_hex(dominant_color)
            colors = [rgb_to_hex(color) for color in palette]

            # Create brand color palette
            brand_colors = BrandColors(
                primary=primary,
                secondary=colors[1] if len(colors) > 1 else BrandColors().secondary,
                accent=colors[2] if len(colors) > 2 else BrandColors().accent,
            )

            # Generate complementary colors if needed
            if len(colors) > 3:
                brand_colors.success = colors[3]
            if len(colors) > 4:
                brand_colors.info = colors[4]

            return brand_colors

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    async def _process_logo_asset(self, tenant_id: str, asset_type: AssetType,
                                content: bytes, content_type: str) -> Dict[str, str]:
        """Process and optimize logo assets"""

        uploaded_assets = {}

        if content_type == 'image/svg+xml':
            # Handle SVG logos
            svg_url = await self._upload_to_s3(
                f"brands/{tenant_id}/{asset_type.value}.svg",
                content,
                content_type
            )
            uploaded_assets[asset_type] = svg_url
        else:
            # Process bitmap images - create multiple sizes
            with Image.open(io.BytesIO(content)) as img:
                # Ensure RGBA for transparency
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')

                for size_name, (width, height) in self.logo_sizes.items():
                    # Resize maintaining aspect ratio
                    resized = ImageOps.contain(img, (width, height), Image.Resampling.LANCZOS)

                    # Convert to bytes
                    output = io.BytesIO()
                    resized.save(output, format='PNG', optimize=True)
                    output.seek(0)

                    # Upload
                    url = await self._upload_to_s3(
                        f"brands/{tenant_id}/{asset_type.value}_{size_name}.png",
                        output.getvalue(),
                        'image/png'
                    )

                    uploaded_assets[f"{asset_type}_{size_name}"] = url

                # Use medium size as default
                uploaded_assets[asset_type] = uploaded_assets.get(f"{asset_type}_medium", "")

        return uploaded_assets

    async def _process_favicon(self, tenant_id: str, content: bytes) -> Dict[str, str]:
        """Process favicon in multiple formats"""

        uploaded_assets = {}

        with Image.open(io.BytesIO(content)) as img:
            favicon_sizes = [(16, 16), (32, 32), (48, 48)]

            for width, height in favicon_sizes:
                resized = img.resize((width, height), Image.Resampling.LANCZOS)

                # PNG format
                png_output = io.BytesIO()
                resized.save(png_output, format='PNG', optimize=True)
                png_output.seek(0)

                png_url = await self._upload_to_s3(
                    f"brands/{tenant_id}/favicon_{width}x{height}.png",
                    png_output.getvalue(),
                    'image/png'
                )

                uploaded_assets[f"favicon_{width}x{height}"] = png_url

            # ICO format for legacy support
            ico_output = io.BytesIO()
            img.save(ico_output, format='ICO', sizes=[(16,16), (32,32), (48,48)])
            ico_output.seek(0)

            ico_url = await self._upload_to_s3(
                f"brands/{tenant_id}/favicon.ico",
                ico_output.getvalue(),
                'image/x-icon'
            )

            uploaded_assets[AssetType.FAVICON] = ico_url

        return uploaded_assets

    async def _upload_to_s3(self, key: str, content: bytes, content_type: str) -> str:
        """Upload file to S3 and return CDN URL"""

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=content,
            ContentType=content_type,
            CacheControl='max-age=31536000',  # Cache for 1 year
            ACL='public-read'
        )

        return f"{self.cdn_base_url}/{key}"

    async def _check_branding_permissions(self, tenant_id: str, updates: Dict[str, Any]):
        """Check if tenant has permissions for branding features"""

        async with self.db_pool.acquire() as conn:
            tenant_row = await conn.fetchrow("""
                SELECT plan_type FROM tenant_management.tenants WHERE id = $1
            """, tenant_id)

            if not tenant_row:
                raise HTTPException(status_code=404, detail="Tenant not found")

            plan_type = tenant_row['plan_type']

            # Check plan permissions
            if plan_type == 'starter' and any(key in updates for key in ['custom_domain', 'hide_agentsystem_branding']):
                raise HTTPException(
                    status_code=403,
                    detail="Custom domain and branding removal require Professional plan or higher"
                )

            if plan_type in ['starter', 'professional'] and any(key in updates for key in ['custom_css', 'custom_js']):
                raise HTTPException(
                    status_code=403,
                    detail="Custom CSS/JS requires Enterprise plan or higher"
                )

    def _generate_component_styles(self, brand_config: BrandConfiguration) -> List[str]:
        """Generate component-specific CSS styles"""

        return [
            f"""
            /* Button Styles */
            .btn-primary {{
                background-color: {brand_config.colors.primary};
                border-color: {brand_config.colors.primary};
                font-family: {brand_config.typography.primary_font};
            }}

            .btn-primary:hover {{
                background-color: {self._darken_color(brand_config.colors.primary, 0.1)};
            }}

            /* Navigation */
            .navbar-brand img {{
                max-height: 40px;
            }}

            /* Dashboard Cards */
            .card {{
                border-radius: {brand_config.spacing.border_radius};
                box-shadow: {brand_config.spacing.shadow_medium};
            }}

            /* Links */
            a {{
                color: {brand_config.colors.primary};
            }}

            a:hover {{
                color: {self._darken_color(brand_config.colors.primary, 0.2)};
            }}
            """
        ]

    def _darken_color(self, hex_color: str, factor: float) -> str:
        """Darken a hex color by a factor"""
        # Simple color darkening - in production, use a proper color library
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
        darkened = tuple(int(c * (1 - factor)) for c in rgb)
        return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"

    def _get_default_css(self) -> str:
        """Get default CSS theme"""
        default_colors = BrandColors()
        default_typography = BrandTypography()
        default_spacing = BrandSpacing()

        return f"""
        :root {{
            {default_colors.to_css_variables()}
            {default_typography.to_css_variables()}
            {default_spacing.to_css_variables()}
        }}
        """

# Database schema addition for brand configurations
# Database schema addition for brand configurations
BRAND_SCHEMA_SQL = """
-- Add branding table to tenant_management schema
CREATE TABLE IF NOT EXISTS tenant_management.brand_configurations (
    tenant_id UUID PRIMARY KEY REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    tier VARCHAR(50) NOT NULL DEFAULT 'basic',
    brand_name VARCHAR(255) NOT NULL,
    tagline TEXT,
    description TEXT,
    colors JSONB DEFAULT '{}',
    typography JSONB DEFAULT '{}',
    spacing JSONB DEFAULT '{}',
    logo_urls JSONB DEFAULT '{}',
    custom_css TEXT DEFAULT '',
    custom_js TEXT DEFAULT '',
    custom_domain VARCHAR(255) DEFAULT '',
    custom_subdomain VARCHAR(255) DEFAULT '',
    email_templates JSONB DEFAULT '{}',
    email_signature TEXT DEFAULT '',
    custom_labels JSONB DEFAULT '{}',
    hide_agentsystem_branding BOOLEAN DEFAULT false,
    privacy_policy_url VARCHAR(500) DEFAULT '',
    terms_of_service_url VARCHAR(500) DEFAULT '',
    support_email VARCHAR(255) DEFAULT '',
    custom_head_html TEXT DEFAULT '',
    custom_footer_html TEXT DEFAULT '',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_brand_configurations_tier
ON tenant_management.brand_configurations(tier);

-- Index for custom domain lookups
CREATE INDEX IF NOT EXISTS idx_brand_configurations_custom_domain
ON tenant_management.brand_configurations(custom_domain)
WHERE custom_domain != '';
"""

# Pydantic models for API requests
class BrandConfigurationRequest(BaseModel):
    brand_name: str = Field(..., min_length=1, max_length=255)
    tagline: str = Field("", max_length=500)
    description: str = Field("", max_length=2000)
    primary_color: str = Field("#007bff", regex=r"^#[0-9a-fA-F]{6}$")
    secondary_color: str = Field("#6c757d", regex=r"^#[0-9a-fA-F]{6}$")
    custom_domain: str = Field("", max_length=255)
    support_email: str = Field("", max_length=255)

    @validator('custom_domain')
    def validate_domain(cls, v):
        if v and not v.replace('.', '').replace('-', '').isalnum():
            raise ValueError('Invalid domain format')
        return v

class AssetUploadRequest(BaseModel):
    asset_type: AssetType = Field(..., description="Type of asset being uploaded")

class ColorExtractionRequest(BaseModel):
    extract_from_logo: bool = Field(False, description="Extract colors from uploaded logo")

# Export main classes
__all__ = [
    'WhiteLabelService', 'BrandConfiguration', 'BrandColors', 'BrandTypography',
    'BrandSpacing', 'BrandingTier', 'AssetType', 'BrandConfigurationRequest',
    'AssetUploadRequest', 'ColorExtractionRequest', 'BRAND_SCHEMA_SQL'
]

# Shipped landing nginx proposal (staging only)

This directory contains a **proposed** nginx config for claim 0046.
It is not deployed by this PR.

## Scope

`shipped-landing.conf` is intended to route only these dead-air hostnames to one static landing page at `/var/www/shipped/index.html`:

- `dead-air.show`
- `www.dead-air.show`

No other vhost should be changed.

## Apply procedure (staging only)

Replace the existing vhost file in place (do not add a second file with the same `server_name`):

- Existing file to replace: `/etc/nginx/sites-enabled/dead-air.show` (regular file)
- Do not use `sites-available/dead-air.show` for this change; it is a stale copy.

1. Create a dated backup directory outside any webroot and back up the current file:
   ```bash
   BACKUP_DIR="/var/backups/nginx/$(date +%F-%H%M%S)"
   sudo mkdir -p "$BACKUP_DIR"
   sudo cp -a /etc/nginx/sites-enabled/dead-air.show "$BACKUP_DIR/dead-air.show"
   ```
2. Copy in the proposed file content to `/etc/nginx/sites-enabled/dead-air.show`.
3. Ensure docroots exist:
   ```bash
   sudo mkdir -p /var/www/shipped /var/www/acme-dead-air
   sudo chown root:root /var/www/shipped /var/www/acme-dead-air
   sudo chmod 755 /var/www/shipped /var/www/acme-dead-air
   ```
4. Install the landing page file:
   ```bash
   sudo install -m 0644 deploy/shipped-landing/index.html /var/www/shipped/index.html
   ```
5. Validate nginx config:
   ```bash
   sudo nginx -t
   ```
6. Reload only after successful validation:
   ```bash
   sudo systemctl reload nginx
   ```

If `nginx -t` fails, do not reload.

## Rollback

1. Restore the original enabled vhost file:
   ```bash
   cp -a "$BACKUP_DIR/dead-air.show" /etc/nginx/sites-enabled/dead-air.show && nginx -t && systemctl reload nginx
   ```
2. If the one-liner above is split, run:
   ```bash
   sudo nginx -t
   ```
3. Reload nginx:
   ```bash
   sudo systemctl reload nginx
   ```

## Notes

- `/legal/` alias is included as `alias /opt/legal/;`.
- `/robots.txt` alias is included with a clearly marked placeholder path; copy the exact path from the backed-up live file before apply.
- `/rss`, `/podcast/`, and `/episodes/` redirects to `https://agentlooper.org/radio/` are included.
- Certbot renewal on this host uses the nginx authenticator flow, not the ACME webroot path in this file.
- Rollback does not remove `/var/www/shipped` or `/var/www/acme-dead-air`; leaving those directories in place is harmless.
- Out of scope for this claim: Lisa implementation changes, WaveForge/HVACFix product changes, iptables, and rizz.

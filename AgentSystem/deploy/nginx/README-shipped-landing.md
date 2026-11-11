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

- Existing file to replace: `/etc/nginx/sites-available/dead-air.show.conf`

1. Create a dated backup directory outside any webroot and back up the current file:
   ```bash
   BACKUP_DIR="/var/backups/nginx/$(date +%F-%H%M%S)"
   sudo mkdir -p "$BACKUP_DIR"
   sudo cp -a /etc/nginx/sites-available/dead-air.show.conf "$BACKUP_DIR/dead-air.show.conf"
   ```
2. Copy in the proposed file content to `/etc/nginx/sites-available/dead-air.show.conf`.
3. Ensure docroots exist:
   ```bash
   sudo mkdir -p /var/www/shipped /var/www/acme-dead-air
   ```
4. Validate nginx config:
   ```bash
   sudo nginx -t
   ```
5. Reload only after successful validation:
   ```bash
   sudo systemctl reload nginx
   ```

If `nginx -t` fails, do not reload.

## Rollback

1. Restore the same backed-up file to `/etc/nginx/sites-available/dead-air.show.conf`.
2. Re-run:
   ```bash
   sudo nginx -t
   ```
3. Reload nginx:
   ```bash
   sudo systemctl reload nginx
   ```

## Notes

- Legacy redirects for `/legal/`, `/podcast`, `/rss`, and `/episodes` are not included in this proposal.
- Out of scope for this claim: Lisa implementation changes, WaveForge/HVACFix product changes, iptables, and rizz.

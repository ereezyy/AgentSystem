# Shipped landing nginx proposal (staging only)

This directory contains a **proposed** nginx config for claim 0046.
It is not deployed by this PR.

## Scope

`shipped-landing.conf` is intended to route these domains to one static landing page at `/var/www/shipped/index.html`:

- `dead-air.show`
- `agentlooper.org`
- `proofchain.us`
- `self.proofchain.us`

No other vhost should be changed.

## Notes on TLS paths

The TLS certificate paths are intentionally placeholders:

- `/etc/letsencrypt/live/REPLACE_WITH_EXISTING_CERT/fullchain.pem`
- `/etc/letsencrypt/live/REPLACE_WITH_EXISTING_CERT/privkey.pem`

Before any staging rollout, replace them with the exact existing cert paths currently used for those hostnames.

## Validation command

Run syntax validation before reload:

```bash
sudo nginx -t
```

## Rollback

1. Restore the previous server blocks for these four hostnames.
2. Re-run:
   ```bash
   sudo nginx -t
   ```
3. Reload nginx after validation:
   ```bash
   sudo systemctl reload nginx
   ```

If validation fails, do not reload.

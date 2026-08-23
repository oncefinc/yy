# Security policy

## Reporting a vulnerability

Please use GitHub's **Report a vulnerability** flow for this repository. Do
not open a public issue containing API keys, access tokens, WeChat identifiers,
chat excerpts, memory records or other personal data.

If a credential has already been exposed, revoke or rotate it before sending
the report. A Git history rewrite does not invalidate a leaked credential.

## Deployment safety

- Keep `config.json`, `.env`, runtime databases and memory stores outside Git.
- Use a test account and Shadow mode for initial validation.
- Proactive delivery is disabled by default and must be explicitly enabled
  with `INITIATIVE_DELIVERY_ENABLED=true`.
- Review receiver IDs, quiet hours and daily budgets before enabling delivery.
- Treat location and temporal-state databases as private user data.

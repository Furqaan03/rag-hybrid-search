# Configuration Guide

## Server Settings

The API server listens on port 8080 by default. To change it, set the `PORT`
environment variable.

## Database

The `DATABASE_URL` environment variable holds the database connection string.
It must be a valid PostgreSQL URL.

## Caching

Caching is disabled by default. To enable it, set `ENABLE_CACHE=true`. Caching
depends on Redis, so you must also set `REDIS_URL` for it to work.

## Timeouts

The request timeout defaults to 30 seconds and is set via `REQUEST_TIMEOUT`.
The idle connection timeout defaults to 120 seconds and is set via
`IDLE_TIMEOUT`.

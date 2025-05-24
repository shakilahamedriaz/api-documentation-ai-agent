# API Authentication Guide

## Overview
Our API uses API Keys to authenticate requests. You must include an API key in all requests to the API.

## Obtaining Your API Key
You can generate and manage your API Keys from your account dashboard under the "API Settings" section.
1. Log in to your account.
2. Navigate to "Settings" > "API Keys".
3. Click "Generate New Key".
4. Make sure to copy your new API key immediately. For security reasons, we do not show it again.

## Using The API Key
To authenticate an API request, include your API Key in the `Authorization` header. The key should be prefixed with `Bearer `.

**Header Format:**
`Authorization: Bearer YOUR_API_KEY`

Replace `YOUR_API_KEY` with your actual generated API key.

## Security Best Practices
* Keep your API keys confidential.
* Do not embed API keys directly in client-side code.
* Consider using environment variables or a secrets management system to store your keys on the server-side.
* Rotate your API keys periodically.
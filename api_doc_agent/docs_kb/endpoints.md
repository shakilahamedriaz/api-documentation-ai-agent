# API Endpoints Reference

This document provides details about available API endpoints.

## Users API

### Get User Details
* **HTTP Method:** `GET`
* **Endpoint Path:** `/v1/users/{userId}`
* **Description:** Retrieves detailed information for a specific user identified by `userId`.
* **Path Parameters:**
    * `userId` (string, required): The unique identifier of the user. Example: `usr_AKei38DKs`
* **Query Parameters:**
    * `fields` (string, optional): Comma-separated list of fields to include in the response. Example: `name,email,createdAt`
* **Success Response (200 OK):**
  ```json
  {
    "id": "usr_AKei38DKs",
    "name": "Alice Wonderland",
    "email": "alice@example.com",
    "createdAt": "2024-01-15T10:30:00Z",
    "isActive": true
  }
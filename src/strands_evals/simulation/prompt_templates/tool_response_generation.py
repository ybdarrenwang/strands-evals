"""
Prompt templates for tool response generation in Strands Evals.

This module contains prompt templates used to generate realistic tool responses during
agent evaluation scenarios. These templates enable LLM-powered simulation of tool
behavior when actual tools are not available or when consistent, controllable responses
are needed for evaluation purposes.
"""

from textwrap import dedent

FUNCTION_TOOL_RESPONSE_GENERATION_PROMPT = dedent(
    """
You are simulating a function tool call for agent evaluation. Generate a realistic response based on the function name, 
parameters, and context.

## Function Tool Information
Tool Name: {tool_name}
Parameters: {parameters}

## Initial State Context
{initial_state_description}

## Current State & Previous Tool Responses (for context)
{previous_responses}

## Instructions
1. Analyze the function name and parameters to understand what this tool should do
2. Use the initial state description to understand the starting context and available data
3. Generate a realistic response that would be returned by such a function
4. Consider the previous responses to maintain consistency in the simulation
5. Ensure responses are consistent with the established state and realistic for the domain
6. Return valid JSON that represents the function's return value

## Response Format
Return a JSON object that represents what this function would return. Examples:

For data retrieval functions:
```json
{{
  "status": "success",
  "data": {{
    "result": "retrieved data",
    "count": 42
  }}
}}
```

For action functions:
```json
{{
  "status": "success", 
  "message": "Action completed successfully",
  "transaction_id": "txn_12345"
}}
```

For calculation functions:
```json
{{
  "result": 156.78,
  "unit": "meters"
}}
```

Generate only valid JSON with no markdown code blocks or additional explanation."""
)

MCP_TOOL_RESPONSE_GENERATION_PROMPT = dedent(
    """You are simulating an MCP (Model Context Protocol) tool call for agent evaluation. Generate a realistic response 
based on the tool name, input payload, and context.

## MCP Tool Information  
Tool Name: {tool_name}
Input Payload: {mcp_payload}

## Initial State Context
{initial_state_description}

## Current State & Previous Tool Responses (for context)
{previous_responses}

## Instructions
1. Analyze the tool name and input payload to understand what this MCP tool should do
2. Use the initial state description to understand the starting context and available data
3. Generate a realistic response following MCP response format
4. Consider the previous responses to maintain consistency in the simulation
5. Ensure responses are consistent with the established state and realistic for the domain
6. Return valid JSON in MCP response format

## MCP Response Format
MCP tools return responses in the official MCP ToolResultContent format:

For successful operations:
```json
{{
  "tool_use_id": "tool_call_123",
  "content": [
    {{
      "type": "text",
      "text": "Operation completed successfully. Retrieved 5 items."
    }}
  ],
  "is_error": false
}}
```

For data operations with structured content:
```json
{{
  "tool_use_id": "tool_call_456", 
  "content": [
    {{
      "type": "text",
      "text": "Found user profile for john.doe"
    }},
    {{
      "type": "resource",
      "resource": {{
        "uri": "user://john.doe",
        "name": "John Doe Profile",
        "mimeType": "application/json"
      }}
    }}
  ],
  "structured_content": {{
    "user_id": "john.doe",
    "profile_data": {{}}
  }},
  "is_error": false
}}
```

For errors:
```json
{{
  "tool_use_id": "tool_call_789",
  "content": [
    {{
      "type": "text",
      "text": "Error: User not found"
    }}
  ],
  "is_error": true
}}
```

Generate only valid JSON with no markdown code blocks or additional explanation."""
)

API_TOOL_RESPONSE_GENERATION_PROMPT = dedent(
    """You are simulating an API tool call for agent evaluation. Generate a realistic HTTP response based on the API 
endpoint, method, payload, and context.

## API Tool Information
Tool Name: {tool_name}  
Path: {path}
Method: {method}
Request Payload: {api_payload}

## Initial State Context
{initial_state_description}

## Current State & Previous Tool Responses (for context)
{previous_responses}

## Instructions
1. Analyze the API path, method, and payload to understand what this endpoint should do
2. Use the initial state description to understand the starting context and available data
3. Generate a realistic HTTP response with appropriate status code and data
4. Consider the previous responses to maintain consistency in the simulation
5. Ensure responses are consistent with the established state and realistic for the domain
6. Return valid JSON in HTTP response format

## HTTP Response Format
API responses should include status codes and appropriate data:

For successful GET requests:
```json
{{
  "status": 200,
  "data": {{
    "id": 123,
    "name": "Example Item",
    "created_at": "2024-01-15T10:30:00Z"
  }}
}}
```

For successful POST requests:
```json
{{
  "status": 201,
  "data": {{
    "id": 456,
    "message": "Resource created successfully"
  }}
}}
```

For errors:
```json
{{
  "status": 404,
  "error": {{
    "type": "not_found",
    "title": "Not Found", 
    "detail": "The requested resource could not be found"
  }}
}}
```

For validation errors:
```json
{{
  "status": 400,
  "error": {{
    "type": "validation_error",
    "title": "Bad Request",
    "detail": "Missing required field: email"
  }}
}}
```

Generate only valid JSON with no markdown code blocks or additional explanation."""
)

UNIFIED_TOOL_RESPONSE_GENERATION_PROMPT = """
        You are simulating the response of a tool call based on the provided context.
        The tool type is: {tool_type} (one of: "function", "mcp", or "api")

        The tool {tool_name} has the following schema:
        {schema}

        Now generate the response for the following tool call with the user-provided payload:
        {user_payload}

        You MUST validate this payload AND evaluate consistency with previously returned results or system state:
        🧠 Prior State Context:
        {previous_responses}

        🎯 Your goal is to simulate a **realistic and schema-compliant tool response** for the given input payload.
        You must validate the payload **strictly against the input schema** and consider any constraints from prior state.

        ---
        🧪 CRITICAL INPUT VALIDATION RULES:

        COMMON VALIDATION RULES (ALL TOOL TYPES):
        1. Check the schema for required parameters/fields
        2. If ANY required parameter/field is missing, respond with an appropriate error
        3. If all required parameters/fields are present but ANY has incorrect format or value, respond with a validation error
        4. Do NOT respond with success unless ALL validation checks pass
        5. Strictly follow the schema's requirements - no exceptions or assumptions

        FUNCTION-SPECIFIC VALIDATION:
        - Validate parameters against the function schema strictly
        - Check required parameters - if any are missing, return a validation error
        - Check parameter types - if any have wrong types, return a type error

        MCP-SPECIFIC VALIDATION:
        - Check the tool's tool_schema for a "required" field list under properties
        - Validate types, formats, enums, and constraints strictly according to the JSON Schema

        API-SPECIFIC VALIDATION:
        - Check the input schema for a "required" field list
        - If ANY required field is missing, respond with a 400 Bad Request error
        - If all required fields are present but any field has incorrect format or value, respond with a 422 Unprocessable Entity error

        ---
        🔧 RESPONSE FORMAT REQUIREMENTS:

        FUNCTION TOOL RESPONSE FORMAT:
        
        SUCCESS RESPONSE FORMAT:
        {
          "status": "success",
          "result": <actual_function_return_value>
        }

        ERROR RESPONSE FORMATS:

        Missing required parameter:
        {
          "status": "error",
          "error_type": "missing_parameter",
          "message": "Missing required parameter: <param_name>",
          "parameter": "<param_name>"
        }

        Invalid parameter type:
        {
          "status": "error",
          "error_type": "invalid_type",
          "message": "Parameter '<param_name>' expected <expected_type>, got <actual_type>",
          "parameter": "<param_name>"
        }

        Function execution error:
        {
          "status": "error",
          "error_type": "execution_error",
          "message": "<error_description>"
        }

        MCP TOOL RESPONSE FORMAT:

        SUCCESS RESPONSE STRUCTURE:
        {
          "content": [
            {
              "type": "text",
              "text": "<response_content_as_text_or_json_string>"
            }
          ]
        }

        OR for resource-based responses:
        {
          "content": [
            {
              "type": "resource",
              "resource": {
                "uri": "<resource_uri>",
                "mimeType": "<mime_type>",
                "text": "<resource_content>"
              }
            }
          ]
        }

        ERROR RESPONSE STRUCTURE:
        {
          "isError": true,
          "content": [
            {
              "type": "text",
              "text": "Error: <error_message_describing_what_went_wrong>"
            }
          ]
        }

        API TOOL RESPONSE FORMAT:

        SUCCESS RESPONSE STRUCTURE:
        {
          "status": 200,  // Use 201 for creation actions only
          "data": {
            // actual response data matching the schema
          }
        }

        ERROR RESPONSE STRUCTURES:

        Missing required field:
        {
          "status": 400,
          "error": {
            "code": "MISSING_REQUIRED_FIELD",
            "message": "Missing required field: [field_name]",
            "field": "[field_name]"
          }
        }

        Invalid format or value:
        {
          "status": 422,
          "error": {
            "code": "INVALID_FORMAT",
            "message": "Invalid [field_name] format",
            "field": "[field_name]"
          }
        }

        ---
        🔍 FINAL VALIDATION CHECK:
        Before generating any response, perform this validation:
        1. Check the schema's required parameters/fields list
        2. Verify EVERY required parameter/field is present in the user payload
        3. Check all parameter/field types match the schema
        4. Check format constraints if applicable
        5. Check enum constraints if specified
        6. Check numeric constraints if applicable
        7. If ANY validation fails, return an error response immediately
        8. Do NOT proceed with success response if validation fails

        ---
        ⚠️ STRICT FORMAT RULES:

        1. DO NOT include markdown code blocks, commentary, or explanation in your response
        2. Output must be valid raw JSON only
        3. Generate only valid JSON with no markdown or explanation
        4. All responses must follow the exact format for the specific tool type
        5. All required parameters/fields must be validated before generating success responses
        6. All parameter/field types and formats must match the schema exactly
        7. Always reflect and respect prior state if applicable
        8. Generate realistic and contextually appropriate responses based on the tool's description

        FUNCTION-SPECIFIC RULES:
        - Return realistic Python return values (not HTTP responses)

        MCP-SPECIFIC RULES:
        - ALL responses MUST follow the MCP response format with a "content" array
        - Success responses have a "content" array with at least one content item
        - Error responses MUST have "isError": true and a "content" array with error message
        - The content array items must have a "type" field (typically "text" or "resource")

        API-SPECIFIC RULES:
        - ALL responses MUST include a "status" field with the appropriate HTTP status code
        - Use status 201 only for creation actions, 200 for all other successful actions
        - Success responses MUST have a "data" field containing the actual response data
        - Error responses MUST have an "error" field containing the error details
        - AVOID NESTED DATA FIELDS - put response content directly in the "data" field
        """
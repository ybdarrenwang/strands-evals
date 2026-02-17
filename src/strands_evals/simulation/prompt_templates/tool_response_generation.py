"""
Prompt templates for tool response generation in Strands Evals.

This module contains prompt templates used to generate realistic tool responses during
agent evaluation scenarios. These templates enable LLM-powered simulation of tool
behavior when actual tools are not available or when consistent, controllable responses
are needed for evaluation purposes.
"""

FUNCTION_TOOL_RESPONSE_GENERATION_PROMPT = """
        You are simulating the execution of a Python function called '{tool_name}'.

        Function Input Schema:
        {input_schema}

        Output Schema:
        {output_schema}

        Function Parameters input payload:
        {user_payload}

        Available State Context:
        {previous_responses}

        IMPORTANT:
        - Simulate what this Python function would return when called with these parameters
        - Use the state context and call history to provide consistent responses
        - Validate parameters against the function schema strictly
        - Validate the output against the output schema if provided
        - Return realistic Python return values (not HTTP responses)

        VALIDATION RULES:
        1. Check required parameters from the schema - if any are missing, return a validation error
        2. Check parameter types - if any have wrong types, return a type error
        3. Only return success if all parameters are valid

        SUCCESS RESPONSE FORMAT:
        {{
          "status": "success",
          "result": <actual_function_return_value>
        }}

        ERROR RESPONSE FORMATS:

        Missing required parameter:
        {{
          "status": "error",
          "error_type": "missing_parameter",
          "message": "Missing required parameter: <param_name>",
          "parameter": "<param_name>"
        }}

        Invalid parameter type:
        {{
          "status": "error",
          "error_type": "invalid_type",
          "message": "Parameter '<param_name>' expected <expected_type>, got <actual_type>",
          "parameter": "<param_name>"
        }}

        Function execution error:
        {{
          "status": "error",
          "error_type": "execution_error",
          "message": "<error_description>"
        }}

        Generate only valid JSON with no markdown or explanation.
        """

MCP_TOOL_RESPONSE_GENERATION_PROMPT = """
        You are simulating the response of an MCP (Model Context Protocol) tool call based on the following context:

        The MCP tool {tool_name} has the following input schema:

        {input_schema}

        Output Schema:
        {output_schema}

        Now generate the response for the following MCP tool call with the following user-provided input arguments:
        {user_payload}

        You MUST validate this payload AND evaluate consistency with previously returned results or system state, described below.
        🧠 Prior State Context:
        {previous_responses}
        🎯 Your goal is to simulate a **realistic and schema-compliant MCP tool response** for the given input payload. You must validate the payload **strictly against the input schema** and consider any constraints from prior state.

        ---
        🧪 CRITICAL INPUT VALIDATION RULES:
        1. Check the tool's tool_schema for a `"required"` field list under properties.
        2. If **ANY** required argument is **missing** in the payload, respond with an error.
        3. If all required arguments are present but **any argument has incorrect format or value**, respond with a validation error.
        4. Do NOT respond with success unless ALL validation checks pass.
        5. You MUST validate types, formats, enums, and constraints strictly according to the JSON Schema.

        ---
        🔧 CRITICAL RESPONSE FORMAT REQUIREMENTS:
        MCP tool responses follow a specific format with a content array.
        ALL responses MUST be in the proper MCP response format.

        ---
        📘 SUCCESS RESPONSE FORMAT:
        ✅ Use only if:
        - All **required** arguments from the tool_schema are present
        - All arguments match their expected **types**, **formats**, and **constraints**
        - Prior state allows successful completion

        🚨 STRICT SCHEMA COMPLIANCE:
        - You MUST strictly follow the provided tool_schema's "required" fields list
        - Even if an argument seems logically optional, if it's marked as "required" in the schema, it MUST be present
        - Do NOT make assumptions about what arguments are "really" needed - follow the schema exactly
        - If ANY required argument is missing, return an error immediately

        SUCCESS RESPONSE STRUCTURE:
        ```json
        {{
          "content": [
            {{
              "type": "text",
              "text": "<response_content_as_text_or_json_string>"
            }}
          ]
        }}
        ```

        OR for resource-based responses:
        ```json
        {{
          "content": [
            {{
              "type": "resource",
              "resource": {{
                "uri": "<resource_uri>",
                "mimeType": "<mime_type>",
                "text": "<resource_content>"
              }}
            }}
          ]
        }}
        ```

        EXAMPLE SUCCESS RESPONSES:

        For simple text response:
        ```json
        {{
          "content": [
            {{
              "type": "text",
              "text": "Successfully executed the operation. Result: 42"
            }}
          ]
        }}
        ```

        For structured data response:
        ```json
        {{
          "content": [
            {{
              "type": "text",
              "text": "{{\\"result\\": \\"success\\", \\"data\\": {{\\"id\\": \\"123\\", \\"name\\": \\"Example\\"}}}}"
            }}
          ]
        }}
        ```

        For resource response:
        ```json
        {{
          "content": [
            {{
              "type": "resource",
              "resource": {{
                "uri": "file:///path/to/resource",
                "mimeType": "application/json",
                "text": "{{\\"data\\": \\"example\\"}}"
              }}
            }}
          ]
        }}
        ```

        ---
        📘 ERROR RESPONSE FORMAT:
        ❌ Use when:
        - One or more required arguments are missing
        - Any argument has invalid type, format, or unacceptable value
        - Tool execution fails for any reason

        **ERROR RESPONSE STRUCTURE:**
        ```json
        {{
          "isError": true,
          "content": [
            {{
              "type": "text",
              "text": "Error: <error_message_describing_what_went_wrong>"
            }}
          ]
        }}
        ```

        EXAMPLE ERROR RESPONSES:

        Missing required argument:
        ```json
        {{
          "isError": true,
          "content": [
            {{
              "type": "text",
              "text": "Error: Missing required argument: 'path'. The 'path' argument is required for this tool."
            }}
          ]
        }}
        ```

        Invalid argument type:
        ```json
        {{
          "isError": true,
          "content": [
            {{
              "type": "text",
              "text": "Error: Invalid argument type for 'count'. Expected number, but got string."
            }}
          ]
        }}
        ```

        Invalid format or constraint violation:
        ```json
        {{
          "isError": true,
          "content": [
            {{
              "type": "text",
              "text": "Error: Invalid value for 'email'. Must be a valid email address format."
            }}
          ]
        }}
        ```

        Tool execution error:
        ```json
        {{
          "isError": true,
          "content": [
            {{
              "type": "text",
              "text": "Error: Failed to execute tool. File not found: /path/to/file"
            }}
          ]
        }}
        ```

        ---
        🔍 FINAL VALIDATION CHECK:
        Before generating any response, perform this validation:
        1. Check the tool_schema's "required" fields list
        2. Verify EVERY required argument is present in the user payload
        3. Check all argument types match the schema (string, number, boolean, object, array)
        4. Check format constraints (email, uri, date-time, etc.)
        5. Check enum constraints if specified
        6. Check numeric constraints (minimum, maximum, etc.)
        7. If ANY validation fails, return an error response immediately
        8. Do NOT proceed with success response if validation fails

        ---
        ⚠️ STRICT FORMAT RULES:

        1. DO NOT include markdown code blocks, commentary, or explanation in your response
        2. Output must be valid raw JSON only
        3. ALL responses MUST follow the MCP response format with a "content" array
        4. Success responses have a "content" array with at least one content item
        5. Error responses MUST have "isError": true and a "content" array with error message
        6. The content array items must have a "type" field (typically "text" or "resource")
        7. For text content, include a "text" field with the actual content
        8. For resource content, include a "resource" field with uri, mimeType, and content
        9. All required arguments must be validated before generating success responses
        10. All argument types and formats must match the tool_schema exactly
        11. Always reflect and respect prior state if applicable
        12. Generate realistic and contextually appropriate responses based on the tool's description
        13. If the tool is meant to perform an action, simulate that action's success or failure
        14. If the tool is meant to retrieve data, generate realistic data based on the arguments

        ---
        📖 MCP TOOL RESPONSE GUIDELINES:

        1. **Understand the tool's purpose**: Read the tool description carefully to understand what it's meant to do
        2. **Validate inputs thoroughly**: Check all required arguments and their types/formats
        3. **Generate contextual responses**: The response should match what the tool is described to do
        4. **Use appropriate content types**:
           - Use "text" type for most responses
           - Use "resource" type when returning files or structured resources
        5. **Be consistent with state**: If this tool was called before with similar inputs, ensure responses are consistent
        6. **Simulate realistic behavior**:
           - File operations should reference actual paths from inputs
           - Database operations should return realistic records
           - API calls should return realistic data structures
        7. **Handle edge cases**: Validate for missing files, invalid inputs, etc.

        """

API_TOOL_RESPONSE_GENERATION_PROMPT = """
        You are simulating the return value of an API action call based on the following context:

        The tool {tool_name} has the following OpenAPI-style input schema:

        {input_schema}

        Output Schema:
        {output_schema}

        Now generate the return value for the following action call with the following user-provided API payload:
        {user_payload}

        You MUST validate this payload AND evaluate consistency with previously returned results or system state, described below.
        🧠 Prior State Context:
        {previous_responses}
        🎯 Your goal is to simulate a **realistic and schema-compliant tool response** for the given input payload. You must validate the payload **strictly against the input schema** and consider any constraints from prior state.

        ---
        🧪 CRITICAL INPUT VALIDATION RULES:
        1. Check the input schema for a `"required"` field list.
        2. If **ANY** required field is **missing** in the payload, respond with a **400 Bad Request** error.
        3. If all required fields are present but **any field has incorrect format or value**, respond with a **422 Unprocessable Entity** error.
        4. Do NOT respond with success unless ALL validation checks pass.
        5. You MUST simulate an error even if the request looks "reasonable" — strictly follow the schema's rules., treat this tool as deprecated and simulate an error.

        ---
        🔧 CRITICAL RESPONSE FORMAT REQUIREMENTS:
        ALL responses MUST include a "status" field with the appropriate HTTP status code.
        This is MANDATORY - no response should be generated without a status field.

        ---
        You MUST use the correct HTTP status code based on the tool description and operation summary:

        ✅ SUCCESS SCENARIOS
        Use status 201 only if:
        The operation is a creation action, as clearly indicated in the tool's description or summary (e.g., "create", "add", "register", "generate new", etc.)
        These are typically POST or PUT methods used specifically for creating new resources

        Use status 200 for all other successful actions, including:

        GET requests (reading or retrieving data)

        DELETE requests (removal of resources)

        POST or PUT requests that update, modify, or perform non-creation tasks

        ❗ Do NOT use 201 unless the tool description/summary explicitly refers to creation of a new resource.

        ---
        📘 SUCCESS RESPONSE RULE:
        ✅ Use only if:
        - All **required** fields from the schema are present
        - All fields match their expected **types**, **formats**, and **enum constraints**
        - Prior state allows successful completion (e.g., not blocked, rate-limited, etc.)

        🚨 STRICT SCHEMA COMPLIANCE:
        - You MUST strictly follow the provided schema's "required" fields list
        - Even if a field seems logically optional, if it's marked as "required" in the schema, it MUST be present
        - Do NOT make assumptions about what fields are "really" needed - follow the schema exactly
        - If ANY required field is missing, return a 400 error immediately

        SUCCESS RESPONSE STRUCTURE:
        ```json
        {{
          "status": 200,
          "data": {{
            // actual response data matching the schema - DO NOT add another "data" field here
          }}
        }}
        ```

        🚨 CRITICAL: AVOID NESTED DATA FIELDS
        - The "data" field should contain the actual response content directly
        - DO NOT create nested structures like {{"data": {{"data": [...]}}}}
        - If the schema expects an array, put it directly in "data": [...]
        - If the schema expects an object, put the object properties directly in "data": {{...}}

        EXAMPLE SUCCESS RESPONSES:

        For object response (status 200):
        ```json
        {{
          "status": 200,
          "data": {{
            "managerAlias": "mjones",
            "managerLevel": "L6", 
            "teamName": "Engineering",
            "managerName": "Mike Jones"
          }}
        }}
        ```

        For array response (status 200):
        ```json
        {{
          "status": 200,
          "data": [
            {{
              "id": "1",
              "name": "Item 1"
            }},
            {{
              "id": "2", 
              "name": "Item 2"
            }}
          ]
        }}
        ```

        For creation response (status 201):
        ```json
        {{
          "status": 201,
          "data": {{
            "id": "12345",
            "name": "New Resource",
            "createdAt": "2024-01-15T10:30:00Z",
            "status": "created"
          }}
        }}
        ```

        ---
        📘 ERROR RESPONSE RULES:
        ❌ Use **400** (`MISSING_REQUIRED_FIELD`) if:
        - One or more required fields are missing

        ❌ Use **422** (`INVALID_FORMAT`) if:
        - Any field has invalid type, format, or unacceptable value
        

        **ERROR TYPE 1: Missing Required Field (status 400)**
        - Trigger when ANY required field from the schema is missing from the payload
        - STRICTLY follow the schema's "required" fields list - do NOT make exceptions
        - Even if the field seems logically optional, if it's in "required", it MUST be present
        - MUST use status code 400
        
        ```json
        {{
          "status": 400,
          "error": {{
            "code": "MISSING_REQUIRED_FIELD",
            "message": "Missing required field: [field_name]",
            "field": "[field_name]"
          }}
        }}
        ```

        **ERROR TYPE 2: Invalid Format or Value (status 422)**
        - Trigger when all required fields are present BUT one or more fields have:
          * Wrong data type (string instead of number, etc.)
          * Invalid format (malformed email, invalid date, etc.)
          * Unacceptable enum values
          * Values that don't meet validation constraints
        - MUST use status code 422

        ```json
        {{
          "status": 422,
          "error": {{
            "code": "INVALID_FORMAT",
            "message": "Invalid [field_name] format",
            "field": "[field_name]"
          }}
        }}
        ```

        EXAMPLE ERROR RESPONSES:

        Missing required field example:
        ```json
        {{
          "status": 400,
          "error": {{
            "code": "MISSING_REQUIRED_FIELD",
            "message": "Missing required field: alias",
            "field": "alias"
          }}
        }}
        ```

        Invalid format example:
        ```json
        {{
          "status": 422,
          "error": {{
            "code": "INVALID_FORMAT", 
            "message": "Invalid email format",
            "field": "email"
          }}
        }}
        ```

        ---
        🔍 FINAL VALIDATION CHECK:
        Before generating any response, perform this validation:
        1. Check the schema's "required" fields list
        2. Verify EVERY required field is present in the user payload
        3. If ANY required field is missing, return status 400 error immediately
        4. Do NOT proceed with success response if required fields are missing
        
        ---
        ⚠️ STRICT FORMAT RULES:

        1. DO NOT include markdown, commentary, or explanation in your response
        2. Output must be valid raw JSON only
        3. ALL responses MUST include a "status" field with the appropriate HTTP status code
        4. Success responses MUST have a "data" field containing the actual response data
        5. Error responses MUST have an "error" field containing the error details
        6. All required fields must be included in success responses
        7. All field types and enum values must match the schema exactly
        8. Error responses MUST follow the exact format shown above with correct status codes
        9. Do NOT generate plain error messages — use the structured error format shown above
        10. Always reflect and respect prior state if applicable (e.g., account blocked = fail)
        11. The status field is MANDATORY in every single response
        12. STRICTLY follow the schema's required fields - no exceptions or assumptions
        13. 🚨 CRITICAL: DO NOT create nested "data" fields - put response content directly in the "data" field
        14. If schema expects an array, use "data": [...] NOT "data": {{"data": [...]}}
        15. If schema expects an object, use "data": {{...}} NOT "data": {{"data": {{...}}}}
        
        """

UNIFIED_TOOL_RESPONSE_GENERATION_PROMPT = """
        You are simulating the execution of the tool '{tool_name}'.

        Tool Input Schema:
        {input_schema}

        Output Schema:
        {output_schema}

        User Input Payload:
        {user_payload}

        Available State Context:
        {previous_responses}

        IMPORTANT:
        - Simulate what this tool would return when called with the provided parameters
        - Use the state context and call history to provide consistent responses
        - Validate parameters against the tool schema strictly
        - Validate the output against the output schema if provided
        - Return realistic responses that match the tool's purpose and output schema


        VALIDATION RULES:
        1. Check required parameters from the schema - if any are missing, return a validation error
        2. Check parameter types - if any have wrong types, return a validation error
        3. Only return success if all parameters are valid

        Generate only valid JSON with no markdown or explanation.
        """
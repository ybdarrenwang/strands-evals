from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field


class ToolType(Enum):
    """
    Enumeration of supported tool types for simulation.

    Attributes:
        FUNCTION: Python function tools that can be called directly.
        MCP: Model Context Protocol tools with structured schemas.
        API: REST API endpoints with HTTP methods and paths.
    """

    FUNCTION = "function"
    MCP = "mcp"
    API = "api"


class RegisteredTool(BaseModel):
    """
    Represents a registered tool in the simulator.

    Attributes:
        name: Name of the tool for identification and registration.
        tool_type: Type of the tool (FUNCTION, MCP, or API).
        function: Function callable for FUNCTION type tools (excluded from serialization).
        output_schema: Pydantic BaseModel for output schema (excluded from serialization).
        prompt_template: Custom prompt template override.
        mcp_schema: MCP tool schema dictionary for MCP type tools.
        api_path: API endpoint path for API type tools.
        api_method: HTTP method for API type tools (GET, POST, etc.).
        initial_state_description: Initial state description for the tool's context.
        simulator_kwargs: Additional simulator configuration parameters.
    """

    name: str = Field(..., description="Name of the tool")
    tool_type: ToolType = Field(default=ToolType.FUNCTION, description="Type of the tool")
    function: Callable | None = Field(default=None, description="Function callable", exclude=True)
    output_schema: type[BaseModel] | None = Field(
        default=None, description="Pydantic BaseModel for output schema", exclude=True
    )
    prompt_template: str | None = Field(default=None, description="Custom prompt template override")
    mcp_schema: dict[str, Any] | None = Field(default=None, description="MCP tool schema")
    api_path: str | None = Field(default=None, description="API endpoint path")
    api_method: str | None = Field(default=None, description="HTTP method")
    initial_state_description: str | None = Field(
        default=None, description="Initial state description for the tool's context"
    )
    share_state_id: str | None = Field(default=None, description="Shared state identifier for state sharing")
    simulator_kwargs: dict[str, Any] | None = Field(
        default_factory=dict, description="Additional simulator configuration"
    )

    model_config = {"arbitrary_types_allowed": True}


class ContentBlock(BaseModel):
    """Individual content item in MCP response following official MCP specification."""

    type: str = Field(..., description="Type of content (text, resource, etc.)")
    text: str | None = Field(default=None, description="Text content")
    resource: dict[str, Any] | None = Field(default=None, description="Resource information")


class MCPToolResponse(BaseModel):
    """
    Response model for MCP tool simulation following official MCP specification.

    Matches the official MCP ToolResultContent format for consistency with MCP ecosystem.
    """

    tool_use_id: str = Field(..., description="Unique identifier corresponding to tool call's id")
    content: list[ContentBlock] = Field(
        default_factory=list, description="List of content objects representing tool result"
    )
    structured_content: dict[str, Any] | None = Field(
        default=None, description="Structured tool output matching outputSchema"
    )
    is_error: bool | None = Field(default=False, description="Whether tool execution resulted in error")
    meta: dict[str, Any] | None = Field(default=None, description="Metadata following MCP specification")


class APIErrorDetail(BaseModel):
    """
    Error detail structure for API responses following RFC 9457 (Problem Details for HTTP APIs).
    
    Conforms to RFC 9457 standard referenced in OpenAPI v3.2.0 specification.
    All fields are optional per RFC 9457, but type, title, and detail are required here for consistency.
    Supports extension members via extra="allow" configuration.
    
    Reference: https://www.rfc-editor.org/rfc/rfc9457.html
    """

    type: str = Field(..., description="Error type identifier (URI reference identifying the problem type)")
    title: str = Field(..., description="Human-readable error title (short summary of the problem)")
    detail: str = Field(..., description="Detailed error description (human-readable explanation specific to this occurrence)")
    status: int | None = Field(default=None, description="HTTP status code (for convenience, though also available in APIToolResponse)")
    instance: str | None = Field(default=None, description="URI reference identifying the specific occurrence of the problem")
    
    model_config = {"extra": "allow"}  # Allow RFC 9457 extension members


class APIToolResponse(BaseModel):
    """
    Runtime response wrapper for API tool simulation.
    
    This class represents the actual response returned during tool execution, not an OpenAPI
    schema definition. It wraps HTTP responses in a consistent format for simulation purposes.
    
    Key Differences from OpenAPI Response Object:
    - This is a runtime response model (data returned during execution)
    - OpenAPI Response Object is a specification/documentation structure (describes API contract)
    - Status code is a field here; in OpenAPI it's a key in the Responses Object
    - Uses simple data/error fields; OpenAPI uses media-type mapping and content negotiation
    
    Structure:
    - status: HTTP status code (200, 404, 500, etc.)
    - data: Response payload for successful operations (2xx status codes)
    - error: RFC 9457-compliant error details for failures (4xx, 5xx status codes)
    
    Note: If OpenAPI schema generation is needed, a separate converter should be implemented
    to transform this runtime model into OpenAPI Response Object format.
    """

    status: int = Field(..., description="HTTP status code")
    data: Any | None = Field(default=None, description="Response data for successful requests")
    error: APIErrorDetail | None = Field(default=None, description="Error details for failed requests")

    # Allow additional fields for flexibility
    model_config = {"extra": "allow"}
"""OpenEnv client: PermitForge(url).sync(), then reset and CallToolAction steps."""
from openenv.core.mcp_client import MCPToolClient


class PermitForge(MCPToolClient):
    """One PermitForge tool session, graded once on submit_design."""

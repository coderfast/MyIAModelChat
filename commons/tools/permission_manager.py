"""Permission Manager for agentic tool execution.

Implements Layer 1 of the 3-layer security model:
- Layer 1: PermissionManager (this module) - asks user for permission
- Layer 2: ShellSecurity (in tool_executor.py) - validates commands
- Layer 3: Model thinking - informational warnings only
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk classification for tool execution."""
    NONE = "none"           # Auto-approve (calculator, current_date, etc.)
    LOW = "low"             # Auto-approve for read operations
    MEDIUM = "medium"       # Auto-approve for whitelisted shell commands
    HIGH = "high"           # Requires explicit user permission
    CRITICAL = "critical"   # Always blocked


class PermissionMode(Enum):
    """Operation modes for permission requests."""
    INTERACTIVE = "interactive"    # Ask user for every tool call
    AUTO_APPROVE = "auto_approve"  # Auto-approve all (testing/development)
    DRY_RUN = "dry_run"           # Show command without executing
    WHITELIST = "whitelist"        # Auto-approve read operations only


@dataclass
class PermissionRequest:
    """Represents a permission request for tool execution."""
    tool_name: str
    command: str
    platform: str
    risk_level: RiskLevel
    timeout: int = 30
    dry_run: bool = False


@dataclass
class PermissionResult:
    """Result of a permission request."""
    approved: bool
    mode: PermissionMode
    reason: str = ""
    dry_run_output: Optional[str] = None


class PermissionManager:
    """Manages permissions for tool execution.
    
    The model ONLY generates tool_call. Actual execution is the responsibility
    of the agent, which MUST ask the user for permission before executing.
    
    Security Model:
    - NONE risk: Auto-approved (calculator, current_date, word_count, get_platform)
    - LOW risk: Auto-approved for read operations (read_file, list_directory)
    - MEDIUM risk: Auto-approved for whitelisted shell commands
    - HIGH risk: Requires explicit user permission
    - CRITICAL risk: Always blocked by ShellSecurity
    """
    
    # Tools that auto-approve (no permission needed)
    AUTO_APPROVE_TOOLS = {
        'calculator', 'current_date', 'word_count', 'get_platform'
    }
    
    # Tools that require permission for write operations
    SHELL_TOOLS = {'shell', 'powershell', 'bash'}
    
    def __init__(self, mode: PermissionMode = PermissionMode.INTERACTIVE,
                 timeout: int = 30):
        """Initialize PermissionManager.
        
        Args:
            mode: Operation mode (interactive, auto_approve, dry_run, whitelist)
            timeout: Default timeout for shell commands in seconds
        """
        self.mode = mode
        self.timeout = timeout
        self.permission_log: list[dict] = []
        logger.info(f"PermissionManager initialized in {mode.value} mode")
    
    def classify_risk(self, tool_name: str, command: str = "") -> RiskLevel:
        """Classify the risk level of a tool execution request.
        
        Args:
            tool_name: Name of the tool to execute
            command: The command to execute (for shell tools)
            
        Returns:
            RiskLevel classification
        """
        if tool_name in self.AUTO_APPROVE_TOOLS:
            return RiskLevel.NONE
        
        if tool_name not in self.SHELL_TOOLS:
            # Unknown tool - require permission
            return RiskLevel.HIGH
        
        # For shell tools, analyze the command
        if not command:
            return RiskLevel.MEDIUM
        
        # Check for dangerous patterns
        critical_patterns = [
            'rm -rf', 'format c:', 'Format-Volume', 'Initialize-Disk',
            'shutdown', 'reboot', 'poweroff',
            'Remove-Item -Recurse -Force',
        ]
        dangerous_patterns = [
            'sudo', 'del /s', 'Remove-Item -Recurse',
            'chmod 777', 'chmod -R 777', 'chown',
            '> /dev/null', '2>&1', '|',
            '&&', '||',
        ]
        
        command_lower = command.lower()
        for pattern in critical_patterns:
            if pattern.lower() in command_lower:
                return RiskLevel.CRITICAL
        for pattern in dangerous_patterns:
            if pattern.lower() in command_lower:
                return RiskLevel.HIGH
        
        # Check if it's a read-only command
        read_commands = [
            'Get-ChildItem', 'Get-Content', 'Get-Item', 'ls', 'dir',
            'cat', 'head', 'tail', 'wc', 'grep', 'find', 'ps', 'top',
            'Get-Process', 'Get-Service', 'Get-Date', 'Get-Host',
        ]
        
        for read_cmd in read_commands:
            if read_cmd.lower() in command_lower:
                return RiskLevel.LOW
        
        return RiskLevel.MEDIUM
    
    def format_permission_request(self, request: PermissionRequest) -> str:
        """Format a permission request for display to the user.
        
        Args:
            request: The permission request to format
            
        Returns:
            Formatted string for display
        """
        risk_indicator = {
            RiskLevel.NONE: "[AUTO]",
            RiskLevel.LOW: "[READ]",
            RiskLevel.MEDIUM: "[SHELL]",
            RiskLevel.HIGH: "[PERMISSION REQUIRED]",
            RiskLevel.CRITICAL: "[BLOCKED]"
        }
        
        lines = [
            "=" * 50,
            "TOOL CALL REQUEST",
            "=" * 50,
            f"Tool:      {request.tool_name}",
            f"Platform:  {request.platform}",
            f"Risk:      {risk_indicator.get(request.risk_level, '[UNKNOWN]')}",
            f"Timeout:   {request.timeout}s",
            "-" * 50,
            f"Command:",
            f"  {request.command}",
            "-" * 50,
        ]
        
        if request.dry_run:
            lines.append("Mode: DRY RUN (command will NOT be executed)")
            lines.append("-" * 50)
        
        if request.risk_level == RiskLevel.HIGH:
            lines.append("This command requires your permission to execute.")
            lines.append("Enter: 'yes' to approve, 'no' to deny, 'dry-run' to preview")
        elif request.risk_level == RiskLevel.CRITICAL:
            lines.append("BLOCKED: This command is always blocked for security.")
        else:
            lines.append("Auto-approved (safe operation)")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def request_permission(self, tool_name: str, command: str, 
                          platform: str, timeout: Optional[int] = None) -> PermissionResult:
        """Request permission to execute a tool.
        
        Args:
            tool_name: Name of the tool to execute
            command: The command to execute
            platform: Current platform (windows, linux, darwin)
            timeout: Optional timeout override
            
        Returns:
            PermissionResult with approval status
        """
        if timeout is None:
            timeout = self.timeout
        
        risk_level = self.classify_risk(tool_name, command)
        
        request = PermissionRequest(
            tool_name=tool_name,
            command=command,
            platform=platform,
            risk_level=risk_level,
            timeout=timeout,
            dry_run=(self.mode == PermissionMode.DRY_RUN)
        )
        
        # Log the request
        log_entry = {
            'tool_name': tool_name,
            'command': command,
            'platform': platform,
            'risk_level': risk_level.value,
            'mode': self.mode.value,
        }
        
        # CRITICAL is always blocked, regardless of mode
        if risk_level == RiskLevel.CRITICAL:
            log_entry['approved'] = False
            log_entry['reason'] = 'blocked: critical risk'
            self.permission_log.append(log_entry)
            return PermissionResult(
                approved=False,
                mode=self.mode,
                reason="Blocked: critical risk command"
            )

        # Handle based on mode
        if self.mode == PermissionMode.AUTO_APPROVE:
            log_entry['approved'] = True
            log_entry['reason'] = 'auto_approve mode'
            self.permission_log.append(log_entry)
            logger.info(f"Auto-approved: {tool_name}")
            return PermissionResult(
                approved=True,
                mode=self.mode,
                reason="Auto-approved mode"
            )
        
        if self.mode == PermissionMode.DRY_RUN:
            formatted = self.format_permission_request(request)
            log_entry['approved'] = False
            log_entry['reason'] = 'dry_run mode'
            log_entry['preview'] = formatted
            self.permission_log.append(log_entry)
            logger.info(f"Dry run preview for: {tool_name}")
            return PermissionResult(
                approved=False,
                mode=self.mode,
                reason="Dry run mode - command not executed",
                dry_run_output=formatted
            )
        
        # Risk-based auto-approval
        if risk_level == RiskLevel.NONE:
            log_entry['approved'] = True
            log_entry['reason'] = 'auto-approve: none risk'
            self.permission_log.append(log_entry)
            return PermissionResult(
                approved=True,
                mode=self.mode,
                reason="Auto-approved: no risk"
            )
        
        if risk_level == RiskLevel.LOW and self.mode == PermissionMode.WHITELIST:
            log_entry['approved'] = True
            log_entry['reason'] = 'auto-approve: whitelist mode, low risk'
            self.permission_log.append(log_entry)
            return PermissionResult(
                approved=True,
                mode=self.mode,
                reason="Auto-approved: whitelist mode, read operation"
            )
        
        if risk_level == RiskLevel.CRITICAL:
            log_entry['approved'] = False
            log_entry['reason'] = 'blocked: critical risk'
            self.permission_log.append(log_entry)
            return PermissionResult(
                approved=False,
                mode=self.mode,
                reason="Blocked: critical risk command"
            )
        
        if risk_level == RiskLevel.MEDIUM and self.mode == PermissionMode.WHITELIST:
            log_entry['approved'] = True
            log_entry['reason'] = 'auto-approve: whitelist mode, medium risk'
            self.permission_log.append(log_entry)
            return PermissionResult(
                approved=True,
                mode=self.mode,
                reason="Auto-approved: whitelist mode"
            )
        
        # Interactive mode - would normally ask user
        # For non-interactive contexts, return the formatted request
        formatted = self.format_permission_request(request)
        log_entry['preview'] = formatted
        self.permission_log.append(log_entry)
        
        # In interactive mode, we return the request for the caller to handle
        # The actual user input would be handled by the chat engine
        return PermissionResult(
            approved=False,
            mode=self.mode,
            reason="Interactive mode - awaiting user response",
            dry_run_output=formatted
        )
    
    def get_log(self) -> list[dict]:
        """Return the permission log."""
        return self.permission_log.copy()
    
    def clear_log(self):
        """Clear the permission log."""
        self.permission_log.clear()

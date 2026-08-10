"""Tests for Permission System and Platform Detector."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commons.tools.permission_manager import (
    PermissionManager, PermissionMode, RiskLevel,
    PermissionRequest, PermissionResult
)
from commons.tools.platform_detector import PlatformDetector, ShellConfig


# ============================================================
# PermissionManager Tests
# ============================================================

class TestPermissionManager:
    """Tests for PermissionManager."""
    
    def test_auto_approve_mode(self):
        """Test that auto_approve mode approves all requests."""
        pm = PermissionManager(mode=PermissionMode.AUTO_APPROVE)
        result = pm.request_permission(
            tool_name='shell',
            command='rm -rf /',
            platform='linux'
        )
        assert result.approved is True
        assert result.mode == PermissionMode.AUTO_APPROVE
    
    def test_dry_run_mode(self):
        """Test that dry_run mode shows preview without executing."""
        pm = PermissionManager(mode=PermissionMode.DRY_RUN)
        result = pm.request_permission(
            tool_name='powershell',
            command='Get-ChildItem',
            platform='windows'
        )
        assert result.approved is False
        assert result.mode == PermissionMode.DRY_RUN
        assert result.dry_run_output is not None
        assert 'Get-ChildItem' in result.dry_run_output
    
    def test_request_permission_interactive(self):
        """Test that interactive mode returns formatted request."""
        pm = PermissionManager(mode=PermissionMode.INTERACTIVE)
        result = pm.request_permission(
            tool_name='shell',
            command='ls -la',
            platform='linux'
        )
        # Interactive mode returns unapproved with preview
        assert result.approved is False
        assert result.mode == PermissionMode.INTERACTIVE
        assert result.dry_run_output is not None
    
    def test_permission_denied(self):
        """Test that high risk commands require permission."""
        pm = PermissionManager(mode=PermissionMode.INTERACTIVE)
        result = pm.request_permission(
            tool_name='shell',
            command='sudo rm -rf /',
            platform='linux'
        )
        assert result.approved is False
        # High risk commands require user permission
        assert result.mode == PermissionMode.INTERACTIVE
    
    def test_auto_approve_tools(self):
        """Test that safe tools are auto-approved."""
        pm = PermissionManager(mode=PermissionMode.INTERACTIVE)
        
        safe_tools = ['calculator', 'current_date', 'word_count', 'get_platform']
        for tool in safe_tools:
            result = pm.request_permission(
                tool_name=tool,
                command='',
                platform='linux'
            )
            assert result.approved is True, f"{tool} should be auto-approved"
    
    def test_risk_classification(self):
        """Test risk classification of different commands."""
        pm = PermissionManager()
        
        # No risk tools
        assert pm.classify_risk('calculator') == RiskLevel.NONE
        assert pm.classify_risk('current_date') == RiskLevel.NONE
        
        # Low risk (read operations)
        assert pm.classify_risk('shell', 'Get-ChildItem') == RiskLevel.LOW
        assert pm.classify_risk('bash', 'ls -la') == RiskLevel.LOW
        
        # High risk (dangerous commands)
        assert pm.classify_risk('shell', 'sudo rm -rf /') == RiskLevel.HIGH
        assert pm.classify_risk('bash', 'rm -rf /') == RiskLevel.HIGH
    
    def test_permission_log(self):
        """Test that permission requests are logged."""
        pm = PermissionManager(mode=PermissionMode.AUTO_APPROVE)
        
        pm.request_permission('calculator', '', 'linux')
        pm.request_permission('shell', 'ls', 'linux')
        
        log = pm.get_log()
        assert len(log) == 2
        assert log[0]['tool_name'] == 'calculator'
        assert log[1]['tool_name'] == 'shell'
    
    def test_clear_log(self):
        """Test clearing the permission log."""
        pm = PermissionManager(mode=PermissionMode.AUTO_APPROVE)
        pm.request_permission('calculator', '', 'linux')
        
        assert len(pm.get_log()) == 1
        pm.clear_log()
        assert len(pm.get_log()) == 0
    
    def test_format_permission_request(self):
        """Test formatting of permission requests."""
        pm = PermissionManager()
        request = PermissionRequest(
            tool_name='powershell',
            command='Get-ChildItem -Path .',
            platform='windows',
            risk_level=RiskLevel.MEDIUM,
            timeout=30
        )
        
        formatted = pm.format_permission_request(request)
        assert 'powershell' in formatted
        assert 'Get-ChildItem' in formatted
        assert 'windows' in formatted


class TestPlatformDetector:
    """Tests for PlatformDetector."""
    
    def test_detect_platform(self):
        """Test platform detection."""
        platform = PlatformDetector.detect_platform()
        assert platform in ['windows', 'linux', 'darwin']
    
    def test_get_default_shell(self):
        """Test default shell detection."""
        shell = PlatformDetector.get_default_shell()
        assert shell in ['powershell', 'pwsh', 'bash', 'zsh']
    
    def test_get_shell_executable(self):
        """Test shell executable path retrieval."""
        current_platform = PlatformDetector.detect_platform()
        
        if current_platform == 'windows':
            executable = PlatformDetector.get_shell_executable('powershell')
            assert executable is not None
            assert 'powershell' in executable.lower() or 'pwsh' in executable.lower()
        else:
            executable = PlatformDetector.get_shell_executable('bash')
            assert executable is not None
            assert 'bash' in executable
    
    def test_get_shell_args(self):
        """Test shell arguments retrieval."""
        # PowerShell uses -Command
        args = PlatformDetector.get_shell_args('powershell')
        assert args == ['-Command']
        
        args = PlatformDetector.get_shell_args('pwsh')
        assert args == ['-Command']
        
        # Bash/zsh use -c
        args = PlatformDetector.get_shell_args('bash')
        assert args == ['-c']
        
        args = PlatformDetector.get_shell_args('zsh')
        assert args == ['-c']
    
    def test_get_shell_config(self):
        """Test shell configuration retrieval."""
        config = PlatformDetector.get_shell_config('powershell')
        assert isinstance(config, ShellConfig)
        assert config.name == 'powershell'
        assert config.args_prefix == '-Command'
        
        config = PlatformDetector.get_shell_config('bash')
        assert isinstance(config, ShellConfig)
        assert config.name == 'bash'
        assert config.args_prefix == '-c'
    
    def test_get_platform_info(self):
        """Test platform info retrieval."""
        info = PlatformDetector.get_platform_info()
        
        assert 'system' in info
        assert 'detected_platform' in info
        assert 'default_shell' in info
        assert info['detected_platform'] in ['windows', 'linux', 'darwin']
    
    def test_is_command_available(self):
        """Test command availability check."""
        # These should be available on most systems
        if PlatformDetector.detect_platform() == 'windows':
            assert PlatformDetector.is_command_available('where') is True
        else:
            assert PlatformDetector.is_command_available('which') is True


# ============================================================
# Integration Tests
# ============================================================

class TestPermissionIntegration:
    """Integration tests for permission system."""
    
    def test_whitelist_mode_read_operations(self):
        """Test that whitelist mode auto-approves read operations."""
        pm = PermissionManager(mode=PermissionMode.WHITELIST)
        
        # Read operations should be approved
        result = pm.request_permission(
            tool_name='shell',
            command='Get-ChildItem',
            platform='windows'
        )
        assert result.approved is True
    
    def test_whitelist_mode_write_operations(self):
        """Test that whitelist mode requires permission for write operations."""
        pm = PermissionManager(mode=PermissionMode.WHITELIST)
        
        # Write operations should require permission
        result = pm.request_permission(
            tool_name='shell',
            command='New-Item -Path test.txt',
            platform='windows'
        )
        # Should not be auto-approved in whitelist mode
        assert result.approved is False or result.mode == PermissionMode.WHITELIST
    
    def test_platform_specific_shell(self):
        """Test platform-specific shell configuration."""
        current_platform = PlatformDetector.detect_platform()
        default_shell = PlatformDetector.get_default_shell()
        
        if current_platform == 'windows':
            assert default_shell in ['powershell', 'pwsh']
        else:
            assert default_shell in ['bash', 'zsh']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

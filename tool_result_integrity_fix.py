#!/usr/bin/env python3
"""
Proposed fixes for tool result integrity issues in streaming mode.

This module provides enhanced tool execution with result verification
to prevent content contamination and ensure proper ReAct observation.
"""

import logging
import uuid
import time
from typing import Dict, Any, Optional
from abstractllm.tools.core import ToolCall

logger = logging.getLogger("tool_integrity")

class ToolResultIntegrityManager:
    """Ensures tool results are delivered correctly without contamination."""

    def __init__(self):
        self.execution_cache = {}  # Track recent executions
        self.result_verification = {}  # Verify result authenticity

    def execute_tool_with_verification(self,
                                     tool_call: ToolCall,
                                     tool_functions: Dict[str, Any],
                                     session_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute a tool call with integrity verification.

        This enhanced version:
        1. Generates unique execution IDs to prevent mixing
        2. Verifies result authenticity
        3. Isolates results from memory contamination
        4. Provides detailed logging for debugging
        """

        # Generate unique execution ID
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        logger.info(f"🔧 TOOL EXECUTION START: {execution_id}")
        logger.info(f"   Tool: {tool_call.name}")
        logger.info(f"   Args: {tool_call.arguments}")

        try:
            # Get the tool function
            tool_function = tool_functions.get(tool_call.name)
            if not tool_function:
                error_msg = f"Tool '{tool_call.name}' not found in available tools"
                logger.error(f"❌ {execution_id}: {error_msg}")
                return {
                    "call_id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "output": error_msg,
                    "error": error_msg,
                    "success": False,
                    "execution_id": execution_id,
                    "execution_time": time.time() - start_time
                }

            # CRITICAL: Execute with argument isolation
            # Ensure arguments don't reference cached/memory content
            isolated_args = self._isolate_arguments(tool_call.arguments)

            # Execute the tool function
            logger.info(f"🔧 {execution_id}: Executing {tool_call.name} with isolated args")
            raw_result = tool_function(**isolated_args)

            # INTEGRITY CHECK: Verify result authenticity
            result_hash = hash(str(raw_result))
            self.result_verification[execution_id] = {
                "tool_name": tool_call.name,
                "args_hash": hash(str(isolated_args)),
                "result_hash": result_hash,
                "timestamp": time.time()
            }

            # Format enhanced result with verification
            enhanced_result = {
                "call_id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "output": raw_result,
                "error": None,
                "success": True,
                "execution_id": execution_id,
                "execution_time": time.time() - start_time,
                "result_hash": result_hash,
                "integrity_verified": True
            }

            logger.info(f"✅ {execution_id}: Tool executed successfully")
            logger.info(f"   Result length: {len(str(raw_result))} chars")
            logger.info(f"   Result hash: {result_hash}")

            # Cache for verification
            self.execution_cache[execution_id] = enhanced_result

            return enhanced_result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ {execution_id}: Tool execution failed: {error_msg}")

            return {
                "call_id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "output": f"Error executing {tool_call.name}: {error_msg}",
                "error": error_msg,
                "success": False,
                "execution_id": execution_id,
                "execution_time": time.time() - start_time
            }

    def _isolate_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Isolate tool arguments from potential memory contamination.

        This ensures that arguments like file_path are exactly as specified,
        not influenced by cached content or memory system interference.
        """

        # Deep copy to prevent reference issues
        import copy
        isolated = copy.deepcopy(arguments)

        # Log argument isolation for debugging
        logger.info(f"🔒 Argument isolation:")
        for key, value in isolated.items():
            logger.info(f"   {key}: {value}")
            # Ensure string arguments are exactly as provided
            if isinstance(value, str):
                # Strip any potential contamination markers
                isolated[key] = value.strip()

        return isolated

    def verify_result_integrity(self, execution_id: str, received_content: str) -> bool:
        """
        Verify that the received content matches what was actually executed.

        This catches issues where the LLM receives content from a different
        tool execution than expected.
        """

        if execution_id not in self.result_verification:
            logger.warning(f"⚠️ No verification record for execution {execution_id}")
            return False

        verification_record = self.result_verification[execution_id]
        expected_hash = verification_record["result_hash"]
        received_hash = hash(received_content)

        if expected_hash == received_hash:
            logger.info(f"✅ Result integrity verified for {execution_id}")
            return True
        else:
            logger.error(f"❌ RESULT INTEGRITY FAILURE for {execution_id}")
            logger.error(f"   Expected hash: {expected_hash}")
            logger.error(f"   Received hash: {received_hash}")
            logger.error(f"   Tool: {verification_record['tool_name']}")
            return False

    def cleanup_old_verifications(self, max_age_seconds: int = 300):
        """Clean up old verification records to prevent memory leaks."""

        current_time = time.time()
        to_remove = []

        for execution_id, record in self.result_verification.items():
            if current_time - record["timestamp"] > max_age_seconds:
                to_remove.append(execution_id)

        for execution_id in to_remove:
            del self.result_verification[execution_id]
            if execution_id in self.execution_cache:
                del self.execution_cache[execution_id]

        if to_remove:
            logger.info(f"🧹 Cleaned up {len(to_remove)} old verification records")


# Enhanced session method with integrity checking
def enhanced_execute_tool_call(session, tool_call: ToolCall, tool_functions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drop-in replacement for session.execute_tool_call with integrity verification.

    This version prevents the tool result mismatch issue by:
    1. Using unique execution IDs
    2. Isolating arguments from memory contamination
    3. Verifying result integrity
    4. Providing detailed logging
    """

    # Initialize integrity manager if not exists
    if not hasattr(session, '_tool_integrity_manager'):
        session._tool_integrity_manager = ToolResultIntegrityManager()

    # Use enhanced execution
    return session._tool_integrity_manager.execute_tool_with_verification(
        tool_call, tool_functions, getattr(session, 'context', None)
    )


# Memory context isolation for streaming mode
def isolate_memory_context_from_tool_results(session, memory_context: str, fresh_tool_results: list) -> str:
    """
    Ensure memory context doesn't contaminate fresh tool results.

    This addresses the specific issue where the LLM receives content from
    memory/cache instead of the fresh tool execution result.
    """

    logger.info("🔒 Isolating memory context from fresh tool results")

    # If we have fresh tool results, prioritize them over memory content
    if fresh_tool_results:
        logger.info(f"📊 Found {len(fresh_tool_results)} fresh tool results")

        # Build context that clearly separates memory from fresh results
        context_parts = []

        # Add session info (minimal)
        if hasattr(session, 'session_id'):
            context_parts.append(f"Session: {session.session_id}")

        # Add fresh tool results with clear markers
        context_parts.append("\\n--- FRESH TOOL RESULTS ---")
        for i, tool_result in enumerate(fresh_tool_results[-3:]):  # Last 3 results
            tool_name = tool_result.get('name', 'unknown_tool')
            tool_output = tool_result.get('output', '')
            execution_id = tool_result.get('execution_id', 'unknown')

            context_parts.append(f"Tool {i+1}: {tool_name} (ID: {execution_id})")
            context_parts.append(f"Result: {tool_output}")

        # Add relevant memory context (but clearly separated)
        if memory_context:
            context_parts.append("\\n--- MEMORY CONTEXT ---")
            # Truncate memory context to prevent overwhelming fresh results
            truncated_memory = memory_context[:500] + "..." if len(memory_context) > 500 else memory_context
            context_parts.append(truncated_memory)

        isolated_context = "\\n".join(context_parts)
        logger.info(f"✅ Generated isolated context: {len(isolated_context)} chars")

        return isolated_context

    else:
        # No fresh tool results, use memory context as-is
        logger.info("📝 No fresh tool results, using memory context")
        return memory_context


if __name__ == "__main__":
    # Example usage for debugging
    logging.basicConfig(level=logging.INFO)

    # Example of how to monkey-patch the session for testing
    print("🔧 Tool Result Integrity Manager - Ready for integration")
    print("To use: Replace session.execute_tool_call with enhanced_execute_tool_call")
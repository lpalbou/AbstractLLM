"""
Enhanced Session management with integrated SOTA features.

This module extends the base Session with:
- Hierarchical memory system
- ReAct cycle management
- Retry strategies
- Structured response support
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Generator, TYPE_CHECKING
from datetime import datetime
from pathlib import Path
import json
import logging
import uuid

# Import base Session
from abstractllm.session import Session as BaseSession, Message, MessageRole
from abstractllm.interface import AbstractLLMInterface, ModelParameter

# Import SOTA improvements
from abstractllm.memory_v2 import HierarchicalMemory, ReActCycle, MemoryComponent
from abstractllm.retry_strategies import RetryManager, RetryConfig, with_retry
from abstractllm.structured_response import (
    StructuredResponseHandler, 
    StructuredResponseConfig,
    ResponseFormat
)
from abstractllm.scratchpad_manager import (
    ScratchpadManager, 
    get_scratchpad_manager,
    ReActPhase,
    CyclePhaseEvent
)

# Import types
if TYPE_CHECKING:
    from abstractllm.types import GenerateResponse

logger = logging.getLogger(__name__)


class EnhancedSession(BaseSession):
    """
    Enhanced session with hierarchical memory, retry strategies, and structured responses.
    """
    
    def __init__(self,
                 system_prompt: Optional[str] = None,
                 provider: Optional[Union[str, AbstractLLMInterface]] = None,
                 provider_config: Optional[Dict[Union[str, ModelParameter], Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                 # New parameters
                 enable_memory: bool = True,
                 memory_config: Optional[Dict[str, Any]] = None,
                 enable_retry: bool = True,
                 retry_config: Optional[RetryConfig] = None,
                 persist_memory: Optional[Path] = None):
        """
        Initialize enhanced session with SOTA features.
        
        Args:
            system_prompt: Default system prompt
            provider: LLM provider instance or name
            provider_config: Provider configuration
            metadata: Session metadata
            tools: Available tools
            enable_memory: Enable hierarchical memory system
            memory_config: Memory system configuration
            enable_retry: Enable retry strategies
            retry_config: Retry configuration
            persist_memory: Path to persist memory
        """
        # Initialize base session
        super().__init__(
            system_prompt=system_prompt,
            provider=provider,
            provider_config=provider_config,
            metadata=metadata,
            tools=tools
        )
        
        # Initialize hierarchical memory
        self.enable_memory = enable_memory
        if enable_memory:
            memory_cfg = memory_config or {}
            self.memory = HierarchicalMemory(
                working_memory_size=memory_cfg.get('working_memory_size', 10),
                episodic_consolidation_threshold=memory_cfg.get('consolidation_threshold', 5),
                persist_path=persist_memory
            )
        else:
            self.memory = None
        
        # Initialize retry manager
        self.enable_retry = enable_retry
        if enable_retry:
            self.retry_manager = RetryManager(retry_config or RetryConfig())
        else:
            self.retry_manager = None
        
        # Track current ReAct cycle
        self.current_cycle: Optional[ReActCycle] = None
        
        # Structured response handlers per provider
        self.response_handlers: Dict[str, StructuredResponseHandler] = {}
        
        # Enhanced telemetry tracking
        self.current_tool_traces: List[Dict[str, Any]] = []
        self.current_retry_attempts: int = 0
        self.facts_before_generation: int = 0
        
        # SOTA Scratchpad Manager with complete observability
        memory_folder = persist_memory.parent if persist_memory else Path("./memory")
        session_id = self.memory.session_id if self.memory else f"session_{uuid.uuid4().hex[:8]}"
        self.scratchpad = get_scratchpad_manager(session_id, memory_folder)
    
    def generate(self,
                prompt: Optional[str] = None,
                provider: Optional[Union[str, AbstractLLMInterface]] = None,
                system_prompt: Optional[str] = None,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None,
                top_p: Optional[float] = None,
                frequency_penalty: Optional[float] = None,
                presence_penalty: Optional[float] = None,
                tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                tool_functions: Optional[Dict[str, Callable[..., Any]]] = None,
                max_tool_calls: int = 25,
                stream: bool = False,
                files: Optional[List[Union[str, Path]]] = None,
                # New parameters
                use_memory_context: bool = True,
                create_react_cycle: bool = True,
                structured_config: Optional[StructuredResponseConfig] = None,
                **kwargs) -> "GenerateResponse":
        """
        Enhanced generate with memory, retry, and structured response support.
        
        Args:
            prompt: Input prompt
            provider: Provider override
            system_prompt: System prompt override
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            top_p: Top-p sampling
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            tools: Available tools
            tool_functions: Tool implementations
            max_tool_calls: Maximum tool iterations
            stream: Stream response
            files: Files to process
            use_memory_context: Include memory context
            create_react_cycle: Create ReAct cycle for this query
            structured_config: Structured response configuration
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        # Get provider
        provider_instance = self._get_provider(provider)
        provider_name = self._get_provider_name(provider_instance)
        
        # Initialize telemetry tracking
        generation_start_time = datetime.now()
        self.current_tool_traces = []  # Reset for this generation
        self.current_retry_attempts = 0
        
        # Track memory state before generation
        if self.enable_memory:
            self.facts_before_generation = len(self.memory.semantic_memory) if self.memory else 0
        
        # Start ReAct cycle if enabled
        if self.enable_memory and create_react_cycle and prompt:
            self.current_cycle = self.memory.start_react_cycle(
                query=prompt,
                max_iterations=max_tool_calls
            )
            # Add initial thought
            self.current_cycle.add_thought(
                f"Processing query with {provider_name} provider",
                confidence=1.0
            )
            
            # Start scratchpad cycle with complete observability
            self.scratchpad.start_cycle(self.current_cycle.cycle_id, prompt)
            self.scratchpad.add_thought(
                f"Processing query with {provider_name} provider",
                confidence=1.0,
                metadata={"provider": provider_name, "model": getattr(provider_instance, 'model', 'unknown')}
            )
        
        # Add memory context if enabled
        enhanced_prompt = prompt
        if self.enable_memory and use_memory_context and prompt:
            context = self.memory.get_context_for_query(prompt)
            if context:
                enhanced_prompt = f"{context}\n\nUser: {prompt}"
                logger.debug(f"Added memory context: {len(context)} chars")
        
        # Prepare for structured response if configured
        if structured_config:
            handler = self._get_response_handler(provider_name)
            request_params = handler.prepare_request(
                prompt=enhanced_prompt,
                config=structured_config,
                system_prompt=system_prompt
            )
            enhanced_prompt = request_params.pop("prompt")
            system_prompt = request_params.pop("system_prompt", system_prompt)
            kwargs.update(request_params)
        
        # Define generation function
        def _generate():
            if tools or tool_functions:
                # Use generate_with_tools for tool support
                return super(EnhancedSession, self).generate_with_tools(
                    prompt=enhanced_prompt,
                    provider=provider_instance,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    tools=tools,
                    tool_functions=tool_functions,
                    max_tool_calls=max_tool_calls,
                    files=files,
                    **kwargs
                )
            else:
                # Regular generation
                return super(EnhancedSession, self).generate(
                    prompt=enhanced_prompt,
                    provider=provider_instance,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stream=stream,
                    files=files,
                    **kwargs
                )
        
        # Apply retry if enabled
        if self.enable_retry:
            try:
                response = self.retry_manager.retry_with_backoff(
                    _generate,
                    key=f"{provider_name}_generate"
                )
            except Exception as e:
                if self.current_cycle:
                    self.current_cycle.error = str(e)
                    self.current_cycle.complete("Failed to generate response", success=False)
                raise
        else:
            response = _generate()
        
        # Parse structured response if configured
        if structured_config:
            handler = self._get_response_handler(provider_name)
            try:
                response = handler.parse_response(response, structured_config)
            except Exception as e:
                logger.error(f"Failed to parse structured response: {e}")
                if self.enable_retry and structured_config.max_retries > 0:
                    # Retry with feedback
                    response = handler.generate_with_retry(
                        generate_fn=provider_instance.generate,
                        prompt=prompt,
                        config=structured_config,
                        system_prompt=system_prompt,
                        **kwargs
                    )
        
        # Calculate total generation time
        total_generation_time = (datetime.now() - generation_start_time).total_seconds()
        
        # Update memory if enabled
        extracted_facts = []
        
        if self.enable_memory and prompt:
            # Track facts extracted before adding new messages
            facts_before = len(self.memory.semantic_memory)
            
            # Add to chat history (this triggers fact extraction)
            msg_id = self.memory.add_chat_message(
                role="user",
                content=prompt,
                cycle_id=self.current_cycle.cycle_id if self.current_cycle else None
            )
            
            # Add response
            response_content = str(response)
            resp_id = self.memory.add_chat_message(
                role="assistant",
                content=response_content,
                cycle_id=self.current_cycle.cycle_id if self.current_cycle else None
            )
            
            # Calculate facts extracted during this generation
            facts_after = len(self.memory.semantic_memory)
            new_facts_count = facts_after - self.facts_before_generation
            
            # Get the actual facts that were extracted
            if new_facts_count > 0:
                # Get the most recent facts
                all_facts = list(self.memory.semantic_memory.values())
                extracted_facts = all_facts[-new_facts_count:]
            
                # Complete ReAct cycle
            if self.current_cycle:
                self.current_cycle.complete(response_content, success=True)
                
                # Complete scratchpad cycle with final answer
                self.scratchpad.complete_cycle(
                    final_answer=response_content,
                    success=True
                )
        
        # Build enhanced response with telemetry
        enhanced_response = self._build_enhanced_response(
            base_response=response,
            provider_name=provider_name,
            total_time=total_generation_time,
            extracted_facts=extracted_facts
        )
        
        # Clear current cycle after building response
        if self.current_cycle:
            self.current_cycle = None
        
        return enhanced_response
    
    def execute_tool_call(self,
                         tool_call: "ToolCall",
                         tool_functions: Dict[str, Callable[..., Any]]) -> Dict[str, Any]:
        """
        Execute tool call with retry and memory tracking.
        """
        import time
        start_time = time.time()
        
        # Track in ReAct cycle
        if self.current_cycle:
            action_id = self.current_cycle.add_action(
                tool_name=tool_call.name,
                arguments=tool_call.arguments if hasattr(tool_call, 'arguments') else {},
                reasoning=f"Executing {tool_call.name} to gather information"
            )
        else:
            action_id = f"action_{len(self.current_tool_traces)}"
            
        # Add to scratchpad with complete observability
        scratchpad_action_id = self.scratchpad.add_action(
            tool_name=tool_call.name,
            tool_args=tool_call.arguments if hasattr(tool_call, 'arguments') else {},
            reasoning=f"Executing {tool_call.name} to gather information",
            metadata={"session_action_id": action_id}
        )
        
        # Execute with retry if enabled
        success = True
        error = None
        
        if self.enable_retry:
            try:
                result = self.retry_manager.retry_with_backoff(
                    super().execute_tool_call,
                    tool_call,
                    tool_functions,
                    key=f"tool_{tool_call.name}"
                )
                # Track retry attempts
                retry_attempts = getattr(self.retry_manager, '_current_attempts', {}).get(f"tool_{tool_call.name}", 0)
                self.current_retry_attempts += retry_attempts
            except Exception as e:
                success = False
                error = str(e)
                result = {"error": str(e)}
                if self.current_cycle and action_id:
                    self.current_cycle.add_observation(
                        action_id=action_id,
                        content=str(e),
                        success=False
                    )
                # Still create the trace for failed tools
        else:
            try:
                result = super().execute_tool_call(tool_call, tool_functions)
            except Exception as e:
                success = False
                error = str(e)
                result = {"error": str(e)}
        
        execution_time = time.time() - start_time
        
        # Create tool execution trace
        tool_trace = {
            "name": tool_call.name,
            "arguments": tool_call.arguments if hasattr(tool_call, 'arguments') else {},
            "result": str(result.get("output", result.get("error", result)))[:500],  # Truncate for readability
            "success": success,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "action_id": action_id,
            "reasoning": f"Executing {tool_call.name} to gather information",
            "error": error
        }
        
        # Add to current traces
        self.current_tool_traces.append(tool_trace)
        
        # Track observation in ReAct cycle
        if self.current_cycle and action_id:
            self.current_cycle.add_observation(
                action_id=action_id,
                content=result.get("output", result.get("error")),
                success=success
            )
            
        # Add observation to scratchpad (COMPLETE - no truncation)
        self.scratchpad.add_observation(
            action_id=scratchpad_action_id,
            result=result.get("output", result.get("error", result)),
            success=success,
            execution_time=execution_time,
            metadata={"session_action_id": action_id}
        )
        
        # Re-raise exception if it occurred
        if not success and error:
            raise Exception(error)
        
        return result
    
    def _build_enhanced_response(self,
                                base_response: "GenerateResponse",
                                provider_name: str,
                                total_time: float,
                                extracted_facts: List[Any]) -> "GenerateResponse":
        """Build enhanced response with readable telemetry."""
        
        # Create facts list
        facts_strings = []
        for fact in extracted_facts:
            if hasattr(fact, '__str__'):
                facts_strings.append(str(fact))
            else:
                facts_strings.append(f"{fact.subject} --[{fact.predicate}]--> {fact.object}")
        
        # Get COMPLETE scratchpad trace (NO truncation)
        complete_scratchpad = self.scratchpad.get_complete_trace()
        scratchpad_file_path = self.scratchpad.get_scratchpad_file_path()
        
        # Create enhanced response by copying base response
        enhanced_response = base_response
        
        # Add telemetry fields with complete observability
        enhanced_response.react_cycle_id = self.current_cycle.cycle_id if self.current_cycle else None
        enhanced_response.tool_calls = self.current_tool_traces  # Now readable list
        enhanced_response.tools_executed = self.current_tool_traces
        enhanced_response.facts_extracted = facts_strings
        enhanced_response.reasoning_trace = complete_scratchpad  # COMPLETE scratchpad
        enhanced_response.total_reasoning_time = total_time
        
        # Add scratchpad file reference for external access
        enhanced_response.scratchpad_file = str(scratchpad_file_path)
        enhanced_response.scratchpad_manager = self.scratchpad  # For direct access
        
        return enhanced_response
    
    def _get_response_handler(self, provider_name: str) -> StructuredResponseHandler:
        """Get or create structured response handler for provider."""
        if provider_name not in self.response_handlers:
            self.response_handlers[provider_name] = StructuredResponseHandler(provider_name)
        return self.response_handlers[provider_name]
    
    def get_memory_stats(self) -> Optional[Dict[str, Any]]:
        """Get memory system statistics."""
        if self.memory:
            return self.memory.get_statistics()
        return None
    
    def save_memory(self):
        """Save memory to disk."""
        if self.memory:
            self.memory.save_to_disk()
            logger.info("Memory saved to disk")
    
    def visualize_memory_links(self) -> Optional[str]:
        """Get memory link visualization."""
        if self.memory:
            return self.memory.visualize_links()
        return None
    
    def query_memory(self, query: str) -> Optional[Dict[str, Any]]:
        """Query memory for relevant information."""
        if self.memory:
            return self.memory.query_memory(query)
        return None


def create_enhanced_session(
    provider: Optional[Union[str, AbstractLLMInterface]] = None,
    enable_memory: bool = True,
    enable_retry: bool = True,
    persist_memory: Optional[str] = None,
    **kwargs
) -> EnhancedSession:
    """
    Create an enhanced session with SOTA features.
    
    Args:
        provider: Provider name or instance
        enable_memory: Enable hierarchical memory
        enable_retry: Enable retry strategies
        persist_memory: Path to persist memory
        **kwargs: Additional session parameters
        
    Returns:
        Enhanced session instance
    """
    persist_path = Path(persist_memory) if persist_memory else None
    
    return EnhancedSession(
        provider=provider,
        enable_memory=enable_memory,
        enable_retry=enable_retry,
        persist_memory=persist_path,
        **kwargs
    )
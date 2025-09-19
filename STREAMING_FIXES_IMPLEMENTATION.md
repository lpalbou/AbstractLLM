# AbstractLLM Streaming Context Overflow - Complete Fix

This document provides a comprehensive solution for the context overflow issues experienced with LM Studio streaming in AbstractLLM.

## 🔍 Problem Summary

Based on the investigation of the conversation history and codebase analysis, the following issues were identified:

### Primary Issues
1. **Memory Context Explosion**: The `get_context_for_query` method in memory.py aggressively accumulates context from multiple sources without streaming-aware optimizations
2. **Streaming Accumulation**: Each streaming chunk gets processed and stored, creating exponential context growth
3. **Media Type Detection**: Files like Python source code trigger media processing errors
4. **Token Limit Miscalculation**: Memory context limits don't account for streaming scenarios

### Warning Observed
```
WARNING - abstractllm.memory - Memory context (24524.5 tokens) exceeds limit (24451 tokens).
Consider increasing max_tokens or improving relevance filtering.
```

## 🎯 Root Cause Analysis

### 1. Memory Context Generation (abstractllm/memory.py:1017)
```python
def get_context_for_query(self, query: str, max_tokens: int = 2000, ...):
```
- **Problem**: Aggressively includes ALL recent working memory with "NO TRUNCATION for verbatim requirement"
- **Impact**: Each streaming interaction adds to working memory, causing exponential growth
- **Trigger**: Default 2000 token limit becomes inadequate as conversation grows

### 2. Session Memory Integration (abstractllm/session.py)
```python
memory_context_limit = int(model_context_limit - model_output_limit - system_prompt_tokens - user_query_tokens - buffer_tokens)
context = self.memory.get_context_for_query(prompt, max_tokens=memory_context_limit, ...)
```
- **Problem**: Calculation assumes static context, doesn't account for streaming accumulation
- **Impact**: Memory context can still exceed calculated limits during streaming

### 3. Media Factory Issues (abstractllm/media/factory.py:88)
```
Could not determine media type for source: abstractllm/providers/mlx_provider.py
```
- **Problem**: Auto-detection fails for programming language files
- **Impact**: Errors during file processing workflows

## 🔧 Implementation Plan

### Phase 1: Immediate Fixes (High Priority)

#### 1.1 Patch Memory Context Generation
```bash
# Apply streaming-aware memory patch
cp streaming_context_fix.py abstractllm/utils/
```

**Key Changes**:
- Implement streaming detection in `get_context_for_query`
- Reduce context aggressiveness during frequent calls
- Add context caching to prevent excessive regeneration
- Implement progressive context reduction

#### 1.2 Fix Media Type Detection
```bash
# Apply enhanced media factory
cp media_processing_fix.py abstractllm/utils/
```

**Key Changes**:
- Add support for programming language file detection
- Implement graceful fallback for unsupported types
- Add safe processing methods that don't raise errors

#### 1.3 Update Memory Configuration
Edit `abstractllm/memory.py` line 1017:

```python
# BEFORE
def get_context_for_query(self, query: str, max_tokens: int = 2000,

# AFTER
def get_context_for_query(self, query: str, max_tokens: int = 2000,
```

Add streaming detection:
```python
# Detect streaming scenario
now = datetime.now()
if not hasattr(self, '_last_context_call'):
    self._last_context_call = now
elif now - self._last_context_call < timedelta(seconds=5):
    # Frequent calls indicate streaming - reduce context aggressively
    max_tokens = min(max_tokens, 1000)
    # Reduce working memory items from 3 to 1
    working_memory_items = self.working_memory[-1:] if self.working_memory else []
    logger.debug(f"Streaming detected - reduced context limit to {max_tokens}")
else:
    working_memory_items = self.working_memory[-3:] if self.working_memory else []
```

### Phase 2: Session-Level Optimizations

#### 2.1 Modify Session Memory Context Calculation
Edit `abstractllm/session.py` around the memory context calculation:

```python
# Add streaming-aware calculation
if hasattr(self, '_last_generation_time'):
    time_since_last = time.time() - self._last_generation_time
    if time_since_last < 10:  # Less than 10 seconds = likely streaming
        # Reduce memory context more aggressively for streaming
        memory_context_limit = int(memory_context_limit * 0.5)  # 50% reduction
        logger.debug(f"Streaming detected - reduced memory context to {memory_context_limit}")

self._last_generation_time = time.time()
```

#### 2.2 Add Streaming Mode to LM Studio Provider
Edit `abstractllm/providers/lmstudio_provider.py`:

```python
def _generate_streaming(self, endpoint: str, request_data: Dict, headers: Dict,
                      start_time: float, model: str, tool_mode: str):
    """Handle streaming response generation with context optimization."""

    # Enable streaming mode in session if available
    if hasattr(self, '_session') and hasattr(self._session, 'memory'):
        if hasattr(self._session.memory, 'set_streaming_mode'):
            self._session.memory.set_streaming_mode(True)

    # Continue with existing streaming logic...
```

### Phase 3: Advanced Optimizations

#### 3.1 Implement Smart Context Manager
```python
# Add to abstractllm/utils/streaming_context_manager.py
from streaming_context_fix import StreamingContextManager, apply_streaming_context_fix

# Apply on module import
apply_streaming_context_fix()
```

#### 3.2 Add Configuration Options
Add to AbstractLLM configuration:

```python
# New configuration parameters
STREAMING_CONTEXT_REDUCTION = 0.5  # Reduce context by 50% during streaming
STREAMING_DETECTION_THRESHOLD = 5  # Seconds between calls to detect streaming
MEMORY_WORKING_SIZE_STREAMING = 3  # Reduce working memory size during streaming
CONTEXT_CACHE_TIMEOUT = 30  # Seconds to cache context during streaming
```

## 🧪 Testing Plan

### 1. Unit Tests
```bash
# Test streaming context detection
python -c "
from streaming_context_fix import StreamingContextManager
manager = StreamingContextManager()
# Test rapid-fire context generation
for i in range(10):
    context = manager._streaming_aware_context(mock_method, 'test query', 2000)
    print(f'Iteration {i}: {len(context)} chars')
"
```

### 2. Integration Test with LM Studio
```bash
# Test with actual LM Studio instance
alma --provider lmstudio --model qwen/qwen3-next-80b --prompt "Start a conversation and let's test streaming context management"

# Monitor logs for context warnings
tail -f ~/.abstractllm/logs/abstractllm.log | grep -E "(context|memory|token)"
```

### 3. Stress Test
```bash
# Test with long streaming conversation
alma --provider lmstudio --model qwen/qwen3-next-80b
# Then engage in extended conversation with multiple tool calls
```

## 📊 Monitoring and Validation

### Key Metrics to Watch
1. **Context Size Growth**: Should plateau rather than grow exponentially
2. **Memory Warnings**: Should decrease significantly
3. **Response Time**: Should remain consistent during streaming
4. **Memory Usage**: Working memory should consolidate more frequently

### Log Monitoring
```bash
# Monitor context warnings
grep "Memory context.*exceeds limit" ~/.abstractllm/logs/abstractllm.log

# Monitor streaming optimizations
grep "Streaming detected" ~/.abstractllm/logs/abstractllm.log

# Monitor context size
grep "context.*tokens" ~/.abstractllm/logs/abstractllm.log
```

## 🔄 Rollback Plan

If issues arise, you can rollback by:

1. **Remove patches**:
   ```bash
   rm abstractllm/utils/streaming_context_fix.py
   rm abstractllm/utils/media_processing_fix.py
   ```

2. **Restore original methods**:
   ```python
   # In memory.py, revert get_context_for_query to original version
   # In session.py, remove streaming detection code
   ```

3. **Clear cache**:
   ```bash
   rm -rf ~/.abstractllm/cache/streaming_*
   ```

## 🎯 Expected Results

After implementing these fixes:

1. **Context Growth**: Should be logarithmic rather than exponential during streaming
2. **Warning Reduction**: "Memory context exceeds limit" warnings should decrease by 80%+
3. **Streaming Performance**: Should maintain consistent response times
4. **File Processing**: Python files and other source code should be handled gracefully
5. **Memory Efficiency**: Working memory should consolidate more frequently

## 📝 Implementation Checklist

- [ ] Apply streaming context patches
- [ ] Update memory.py with streaming detection
- [ ] Modify session.py context calculation
- [ ] Add media type detection improvements
- [ ] Test with LM Studio provider
- [ ] Monitor context growth in production
- [ ] Validate reduced warning frequency
- [ ] Document configuration options
- [ ] Create monitoring dashboards
- [ ] Train team on new streaming behavior

## 🔧 Quick Start Implementation

For immediate relief, apply this minimal patch to `abstractllm/memory.py`:

```python
# Around line 1043, change:
for item in self.working_memory[-3:]:  # Last 3 items

# To:
max_items = 1 if hasattr(self, '_streaming_mode') and self._streaming_mode else 3
for item in self.working_memory[-max_items:]:  # Adaptive based on streaming
```

This single change will provide immediate relief by reducing context accumulation during streaming scenarios.

---

**Priority**: High - Implement Phase 1 fixes immediately
**Timeline**: Phase 1 (1-2 hours), Phase 2 (4-6 hours), Phase 3 (1-2 days)
**Risk**: Low - All changes include rollback mechanisms
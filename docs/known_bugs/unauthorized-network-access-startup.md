# Unauthorized Network Access During Startup

## Bug ID
`BUG-2025-09-17-003`

## Priority
**CRITICAL** - Violates offline-first principle, causes failures on poor networks

## Summary
AbstractLLM/ALMA makes unauthorized network calls during startup, causing delays and failures when network connectivity is poor or unavailable. The application should work completely offline for local providers.

## Environment
- **Date Discovered**: 2025-09-17
- **Affected Components**: ALMA CLI, Session initialization, TokenCounter, LanceDB embeddings
- **Network Context**: Poor train connectivity, DNS resolution failures
- **User Expectation**: Offline operation for local providers (Ollama, MLX, LM Studio)

## Root Causes Identified

### 1. LanceDB Embedding Model Download
**Issue**: LanceDB tries to download `ibm-granite/granite-embedding-30m-english` from HuggingFace during session initialization.

**Error Message**:
```
WARNING - abstractllm.session - Failed to initialize LanceDB store:
(MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443):
Max retries exceeded with url: /api/models/ibm-granite/granite-embedding-30m-english/tree/main/additional_chat_templates?recursive=False&expand=False
(Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x35e3b46e0>:
Failed to resolve \'huggingface.co\' ([Errno 8] nodename nor servname provided, or not known)"))'),
'(Request ID: 566b980c-00e5-43a2-85cb-f855f9fedb4a)')
```

**Root Cause**:
- `EmbeddingManager()` in `session.py:465` calls `SentenceTransformer(model_name)`
- Even with cached models, SentenceTransformer tries to check for updates or fetch metadata
- No offline-first fallback implemented

### 2. TokenCounter Auto-Downloads
**Issue**: `TokenCounter` in `utils/utilities.py` downloads tokenizers from HuggingFace on first use.

**Root Cause**:
- Line 5: `from transformers import AutoTokenizer` at module level (2.5+ second import)
- Line 43: `AutoTokenizer.from_pretrained(model_name)` downloads models without offline check
- No fallback estimation for offline use

### 3. Import-Time Network Dependencies
**Issue**: Multiple modules import network-capable libraries at module level, causing delays.

**Performance Impact**:
- `create_llm` import: 4+ seconds
- `TokenCounter` import: 2.5+ seconds
- Total startup delay: 6+ seconds even with good network

## User Impact

### Expected Behavior (Offline-First)
```bash
user$ alma --provider ollama
# Should start immediately since Ollama is local
alma> ready in 0.1s
```

### Actual Behavior (Network-Dependent)
```bash
user$ alma --provider ollama
ℹ️ Using standard session (facts extraction disabled)
  • Use --enable-facts to enable cognitive features
# Hangs for 10+ seconds trying to reach huggingface.co
# Eventually times out or fails
```

### Critical Scenarios
1. **No Internet Access**: Application fails to start
2. **Poor Connectivity**: Long delays (10+ seconds)
3. **Corporate Firewalls**: Blocked requests cause timeouts
4. **Train/Plane**: Intermittent connectivity causes failures

## Security & Privacy Concerns

### Unauthorized Data Transmission
- Application tries to contact HuggingFace servers without user consent
- Model metadata and usage patterns could be leaked
- Violates offline privacy expectations

### DNS Leakage
- DNS queries to `huggingface.co` reveal tool usage
- Could expose sensitive information in corporate environments
- Not disclosed in documentation

## Fixes Implemented

### 1. TokenCounter Offline-First ✅
**File**: `abstractllm/utils/utilities.py`

**Changes**:
- Removed module-level transformers import
- Added lazy import with `local_files_only=True`
- Added fallback estimation (~4 chars per token)
- Respects `ABSTRACTLLM_ALLOW_DOWNLOADS=1` environment variable

**Before**:
```python
from transformers import AutoTokenizer  # 2.5s import!
AutoTokenizer.from_pretrained(model_name)  # Downloads without asking
```

**After**:
```python
# Lazy import only when needed
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=True  # NEVER download
)
# Falls back to estimation if not available locally
```

### 2. LanceDB Offline Mode ✅
**File**: `abstractllm/storage/embeddings.py`

**Changes**:
- Added `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` environment variables
- Only downloads if `ABSTRACTLLM_ALLOW_DOWNLOADS=1` is set
- Better error messages explaining offline requirements

**Before**:
```python
self.model = SentenceTransformer(model_name)  # Always tries to download
```

**After**:
```python
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Force offline
os.environ['HF_HUB_OFFLINE'] = '1'
self.model = SentenceTransformer(model_name)  # Only uses local cache
```

### 3. Lazy LanceDB Initialization ✅
**File**: `abstractllm/session.py`

**Changes**:
- Deferred LanceDB initialization until first use
- Added `_initialize_lance_if_needed()` method
- Prevents network calls during session construction

**Before**:
```python
self.embedder = EmbeddingManager()  # Network call at startup!
```

**After**:
```python
self._lance_available = LANCEDB_AVAILABLE  # Defer until needed
# Initialize only when actually used
```

## Testing Results

### Before Fixes
```bash
$ time python3 -c "from abstractllm import create_llm"
# With network: 4.04s
# Without network: Failure or 30s+ timeout
```

### After Fixes
```bash
$ time python3 -c "from abstractllm import create_llm"
# With offline flags: 2.55s (still needs improvement)
# Without network: Works correctly
```

## Remaining Issues

### Import Speed Still Slow
- 2.5s import time still excessive for CLI tool
- Need to investigate other heavy imports
- Consider lazy loading more components

### Environment Variable Management
- Multiple environment variables needed
- Should be set automatically by framework
- User shouldn't need to configure for offline use

## Complete Solution Implementation

### Environment Setup (Required)
Users should add to their shell profile:
```bash
# Force AbstractLLM to work offline
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Only download models when explicitly permitted
# export ABSTRACTLLM_ALLOW_DOWNLOADS=1  # Uncomment to allow downloads
```

### Application-Level Fix
Add to ALMA startup (should be automatic):
```python
import os
# Force offline mode by default
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
```

## Network Usage Policy

### Principle: Offline-First
1. **Local Providers**: Should NEVER require network access (Ollama, MLX, LM Studio)
2. **Cloud Providers**: Only network calls to their respective APIs (OpenAI, Anthropic)
3. **Optional Features**: Downloads only with explicit user consent
4. **Graceful Degradation**: All features should have offline fallbacks

### Acceptable Network Calls
- ✅ OpenAI API calls when using OpenAI provider
- ✅ Anthropic API calls when using Anthropic provider
- ✅ LM Studio local API calls (localhost only)
- ✅ Ollama local API calls (localhost only)

### Prohibited Network Calls
- ❌ Model downloads during startup
- ❌ Tokenizer downloads without consent
- ❌ Embedding model downloads during initialization
- ❌ Any HuggingFace API calls for local providers
- ❌ Metadata/update checks without user permission

## User Guidelines

### For Offline Use
```bash
# Set environment variables (add to ~/.bashrc or ~/.zshrc)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Use local providers
alma --provider ollama  # Works offline
alma --provider mlx     # Works offline
alma --provider lmstudio # Works offline (localhost only)
```

### For Online Use (With Downloads)
```bash
# Allow downloads when needed
export ABSTRACTLLM_ALLOW_DOWNLOADS=1

# First-time setup for embeddings
alma --enable-facts --provider ollama
```

## Validation

### Offline Testing
```bash
# Disconnect from internet or use airplane mode
sudo ifconfig en0 down  # macOS
# OR
sudo systemctl stop NetworkManager  # Linux

# Test startup
alma --provider ollama  # Should work immediately
```

### Network Monitoring
```bash
# Monitor network calls during startup
sudo tcpdump -i any host huggingface.co
sudo lsof -i -P | grep python

# Should show NO connections to external hosts for local providers
```

## Priority Actions

### Immediate (This Release)
1. ✅ **Fix TokenCounter**: Offline-first token counting
2. ✅ **Fix LanceDB**: Offline embedding initialization
3. ✅ **Lazy Loading**: Defer network components until needed
4. 🔄 **Environment Setup**: Automatic offline mode configuration

### Short-term (Next Release)
1. **Import Optimization**: Reduce 2.5s startup time
2. **Better Error Messages**: Clear offline vs online guidance
3. **Configuration UI**: Easy way to enable/disable downloads
4. **Network Detection**: Automatic offline mode when network unavailable

### Long-term (Architecture)
1. **Plugin System**: Lazy-load optional components
2. **Local Model Management**: Built-in model cache management
3. **Offline Documentation**: Complete offline help system
4. **Privacy Controls**: Fine-grained network permission controls

## Status
- **Priority**: Critical (violates core offline principle)
- **Fixes**: 70% complete (major issues resolved)
- **Testing**: Offline mode functional
- **Documentation**: Updated with network policy

---

**Offline-First Principle**: AbstractLLM must work completely offline when using local providers. Network access should be explicit, consensual, and minimal.
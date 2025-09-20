# Task 06: Migration and Deployment (Priority 1)

**Duration**: 3 hours
**Risk**: High
**Dependencies**: Tasks 01-05 completed

## Objectives
- Create compatibility layer for existing code
- Develop migration tools
- Plan phased rollout
- Document breaking changes
- Create deployment scripts

## Steps

### 1. Compatibility Layer Implementation (60 min)

Create `abstractllm/compat.py`:
```python
"""
Compatibility layer for smooth migration.
Maps old API to new three-package architecture.
"""

import warnings
from typing import Optional, Dict, Any, List, Union
from pathlib import Path


class CompatibilityLayer:
    """
    Provides backward compatibility during migration.
    """

    @staticmethod
    def setup():
        """Setup compatibility mappings"""

        # Try new packages first, fall back to monolithic
        try:
            # Import new packages
            from abstractagent import Agent
            from abstractmemory import TemporalMemory
            from abstractllm import BasicSession

            # Create compatibility mappings
            _create_session_compat(Agent, TemporalMemory, BasicSession)
            _create_import_hooks()

            print("✅ Running with new three-package architecture")
            return "new"

        except ImportError:
            # Fall back to monolithic
            warnings.warn(
                "New packages not found, using monolithic AbstractLLM. "
                "Please migrate to the three-package architecture.",
                DeprecationWarning,
                stacklevel=2
            )
            return "legacy"


def _create_session_compat(Agent, TemporalMemory, BasicSession):
    """Create Session compatibility wrapper"""

    class SessionCompat:
        """
        Compatibility wrapper that mimics old Session API
        using new three-package architecture.
        """

        def __init__(self, **kwargs):
            # Map old parameters to new structure
            llm_config = {
                'provider': kwargs.pop('provider', 'ollama'),
                'model': kwargs.pop('model', None)
            }

            memory_config = None
            if kwargs.pop('enable_memory', False):
                memory_config = {
                    'temporal': True,
                    'persist_path': kwargs.pop('persist_memory', None)
                }

            # Create agent (new architecture)
            from abstractagent import Agent
            self._agent = Agent(
                llm_config=llm_config,
                memory_config=memory_config,
                tools=kwargs.pop('tools', None),
                enable_reasoning=kwargs.pop('enable_react', True),
                enable_retry=kwargs.pop('enable_retry', True)
            )

            # Store extra kwargs for compatibility
            self._compat_kwargs = kwargs

            # Expose commonly used attributes
            self.messages = self._agent.session.messages
            self.id = self._agent.session.id

        # Map old methods to new implementation
        def add_message(self, role, content, **kwargs):
            return self._agent.session.add_message(role, content)

        def generate(self, prompt=None, **kwargs):
            # Map to agent.chat
            use_tools = kwargs.pop('use_tools', False)
            use_reasoning = kwargs.pop('create_react_cycle', False)

            return self._agent.chat(
                prompt=prompt,
                use_reasoning=use_reasoning,
                use_tools=use_tools
            )

        def generate_with_tools(self, **kwargs):
            kwargs['use_tools'] = True
            return self.generate(**kwargs)

        def get_history(self, **kwargs):
            return self._agent.session.get_history(**kwargs)

        def save(self, path):
            self._agent.save_state(path)

        @classmethod
        def load(cls, path, **kwargs):
            agent = Agent(llm_config={'provider': 'ollama'})
            agent.load_state(path)

            # Wrap in compat
            compat = cls.__new__(cls)
            compat._agent = agent
            return compat

    # Replace Session globally
    import sys
    sys.modules['abstractllm'].Session = SessionCompat


def _create_import_hooks():
    """Create import hooks for moved modules"""

    import sys
    from importlib import import_module

    class CompatImporter:
        """Import hook for backward compatibility"""

        # Mapping of old imports to new locations
        IMPORT_MAP = {
            'abstractllm.memory': 'abstractmemory',
            'abstractllm.memory.hierarchical': 'abstractmemory.core',
            'abstractllm.cognitive': 'abstractmemory.cognitive',
            'abstractllm.session': 'abstractagent.agent',
            'abstractllm.react': 'abstractagent.reasoning.react',
        }

        def find_module(self, fullname, path=None):
            if fullname in self.IMPORT_MAP:
                return self
            return None

        def load_module(self, fullname):
            # Check if already loaded
            if fullname in sys.modules:
                return sys.modules[fullname]

            # Map to new location
            new_name = self.IMPORT_MAP[fullname]
            warnings.warn(
                f"Import '{fullname}' is deprecated. "
                f"Use '{new_name}' instead.",
                DeprecationWarning,
                stacklevel=2
            )

            # Import new module
            module = import_module(new_name)
            sys.modules[fullname] = module
            return module

    # Install hook
    sys.meta_path.insert(0, CompatImporter())
```

### 2. Migration Tools (45 min)

Create `tools/migrate_to_three_packages.py`:
```python
#!/usr/bin/env python3
"""
Migration tool to help users transition to three-package architecture.
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Tuple


class CodeMigrator(ast.NodeTransformer):
    """
    AST transformer to migrate imports and usage patterns.
    """

    def __init__(self):
        self.changes = []

    def visit_Import(self, node):
        """Transform import statements"""

        for alias in node.names:
            # Check for old imports
            if alias.name == 'abstractllm':
                # Might need all three packages
                self.changes.append(
                    f"Import '{alias.name}' may need to be split"
                )

        return node

    def visit_ImportFrom(self, node):
        """Transform from imports"""

        if node.module == 'abstractllm':
            # Map common imports
            for alias in node.names:
                if alias.name == 'Session':
                    self.changes.append(
                        "Replace 'from abstractllm import Session' with "
                        "'from abstractagent import Agent'"
                    )
                elif alias.name == 'HierarchicalMemory':
                    self.changes.append(
                        "Replace 'from abstractllm import HierarchicalMemory' with "
                        "'from abstractmemory import TemporalMemory'"
                    )

        elif node.module and node.module.startswith('abstractllm.memory'):
            self.changes.append(
                f"Move import from '{node.module}' to 'abstractmemory'"
            )

        elif node.module and node.module.startswith('abstractllm.cognitive'):
            self.changes.append(
                f"Move import from '{node.module}' to 'abstractmemory.cognitive'"
            )

        return node


def analyze_file(filepath: Path) -> List[str]:
    """Analyze a Python file for migration needs"""

    with open(filepath, 'r') as f:
        tree = ast.parse(f.read())

    migrator = CodeMigrator()
    migrator.visit(tree)

    return migrator.changes


def scan_project(root_dir: Path) -> Dict[Path, List[str]]:
    """Scan entire project for migration needs"""

    results = {}

    for file_path in root_dir.rglob('*.py'):
        # Skip migrations and tests
        if 'migration' in str(file_path) or 'test' in str(file_path):
            continue

        changes = analyze_file(file_path)
        if changes:
            results[file_path] = changes

    return results


def generate_migration_report(results: Dict[Path, List[str]]) -> str:
    """Generate migration report"""

    report = ["# Migration Report", ""]

    if not results:
        report.append("✅ No migration needed - code is already compatible!")
        return "\n".join(report)

    report.append(f"Found {len(results)} files needing migration:")
    report.append("")

    for file_path, changes in results.items():
        report.append(f"## {file_path}")
        for change in changes:
            report.append(f"- {change}")
        report.append("")

    report.append("## Migration Steps")
    report.append("1. Install new packages:")
    report.append("   pip install abstractllm-core abstractmemory abstractagent")
    report.append("2. Update imports as suggested above")
    report.append("3. Test with compatibility layer:")
    report.append("   python -c 'from abstractllm.compat import CompatibilityLayer; CompatibilityLayer.setup()'")
    report.append("4. Run tests to verify")

    return "\n".join(report)


def auto_migrate_file(filepath: Path, backup: bool = True):
    """Automatically migrate a file"""

    if backup:
        backup_path = filepath.with_suffix('.py.backup')
        filepath.rename(backup_path)
        filepath = backup_path.with_suffix('')

    with open(backup_path if backup else filepath, 'r') as f:
        content = f.read()

    # Apply migrations
    migrations = [
        # Old -> New import mappings
        ('from abstractllm import Session', 'from abstractagent import Agent'),
        ('from abstractllm.session import Session', 'from abstractagent import Agent'),
        ('from abstractllm import HierarchicalMemory', 'from abstractmemory import TemporalMemory'),
        ('from abstractllm.memory', 'from abstractmemory'),
        ('from abstractllm.cognitive', 'from abstractmemory.cognitive'),

        # Class name changes
        ('Session(', 'Agent('),
        ('HierarchicalMemory(', 'TemporalMemory('),
    ]

    for old, new in migrations:
        content = content.replace(old, new)

    # Write migrated content
    with open(filepath, 'w') as f:
        f.write(content)

    print(f"✅ Migrated {filepath}")


def main():
    """Main migration tool"""

    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate to three-package AbstractLLM architecture"
    )

    parser.add_argument(
        'path',
        type=Path,
        help='Path to project or file to migrate'
    )

    parser.add_argument(
        '--auto',
        action='store_true',
        help='Automatically apply migrations'
    )

    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backups (use with caution)'
    )

    args = parser.parse_args()

    if args.path.is_file():
        # Single file
        if args.auto:
            auto_migrate_file(args.path, backup=not args.no_backup)
        else:
            changes = analyze_file(args.path)
            for change in changes:
                print(f"- {change}")

    else:
        # Project directory
        results = scan_project(args.path)

        if args.auto:
            for filepath in results.keys():
                auto_migrate_file(filepath, backup=not args.no_backup)
        else:
            report = generate_migration_report(results)
            print(report)


if __name__ == '__main__':
    main()
```

### 3. Phased Rollout Plan (30 min)

Create `docs/rollout_plan.md`:
```markdown
# Phased Rollout Plan

## Phase 1: Internal Testing (Week 1)
- Deploy to development environment
- Run compatibility layer tests
- Test with sample applications
- Monitor performance metrics

## Phase 2: Beta Release (Week 2)
- Release as beta packages:
  - abstractllm-core==2.0.0b1
  - abstractmemory==1.0.0b1
  - abstractagent==1.0.0b1
- Maintain monolithic package with deprecation warnings
- Gather feedback from early adopters

## Phase 3: Documentation & Examples (Week 3)
- Update all documentation
- Create migration guides
- Provide code examples
- Record migration video tutorials

## Phase 4: Production Release (Week 4)
- Release stable versions
- Announce deprecation timeline for monolithic package
- 6-month support window for legacy version
```

### 4. Deployment Scripts (45 min)

Create `scripts/deploy_packages.sh`:
```bash
#!/bin/bash
# Deploy three packages to PyPI

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}AbstractLLM Three-Package Deployment${NC}"
echo "======================================"

# Check we're in the right place
if [ ! -f "abstractllm/setup.py" ]; then
    echo -e "${RED}Error: Not in AbstractLLM root directory${NC}"
    exit 1
fi

# Function to deploy a package
deploy_package() {
    local package_name=$1
    local package_dir=$2

    echo -e "\n${BLUE}Deploying $package_name...${NC}"

    cd $package_dir

    # Clean previous builds
    rm -rf dist/ build/ *.egg-info

    # Build package
    python setup.py sdist bdist_wheel

    # Run tests
    echo "Running tests..."
    pytest tests/ || {
        echo -e "${RED}Tests failed for $package_name${NC}"
        exit 1
    }

    # Check package
    twine check dist/*

    # Upload to TestPyPI first
    if [ "$DEPLOY_ENV" = "test" ]; then
        echo "Uploading to TestPyPI..."
        twine upload --repository testpypi dist/*
    elif [ "$DEPLOY_ENV" = "prod" ]; then
        echo "Uploading to PyPI..."
        twine upload dist/*
    else
        echo "Skipping upload (dry run)"
    fi

    cd -

    echo -e "${GREEN}✅ $package_name deployed successfully${NC}"
}

# Parse arguments
DEPLOY_ENV=${1:-dry}

if [ "$DEPLOY_ENV" = "prod" ]; then
    echo -e "${RED}WARNING: Deploying to production PyPI${NC}"
    read -p "Are you sure? (yes/no) " -r
    if [[ ! $REPLY =~ ^yes$ ]]; then
        echo "Deployment cancelled"
        exit 1
    fi
fi

# Deploy each package
deploy_package "AbstractLLM-Core" "abstractllm"
deploy_package "AbstractMemory" "../abstractmemory"
deploy_package "AbstractAgent" "../abstractagent"

echo -e "\n${GREEN}All packages deployed successfully!${NC}"

# Create compatibility package
echo -e "\n${BLUE}Creating compatibility meta-package...${NC}"

cat > abstractllm-compat/setup.py << 'EOF'
from setuptools import setup

setup(
    name="abstractllm",
    version="2.0.0",
    description="AbstractLLM meta-package for compatibility",
    install_requires=[
        "abstractllm-core>=2.0.0",
        "abstractmemory>=1.0.0",
        "abstractagent>=1.0.0",
    ],
    python_requires=">=3.8",
)
EOF

cd abstractllm-compat
python setup.py sdist bdist_wheel
cd -

echo -e "${GREEN}✅ Compatibility package created${NC}"
```

### 5. Breaking Changes Documentation (30 min)

Create `docs/BREAKING_CHANGES.md`:
```markdown
# Breaking Changes - AbstractLLM 2.0

## Overview
AbstractLLM 2.0 splits the monolithic package into three focused packages:
- **abstractllm-core**: Core LLM abstractions
- **abstractmemory**: Advanced memory system
- **abstractagent**: Agent orchestration

## Breaking Changes

### 1. Package Structure
```python
# OLD
from abstractllm import Session, HierarchicalMemory

# NEW
from abstractagent import Agent
from abstractmemory import TemporalMemory
```

### 2. Session → Agent
The monolithic Session class is replaced by Agent:

```python
# OLD
session = Session(
    provider='openai',
    enable_memory=True,
    enable_retry=True,
    tools=[...]
)
response = session.generate(prompt)

# NEW
agent = Agent(
    llm_config={'provider': 'openai'},
    memory_config={'temporal': True},
    tools=[...]
)
response = agent.chat(prompt)
```

### 3. Memory System
HierarchicalMemory becomes TemporalMemory with improved API:

```python
# OLD
memory = HierarchicalMemory()
memory.add_to_working_memory(item)

# NEW
memory = TemporalMemory()
memory.add_fact(subject, predicate, object, event_time, ingestion_time)
```

### 4. Tools Location
Basic tools remain in core, advanced tools move to agent:

```python
# Basic tools (file, exec) - still in core
from abstractllm.tools import read_file, write_file

# Advanced tools - now in agent
from abstractagent.tools import WebSearchTool, CodeIntelligenceTool
```

### 5. Imports That Moved
| Old Import | New Import |
|------------|------------|
| abstractllm.Session | abstractagent.Agent |
| abstractllm.memory.* | abstractmemory.* |
| abstractllm.cognitive.* | abstractmemory.cognitive.* |
| abstractllm.react | abstractagent.reasoning.react |
| abstractllm.cli | abstractagent.cli |

## Migration Path

### Option 1: Compatibility Layer (Recommended)
```python
# Add to your main entry point
from abstractllm.compat import CompatibilityLayer
CompatibilityLayer.setup()

# Your existing code continues to work
from abstractllm import Session  # Works via compat layer
```

### Option 2: Gradual Migration
1. Install new packages alongside old
2. Migrate module by module
3. Use migration tool to identify changes needed
4. Remove old package when complete

### Option 3: Full Migration
1. Run migration tool: `python migrate_to_three_packages.py your_project/`
2. Review and apply suggested changes
3. Update requirements.txt
4. Test thoroughly

## Support Timeline
- **v1.x (monolithic)**: Supported until January 2025
- **v2.x (three packages)**: Long-term support
- **Compatibility layer**: Available indefinitely

## Getting Help
- Migration guide: docs/migration_guide.md
- Discord: https://discord.gg/abstractllm
- GitHub Issues: https://github.com/abstractllm/abstractllm/issues
```

### 6. Rollback Plan (15 min)

Create `scripts/rollback.sh`:
```bash
#!/bin/bash
# Emergency rollback script

echo "🚨 Emergency Rollback Procedure"
echo "==============================="

# Check for backup
if [ ! -d "abstractllm_backup_*" ]; then
    echo "❌ No backup found. Cannot rollback."
    exit 1
fi

# Get latest backup
BACKUP=$(ls -td abstractllm_backup_* | head -1)
echo "Found backup: $BACKUP"

# Confirm rollback
read -p "Rollback to $BACKUP? This will lose all changes! (yes/no) " -r
if [[ ! $REPLY =~ ^yes$ ]]; then
    echo "Rollback cancelled"
    exit 1
fi

# Perform rollback
echo "Rolling back..."

# Save current state just in case
mv abstractllm abstractllm_failed_$(date +%Y%m%d_%H%M%S)

# Restore backup
cp -r $BACKUP abstractllm

# Reinstall old version
cd abstractllm
pip install -e .

echo "✅ Rollback complete"
echo "Old monolithic version restored"
```

## Validation

### Test Compatibility Layer
```bash
# Test that old code still works
python -c "
from abstractllm.compat import CompatibilityLayer
mode = CompatibilityLayer.setup()
print(f'Running in {mode} mode')

# Old API should work
from abstractllm import Session
session = Session(provider='ollama')
print('✅ Compatibility layer works')
"
```

### Test Migration Tool
```bash
# Test on sample project
cd /tmp
git clone https://github.com/example/uses-abstractllm sample_project
cd sample_project

# Analyze for migration needs
python /Users/albou/projects/abstractllm/tools/migrate_to_three_packages.py .

# Test auto-migration on a copy
cp -r . ../sample_project_migrated
cd ../sample_project_migrated
python /Users/albou/projects/abstractllm/tools/migrate_to_three_packages.py . --auto
```

### Test Deployment
```bash
# Dry run deployment
cd /Users/albou/projects/abstractllm
./scripts/deploy_packages.sh dry

# Test deployment to TestPyPI
./scripts/deploy_packages.sh test

# Verify packages on TestPyPI
pip install -i https://test.pypi.org/simple/ abstractllm-core
pip install -i https://test.pypi.org/simple/ abstractmemory
pip install -i https://test.pypi.org/simple/ abstractagent
```

## Success Criteria

- [ ] Compatibility layer allows old code to run unchanged
- [ ] Migration tool correctly identifies needed changes
- [ ] Auto-migration successfully updates imports
- [ ] Deployment scripts work for all three packages
- [ ] Documentation clearly explains breaking changes
- [ ] Rollback procedure tested and working

## Communication Plan

### Pre-Release (2 weeks before)
- Blog post announcing the change
- Email to major users
- Discord/Slack announcement

### Release Day
- Detailed migration guide published
- Video tutorial released
- Office hours for migration help

### Post-Release (1 month after)
- Gather feedback
- Address common issues
- Update migration tools

## Next Steps

1. **Execute deployment to TestPyPI**
2. **Run beta with selected users**
3. **Incorporate feedback**
4. **Production release**
5. **Monitor adoption metrics**

## End of Migration Tasks

The refactoring plan is now complete. Proceed with implementation!
# Task 01: Backup and Setup (Priority 1)

**Duration**: 2 hours
**Risk**: Low
**Dependencies**: None

## Objectives
- Create complete backup of current codebase
- Set up refactoring branch
- Create tool scripts for automation

## Steps

### 1. Create Complete Backup (30 min)

```bash
# Navigate to projects directory
cd /Users/albou/projects

# Create timestamped backup
cp -r abstractllm abstractllm_backup_$(date +%Y%m%d_%H%M%S)

# Verify backup
ls -la abstractllm_backup_*
du -sh abstractllm_backup_*/
```

### 2. Git Branch Setup (15 min)

```bash
cd /Users/albou/projects/abstractllm

# Create refactoring branch
git checkout -b refactoring_three_packages

# Tag current state
git tag -a pre_refactoring_v1 -m "State before three-package refactoring"

# Push tag
git push origin pre_refactoring_v1
```

### 3. Create Directory Structure (15 min)

```bash
# Create internal staging directories
mkdir -p abstractllm/_refactoring/{memory,agent,session}
mkdir -p abstractllm/_refactoring/tools
mkdir -p docs/final-refactoring/logs

# Create new package directories (for later)
mkdir -p ../abstractllm-core
mkdir -p ../abstractmemory
mkdir -p ../abstractagent
```

### 4. Create Analysis Tools (30 min)

Create `tools/analyze_session.py`:
```python
#!/usr/bin/env python3
"""Analyze session.py to categorize methods"""

import ast
import json

class SessionAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.methods = {
            'core': [],      # Basic conversation
            'memory': [],    # Memory-related
            'tools': [],     # Tool execution
            'react': [],     # ReAct reasoning
            'retry': [],     # Retry strategies
            'structured': [], # Structured responses
            'other': []      # Everything else
        }

    def visit_FunctionDef(self, node):
        name = node.name

        # Categorize based on name and docstring
        if any(x in name.lower() for x in ['message', 'history', 'conversation', 'add_message', 'get_messages']):
            self.methods['core'].append(name)
        elif 'memory' in name.lower() or 'Memory' in name:
            self.methods['memory'].append(name)
        elif 'tool' in name.lower():
            self.methods['tools'].append(name)
        elif 'react' in name.lower() or 'cycle' in name.lower():
            self.methods['react'].append(name)
        elif 'retry' in name.lower():
            self.methods['retry'].append(name)
        elif 'structured' in name.lower():
            self.methods['structured'].append(name)
        else:
            self.methods['other'].append(name)

    def report(self):
        total = sum(len(v) for v in self.methods.values())
        print(f"Total methods: {total}")
        for category, methods in self.methods.items():
            print(f"\n{category.upper()} ({len(methods)} methods):")
            for method in sorted(methods)[:10]:  # Show first 10
                print(f"  - {method}")
            if len(methods) > 10:
                print(f"  ... and {len(methods) - 10} more")

if __name__ == "__main__":
    with open('abstractllm/session.py', 'r') as f:
        tree = ast.parse(f.read())

    analyzer = SessionAnalyzer()
    analyzer.visit(tree)
    analyzer.report()

    # Save to JSON for later use
    with open('docs/final-refactoring/logs/session_analysis.json', 'w') as f:
        json.dump(analyzer.methods, f, indent=2)
```

### 5. Create Dependency Checker (30 min)

Create `tools/check_dependencies.py`:
```python
#!/usr/bin/env python3
"""Check for circular dependencies"""

import ast
import os
from collections import defaultdict

def get_imports(file_path):
    """Extract imports from a Python file"""
    imports = []
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
    except:
        pass
    return imports

def check_circular_deps(directory):
    """Check for circular dependencies in directory"""
    deps = defaultdict(list)

    for root, dirs, files in os.walk(directory):
        # Skip hidden and cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                module_name = file_path.replace('/', '.').replace('.py', '')
                imports = get_imports(file_path)
                deps[module_name] = imports

    # Check for circular dependencies
    circular = []
    for module, imports in deps.items():
        for imp in imports:
            if imp in deps and module in deps[imp]:
                circular.append((module, imp))

    return circular

if __name__ == "__main__":
    circular = check_circular_deps('abstractllm')
    if circular:
        print("❌ Circular dependencies found:")
        for a, b in set(circular):
            print(f"  {a} <-> {b}")
    else:
        print("✅ No circular dependencies found")
```

### 6. Create Migration Tracker (30 min)

Create `tools/migration_tracker.py`:
```python
#!/usr/bin/env python3
"""Track migration progress"""

import json
import datetime
from pathlib import Path

class MigrationTracker:
    def __init__(self):
        self.log_file = Path('docs/final-refactoring/logs/migration_log.json')
        self.load()

    def load(self):
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.log = json.load(f)
        else:
            self.log = {
                'started': str(datetime.datetime.now()),
                'tasks': {},
                'files_moved': [],
                'issues': []
            }

    def save(self):
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump(self.log, f, indent=2)

    def start_task(self, task_name):
        self.log['tasks'][task_name] = {
            'started': str(datetime.datetime.now()),
            'status': 'in_progress'
        }
        self.save()

    def complete_task(self, task_name):
        if task_name in self.log['tasks']:
            self.log['tasks'][task_name]['completed'] = str(datetime.datetime.now())
            self.log['tasks'][task_name]['status'] = 'completed'
        self.save()

    def log_file_move(self, source, destination):
        self.log['files_moved'].append({
            'source': source,
            'destination': destination,
            'timestamp': str(datetime.datetime.now())
        })
        self.save()

    def log_issue(self, issue):
        self.log['issues'].append({
            'issue': issue,
            'timestamp': str(datetime.datetime.now())
        })
        self.save()

if __name__ == "__main__":
    tracker = MigrationTracker()
    tracker.start_task("backup_and_setup")
    print("Migration tracker initialized")
```

## Validation

### Run all setup scripts
```bash
# Make scripts executable
chmod +x tools/*.py

# Run analysis
python tools/analyze_session.py

# Check dependencies
python tools/check_dependencies.py

# Initialize tracker
python tools/migration_tracker.py
```

### Verify backup
```bash
# Check backup exists and is complete
diff -r abstractllm abstractllm_backup_* | head -20
```

## Success Criteria

- [ ] Complete backup created with timestamp
- [ ] Git branch created and tagged
- [ ] Directory structure created
- [ ] Analysis tools working
- [ ] No errors in dependency check
- [ ] Migration tracker initialized

## Output

After completion, you should have:
1. `abstractllm_backup_[timestamp]/` - Complete backup
2. `abstractllm/_refactoring/` - Staging directories
3. `tools/` - Analysis and migration scripts
4. `docs/final-refactoring/logs/` - Analysis results

## Next Task

Proceed to Task 02: Split Session Core
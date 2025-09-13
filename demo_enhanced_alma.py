#!/usr/bin/env python3
"""
Demo of the Enhanced ALMA with Slash Commands and Full Memory Persistence.

This demonstrates all the new features:
- Slash command system with / prefix
- Complete session state save/load
- Memory export/import capabilities  
- Beautiful CLI display
- Interactive command tracking
"""

import os
from abstractllm import create_session
from abstractllm.utils.display import Colors, Symbols, display_success, display_info

def demo_enhanced_alma():
    """Demonstrate the enhanced ALMA system."""
    
    print(f"{Colors.BRIGHT_CYAN}{Symbols.ROCKET} Enhanced ALMA Demo{Colors.RESET}")
    print(f"{Colors.CYAN}{'═' * 60}{Colors.RESET}")
    
    print(f"\n{Colors.BRIGHT_YELLOW}{Symbols.STAR} Available Slash Commands:{Colors.RESET}")
    print(f"{Colors.CYAN}{'─' * 40}{Colors.RESET}")
    
    commands = [
        ("Memory Management", [
            "/memory, /mem - Show memory system insights",
            "/save <file> - Save complete session state", 
            "/load <file> - Load complete session state",
            "/export <file> - Export memory to JSON",
            "/facts [query] - Show extracted facts",
            "/links - Visualize memory links",
            "/scratchpad - Show reasoning traces"
        ]),
        ("Session Control", [
            "/history - Show command history",
            "/clear - Clear conversation history", 
            "/reset - Reset entire session",
            "/status - Show session status",
            "/config - Show current configuration"
        ]),
        ("Navigation", [
            "/help, /h - Show help message",
            "/exit, /quit, /q - Exit interactive mode"
        ])
    ]
    
    for category, cmd_list in commands:
        print(f"\n{Colors.BRIGHT_GREEN}  {category}:{Colors.RESET}")
        for cmd in cmd_list:
            print(f"    {Colors.BRIGHT_BLUE}{cmd}{Colors.RESET}")
    
    print(f"\n{Colors.BRIGHT_YELLOW}{Symbols.STAR} Key Features:{Colors.RESET}")
    print(f"{Colors.CYAN}{'─' * 40}{Colors.RESET}")
    
    features = [
        "✅ Complete session state persistence (messages, memory, commands)",
        "✅ Beautiful CLI with colors, symbols, and metrics dashboard", 
        "✅ Real-time tool execution visualization",
        "✅ Interactive memory analysis with facts() and scratchpad()",
        "✅ Cross-session memory export/import in JSON format",
        "✅ Command history tracking and replay",
        "✅ SOTA parameter configuration (seed, top_p, etc.)",
        "✅ Provider-agnostic memory management"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print(f"\n{Colors.BRIGHT_YELLOW}{Symbols.STAR} Usage Examples:{Colors.RESET}")
    print(f"{Colors.CYAN}{'─' * 40}{Colors.RESET}")
    
    examples = [
        "# Start enhanced interactive mode",
        "python alma_simple.py",
        "",
        "# In interactive mode:",
        "alma> /help                    # Show all commands",
        "alma> What is machine learning?  # Regular query",
        "alma> /memory                  # Check memory status", 
        "alma> /save my_session.pkl     # Save everything",
        "alma> /load my_session.pkl     # Restore everything",
        "alma> /export memory.json      # Export memory to JSON",
        "alma> /facts machine learning  # Search extracted facts",
        "alma> /history                 # View command history",
        "alma> /exit                    # Clean exit"
    ]
    
    for example in examples:
        if example.startswith("#"):
            print(f"  {Colors.BRIGHT_GREEN}{example}{Colors.RESET}")
        elif example.startswith("alma>"):
            print(f"  {Colors.BRIGHT_BLUE}{example}{Colors.RESET}")
        else:
            print(f"  {Colors.DIM}{example}{Colors.RESET}")
    
    print(f"\n{Colors.BRIGHT_YELLOW}{Symbols.STAR} Memory Persistence:{Colors.RESET}")
    print(f"{Colors.CYAN}{'─' * 40}{Colors.RESET}")
    
    memory_features = [
        "📚 Working Memory - Recent context and interactions", 
        "🧠 Episodic Memory - Consolidated experiences and events",
        "🔗 Knowledge Graph - Extracted facts with confidence scores",
        "🤔 ReAct Cycles - Complete reasoning traces and scratchpads",
        "💬 Conversation History - Full message thread with metadata",
        "⚙️ Command History - All slash commands with timestamps",
        "🔧 Provider Config - Model settings and parameters"
    ]
    
    for feature in memory_features:
        print(f"  {feature}")
    
    print(f"\n{Colors.BRIGHT_GREEN}{Symbols.CHECKMARK} Ready to use Enhanced ALMA!{Colors.RESET}")
    print(f"{Colors.CYAN}{'─' * 40}{Colors.RESET}")
    print(f"{Colors.DIM}Run: {Colors.BRIGHT_BLUE}python alma_simple.py{Colors.DIM} to start{Colors.RESET}")
    
    print(f"\n{Colors.CYAN}{'═' * 60}{Colors.RESET}")

if __name__ == "__main__":
    demo_enhanced_alma()
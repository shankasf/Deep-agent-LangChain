#!/usr/bin/env python3
"""
Deep Agents from Scratch - Main Runner

This script runs all the converted notebook examples in sequence, demonstrating
the progressive learning path from basic ReAct agents to complete research agents.

Usage:
    python run_all_examples.py [--lesson N] [--skip-setup]
    
Options:
    --lesson N: Run only lesson N (0-4)
    --skip-setup: Skip environment setup check
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def check_environment():
    """Check if the environment is properly set up."""
    console.print("\n[bold]Checking environment setup...[/bold]")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        console.print("[yellow]WARNING: .env file not found. Please create one based on env_example.txt[/yellow]")
        console.print("Required environment variables:")
        console.print("- ANTHROPIC_API_KEY")
        console.print("- OPENAI_API_KEY")
        console.print("- TAVILY_API_KEY (optional, for real web search)")
        return False
    
    # Check if required packages are installed
    try:
        import langchain
        import langgraph
        import rich
        console.print("Required packages are installed")
        return True
    except ImportError as e:
        console.print(f"[red]ERROR: Missing required package: {e}[/red]")
        console.print("Please run: pip install -r requirements.txt")
        return False


def run_lesson(lesson_num: int):
    """Run a specific lesson."""
    lesson_files = {
        0: "0_create_agent.py",
        1: "1_todo.py", 
        2: "2_files.py",
        3: "3_subagents.py",
        4: "4_full_agent.py"
    }
    
    if lesson_num not in lesson_files:
        console.print(f"[red]ERROR: Invalid lesson number: {lesson_num}. Must be 0-4.[/red]")
        return False
    
    lesson_file = lesson_files[lesson_num]
    
    if not Path(lesson_file).exists():
        console.print(f"[red]ERROR: Lesson file not found: {lesson_file}[/red]")
        return False
    
    console.print(f"\n[bold]Running Lesson {lesson_num}: {lesson_file}[/bold]")
    
    try:
        # Run the lesson script
        result = subprocess.run([sys.executable, lesson_file], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            console.print(f"[green]Lesson {lesson_num} completed successfully![/green]")
            if result.stdout:
                console.print("\n[bold]Output:[/bold]")
                console.print(result.stdout)
            return True
        else:
            console.print(f"[red]ERROR: Lesson {lesson_num} failed with return code {result.returncode}[/red]")
            if result.stderr:
                console.print("\n[bold]Error output:[/bold]")
                console.print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        console.print(f"[red]ERROR: Lesson {lesson_num} timed out after 5 minutes[/red]")
        return False
    except Exception as e:
        console.print(f"[red]ERROR: Error running lesson {lesson_num}: {e}[/red]")
        return False


def main():
    """Main function to run all examples."""
    parser = argparse.ArgumentParser(description="Run Deep Agents from Scratch examples")
    parser.add_argument("--lesson", type=int, choices=range(5), 
                       help="Run only a specific lesson (0-4)")
    parser.add_argument("--skip-setup", action="store_true", 
                       help="Skip environment setup check")
    
    args = parser.parse_args()
    
    # Display welcome message
    console.print(Panel(
        "Deep Agents from Scratch - Python Scripts\n\n"
        "This runner executes all converted notebook examples,\n"
        "demonstrating progressive agent development techniques.",
        title="Deep Agents Runner",
        border_style="green"
    ))
    
    # Check environment setup
    if not args.skip_setup:
        if not check_environment():
            console.print("\n[yellow]WARNING: Environment setup incomplete. Use --skip-setup to run anyway.[/yellow]")
            return 1
    
    # Run specific lesson or all lessons
    if args.lesson is not None:
        # Run single lesson
        success = run_lesson(args.lesson)
        return 0 if success else 1
    else:
        # Run all lessons
        console.print("\n[bold]Running all lessons in sequence...[/bold]")
        
        success_count = 0
        total_lessons = 5
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            for lesson_num in range(total_lessons):
                task = progress.add_task(f"Running Lesson {lesson_num}...", total=None)
                
                if run_lesson(lesson_num):
                    success_count += 1
                    progress.update(task, description=f"SUCCESS: Lesson {lesson_num} completed")
                else:
                    progress.update(task, description=f"FAILED: Lesson {lesson_num} failed")
                
                # Add small delay between lessons
                import time
                time.sleep(1)
        
        # Summary
        console.print(f"\n[bold]Summary: {success_count}/{total_lessons} lessons completed successfully[/bold]")
        
        if success_count == total_lessons:
            console.print(Panel(
                "All lessons completed successfully!\n\n"
                "You have successfully run through the complete Deep Agents from Scratch course:\n"
                "• Lesson 0: Basic ReAct Agent\n"
                "• Lesson 1: Task Planning with TODO Lists\n"
                "• Lesson 2: Context Offloading with Virtual File System\n"
                "• Lesson 3: Context Isolation with Sub-agents\n"
                "• Lesson 4: Complete Research Agent",
                title="Course Complete",
                border_style="green"
            ))
            return 0
        else:
            console.print(Panel(
                f"WARNING: {total_lessons - success_count} lesson(s) failed. Check the output above for details.",
                title="Partial Success",
                border_style="yellow"
            ))
            return 1


if __name__ == "__main__":
    sys.exit(main())

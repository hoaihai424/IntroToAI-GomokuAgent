#!/usr/bin/env python3
"""
Launch the dual match visualization GUI.
Shows Sklearn Agent vs Random and Minimax Agent vs Random side-by-side.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.dual_match_window import main

if __name__ == "__main__":
    print("="*70)
    print("GOMOKU DUAL MATCH VISUALIZATION")
    print("="*70)
    print("\nStarting dual match view...")
    print("  Left:  Sklearn Agent vs Random Agent")
    print("  Right: Minimax Agent vs Random Agent")
    print("\nControls:")
    print("  SPACE - Make one move in each game")
    print("  A     - Toggle auto-play mode")
    print("  P     - Pause/Resume")
    print("  R     - Start new games (after both finish)")
    print("  ESC   - Quit")
    print("\n" + "="*70 + "\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

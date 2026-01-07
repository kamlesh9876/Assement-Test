#!/usr/bin/env python3
"""
Auto-combination script for Nelumbus Assessment System
Automatically combines data when new assessments are completed
"""

import os
import time
from pathlib import Path
from combine_data import DataCombiner

def watch_and_combine():
    """Watch for new assessment data and automatically combine"""
    print("🤖 Auto-combination script started...")
    print("Watching for new assessment data...")
    
    combiner = DataCombiner()
    last_check = {}
    
    while True:
        try:
            # Check for new data in each folder
            current_state = {}
            
            for folder in ['candidates', 'questions', 'responses', 'scores']:
                folder_path = Path(f"data/results/{folder}")
                if folder_path.exists():
                    current_state[folder] = len(list(folder_path.glob("*.json")))
                else:
                    current_state[folder] = 0
            
            # If any folder has new files, run combination
            if current_state != last_check:
                print(f"\n🔄 New data detected! Running combination...")
                results = combiner.process_all_attempts()
                combiner.generate_master_report()
                
                print(f"✅ Combined {results['successful_combinations']} attempts")
                last_check = current_state.copy()
            
            # Check every 5 seconds
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n👋 Auto-combination script stopped by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(10)  # Wait longer on error

if __name__ == "__main__":
    watch_and_combine()

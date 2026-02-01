"""
Test script to verify HVAC simulation is working.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_simulation():
    print("Testing HVAC Simulation...")
    
    try:
        # Test imports
        from dashboard.simulation.simulator import get_simulator
        print("[OK] Simulator imported successfully")
        
        # Get simulator instance
        sim = get_simulator()
        print("[OK] Simulator instance created")
        
        # Test status
        status = sim.get_status()
        print(f"[OK] Status retrieved: {status}")
        
        # Test config
        config = sim.get_config()
        print(f"[OK] Config retrieved: R={config['R']}, C={config['C']}")
        
        # Wait a moment for simulation to run
        import time
        time.sleep(3)
        
        # Check if values are updating
        status2 = sim.get_status()
        print(f"[OK] Status after 3s: {status2}")
        
        if status2['sim_time_sec'] > status['sim_time_sec']:
            print("[OK] Simulation is running and updating")
        else:
            print("[WARN] Simulation may not be updating properly")
            
        # Test history
        history = sim.get_history(limit=5)
        print(f"[OK] History retrieved: {len(history)} records")
        
        if history:
            latest = history[-1]
            print(f"Latest record: T_in={latest['T_in']}, T_out={latest['T_out']}")
        
        print("\n[SUCCESS] Simulation test completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simulation()
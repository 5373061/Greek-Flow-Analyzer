# temp_import_test.py
try:
    from Greek_Energy_FlowII import BlackScholesModel
    print("SUCCESS: Imported BlackScholesModel from Greek_Energy_FlowII.py")
except ImportError as e:
    print(f"FAILED: Could not import BlackScholesModel from Greek_Energy_FlowII.py")
    print(f"Error details: {e}")
except Exception as e_other:
    print(f"FAILED: An unexpected error occurred during import attempt.")
    print(f"Error details: {e_other}")

import shutil, os, sys, textwrap, importlib, traceback, glob

ROOT = os.path.abspath(os.path.dirname(__file__))
PKG  = os.path.join(ROOT, "greek_flow")

def ensure_pkg():
    if os.path.exists(PKG):
        shutil.rmtree(PKG)
    os.makedirs(PKG, exist_ok=True)

    # Find all Python files in the project, including subdirectories
    possible_flow_files = []
    for root, dirs, files in os.walk(ROOT):
        for file in files:
            if file.endswith('.py'):
                possible_flow_files.append(os.path.join(root, file))
    
    print("Available Python files:")
    for i, file in enumerate(possible_flow_files):
        rel_path = os.path.relpath(file, ROOT)
        print(f"{i+1}. {rel_path}")
    
    flow_choice = input("Enter the number of the file containing GreekEnergyFlow class: ")
    try:
        src_flow = possible_flow_files[int(flow_choice) - 1]
    except (ValueError, IndexError):
        sys.exit("❌ Invalid selection")
    
    # Find or create interp.py
    src_interp = os.path.join(ROOT, "interp.py")
    if not os.path.isfile(src_interp):
        print("interp.py not found. Creating a minimal version.")
        with open(src_interp, "w") as f:
            f.write("def _interpolate_greek_at_price(exposure_list, price, greek_name):\n    return 0\n")

    print(f"Copying {os.path.basename(src_flow)} to {os.path.join(PKG, 'flow.py')}")
    shutil.copy(src_flow, os.path.join(PKG, "flow.py"))
    print(f"Copying {os.path.basename(src_interp)} to {os.path.join(PKG, 'interp.py')}")
    shutil.copy(src_interp, os.path.join(PKG, "interp.py"))

    # write simple __init__.py
    init_code = textwrap.dedent("""\
        from .flow   import GreekEnergyFlow
        from .interp import _interpolate_greek_at_price
        __all__ = ["GreekEnergyFlow", "_interpolate_greek_at_price"]
    """)
    with open(os.path.join(PKG, "__init__.py"), "w", encoding="utf8") as f:
        f.write(init_code)

def smoke_test():
    sys.modules.pop("greek_flow", None)
    import greek_flow
    from greek_flow import GreekEnergyFlow
    print("✅ Package repaired. Imported class:", GreekEnergyFlow)

if __name__ == "__main__":
    try:
        ensure_pkg()
        smoke_test()
    except Exception:
        traceback.print_exc()
        sys.exit("❌ Repair failed – see traceback above.")




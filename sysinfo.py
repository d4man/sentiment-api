# sysinfo.py
import platform
import psutil
import cpuinfo

def bytes2gb(n): 
    return f"{n / (1024**3):.2f} GB"

def main():
    print("=== OS ===")
    print("System:", platform.system(), platform.release())
    print("Machine:", platform.machine())
    print("Processor:", platform.processor())
    print()
    print("=== CPU ===")
    info = cpuinfo.get_cpu_info()
    print("Brand:", info.get("brand_raw"))
    print("Cores (physical):", psutil.cpu_count(logical=False))
    print("Cores (logical):", psutil.cpu_count(logical=True))
    print("Max frequency:", f"{psutil.cpu_freq().max:.2f} MHz")
    print()
    print("=== Memory ===")
    vm = psutil.virtual_memory()
    print("Total:", bytes2gb(vm.total))
    print("Available:", bytes2gb(vm.available))
    print()
    print("=== Python ===")
    print("Version:", platform.python_version())

if __name__ == "__main__":
    main()

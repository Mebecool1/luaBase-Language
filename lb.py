
"""
LB - Custom Language Virtual Machine
VM that executes bytecode compiled by lbc
"""

import sys
import json
import ctypes
import ctypes.util
import os
from typing import Any, Dict, List, Optional

# Define raylib struct types using ctypes
class Vector2(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]

class Vector3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]

class Vector4(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float), ("w", ctypes.c_float)]

class Color(ctypes.Structure):
    _fields_ = [("r", ctypes.c_ubyte), ("g", ctypes.c_ubyte), ("b", ctypes.c_ubyte), ("a", ctypes.c_ubyte)]

class Rectangle(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("width", ctypes.c_float), ("height", ctypes.c_float)]

class Matrix(ctypes.Structure):
    _fields_ = [
        ("m0", ctypes.c_float), ("m4", ctypes.c_float), ("m8", ctypes.c_float), ("m12", ctypes.c_float),
        ("m1", ctypes.c_float), ("m5", ctypes.c_float), ("m9", ctypes.c_float), ("m13", ctypes.c_float),
        ("m2", ctypes.c_float), ("m6", ctypes.c_float), ("m10", ctypes.c_float), ("m14", ctypes.c_float),
        ("m3", ctypes.c_float), ("m7", ctypes.c_float), ("m11", ctypes.c_float), ("m15", ctypes.c_float),
    ]

class Quaternion(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float), ("w", ctypes.c_float)]

class Transform(ctypes.Structure):
    _fields_ = [("translation", Vector3), ("rotation", Quaternion), ("scale", Vector3)]

class Camera3D(ctypes.Structure):
    _fields_ = [
        ("position", Vector3),
        ("target", Vector3),
        ("up", Vector3),
        ("fovy", ctypes.c_float),
        ("projection", ctypes.c_int),
    ]

class Camera2D(ctypes.Structure):
    _fields_ = [
        ("offset", Vector2),
        ("target", Vector2),
        ("rotation", ctypes.c_float),
        ("zoom", ctypes.c_float),
    ]

# Map struct names to ctypes classes
RAYLIB_STRUCTS = {
    "Vector2": Vector2,
    "Vector3": Vector3,
    "Vector4": Vector4,
    "Color": Color,
    "Rectangle": Rectangle,
    "Matrix": Matrix,
    "Quaternion": Quaternion,
    "Transform": Transform,
    "Camera3D": Camera3D,
    "Camera2D": Camera2D,
}

class LibraryScanner:
    """Scans system for C libraries and provides ctypes bindings"""
    
    def __init__(self):
        self.loaded_libs: Dict[str, ctypes.CDLL] = {}
        self.platform_extensions = self._get_platform_extensions()
        self.search_paths = self._get_search_paths()
    
    def _get_platform_extensions(self) -> List[str]:
        """Get library extensions for current platform"""
        if sys.platform == "win32":
            return [".dll"]
        elif sys.platform == "darwin":
            return [".dylib", ".so"]
        else:  # Linux and others
            return [".so"]
    
    def _get_search_paths(self) -> List[str]:
        """Get common library search paths for current platform"""
        paths = [
            ".",  # Current directory
            "./lib",
            "./libs",
        ]
        
        if sys.platform == "win32":
            paths.extend([
                os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "System32"),
                os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "SysWOW64"),
            ])
            
            # Add MSYS2 MinGW64 library paths (if they exist)
            msys2_paths = [
                "C:\\msys64\\mingw64\\lib",
                "C:\\msys64\\mingw64\\bin",
                "C:\\msys64\\ucrt64\\lib",
                "C:\\msys64\\ucrt64\\bin",
                "C:\\mingw64\\lib",
                "C:\\mingw64\\bin",
            ]
            
            # Add paths that actually exist
            for path in msys2_paths:
                if os.path.exists(path):
                    paths.append(path)
            
        elif sys.platform == "darwin":
            paths.extend([
                "/usr/local/lib",
                "/usr/lib",
                os.path.expanduser("~/lib"),
            ])
        else:  # Linux
            paths.extend([
                "/usr/local/lib",
                "/usr/lib",
                "/lib",
                os.path.expanduser("~/lib"),
            ])
            # Add architecture-specific paths
            if os.path.exists("/usr/lib/x86_64-linux-gnu"):
                paths.append("/usr/lib/x86_64-linux-gnu")
        
        return paths
    
    def find_library_path(self, header_file: str) -> Optional[str]:
        """Find library file path from header file name"""
        # Extract library name from header file
        # e.g., "raylib.h" -> "raylib"
        lib_name = os.path.splitext(header_file)[0]
        
        # Special handling for math library
        if lib_name == "math":
            if sys.platform == "win32":
                # On Windows, math functions are in msvcrt
                path = ctypes.util.find_library("msvcrt")
                if path:
                    return path
                # Try direct load
                try:
                    test_lib = ctypes.CDLL("msvcrt")
                    return "msvcrt"
                except:
                    pass
            else:
                # On Linux/macOS, try libm
                path = ctypes.util.find_library("m")
                if path:
                    return path
                try:
                    test_lib = ctypes.CDLL("libm.so.6")
                    return "libm.so.6"
                except:
                    pass
        
        # Try to find using ctypes.util first
        lib_path = ctypes.util.find_library(lib_name)
        if lib_path:
            return lib_path
        
        # Manual search in common paths
        for path in self.search_paths:
            for ext in self.platform_extensions:
                full_path = os.path.join(path, lib_name + ext)
                if os.path.exists(full_path):
                    return full_path
                # Try with "lib" prefix (common on Unix)
                full_path = os.path.join(path, "lib" + lib_name + ext)
                if os.path.exists(full_path):
                    return full_path
        
        # Last resort: try direct library name (system-level loading)
        try:
            test_lib = ctypes.CDLL(lib_name)
            return lib_name
        except:
            pass
        
        return None
    
    def load_library(self, header_file: str) -> Optional[ctypes.CDLL]:
        """Load library from header file name"""
        lib_name = os.path.splitext(header_file)[0]
        
        # Check if already loaded
        if lib_name in self.loaded_libs:
            return self.loaded_libs[lib_name]
        
        # Find and load library
        lib_path = self.find_library_path(header_file)
        if not lib_path:
            raise RuntimeError(f"Could not locate library for {header_file}")
        
        try:
            lib = ctypes.CDLL(lib_path)
            self.loaded_libs[lib_name] = lib
            return lib
        except Exception as e:
            raise RuntimeError(f"Failed to load library {lib_path}: {e}")
    
    def get_exported_symbols(self, header_file: str) -> List[str]:
        """Get list of exported function symbols from a library"""
        lib_name = os.path.splitext(header_file)[0]
        
        try:
            lib = self.load_library(header_file)
        except Exception as e:
            raise RuntimeError(f"Cannot get symbols from {header_file}: {e}")
        
        symbols = []
        
        # Try to get symbols using platform-specific methods
        if sys.platform == "win32":
            # On Windows, we can't easily get exported symbols without platform-specific tools
            # So we return empty and rely on error messages when calling
            pass
        elif sys.platform == "darwin" or sys.platform.startswith("linux"):
            # On Unix-like systems, we can try using ctypes to enumerate symbols
            # but this is platform-dependent. For now, we return empty.
            # Users should explicitly import functions they need
            pass
        
        # Return empty list - users should use explicit imports
        # For wildcard imports, we'll handle function resolution at call time
        return symbols
    
    def get_function_if_exists(self, lib: ctypes.CDLL, func_name: str) -> Optional[any]:
        """Try to get a function from a library, return None if not found"""
        try:
            # Try using getattr
            func = getattr(lib, func_name, None)
            return func
        except:
            return None

class VirtualMachine:
    def __init__(self, bytecode_file: str):
        self.bytecode_file = bytecode_file
        self.stack = []
        self.variables = {}  # Global variables
        self.local_vars_stack = []  # Stack of local variable scopes
        self.return_stack = []
        self.pc = 0  # Program counter
        self.instructions = []
        self.labels = {}
        self.call_stack = []
        self.running = True
        self.current_scope = self.variables  # Current scope (global or local)
        self.lib_scanner = LibraryScanner()  # C library scanner
    
    def load_bytecode(self):
        """Load bytecode from file"""
        try:
            with open(self.bytecode_file, 'rb') as f:
                bytecode_str = f.read().decode('utf-8')
            self.instructions = json.loads(bytecode_str)
            
            # Pre-process to find labels
            for i, instr in enumerate(self.instructions):
                if isinstance(instr, (tuple, list)) and len(instr) > 0 and instr[0] == "LABEL":
                    self.labels[instr[1]] = i
        except Exception as e:
            print(f"Error loading bytecode: {e}", file=sys.stderr)
            return False
        return True
    
    def push(self, value: Any):
        """Push value onto stack"""
        self.stack.append(value)
    
    def pop(self) -> Any:
        """Pop value from stack"""
        if not self.stack:
            raise RuntimeError("Stack underflow")
        return self.stack.pop()
    
    def peek(self) -> Any:
        """Peek at top of stack"""
        if not self.stack:
            raise RuntimeError("Stack underflow")
        return self.stack[-1]
    
    def execute_instruction(self, instr):
        """Execute a single instruction"""
        if isinstance(instr, str):
            # Simple string instructions
            if instr == "HALT":
                self.running = False
            elif instr == "PRINT":
                value = self.pop()
                # Print without newline
                print(value, end='')
            elif instr == "RETURN":
                # Pop local scope if in function
                if self.local_vars_stack:
                    self.local_vars_stack.pop()
                
                if self.return_stack:
                    self.pc = self.return_stack.pop()
                else:
                    self.running = False
            else:
                raise RuntimeError(f"Unknown instruction: {instr}")
        
        elif isinstance(instr, (tuple, list)):
            op = instr[0]
            
            if op == "PUSH":
                self.push(instr[1])
            
            elif op == "POP":
                self.pop()
            
            elif op == "LOAD":
                var_name = instr[1]
                # Check local scope first, then global
                if self.local_vars_stack and var_name in self.local_vars_stack[-1]:
                    self.push(self.local_vars_stack[-1][var_name])
                elif var_name in self.variables:
                    self.push(self.variables[var_name])
                else:
                    raise RuntimeError(f"Undefined variable: {var_name}")
            
            elif op == "STORE":
                var_name = instr[1]
                value = self.pop()
                # Store in local scope if in function, else global
                if self.local_vars_stack:
                    self.local_vars_stack[-1][var_name] = value
                else:
                    self.variables[var_name] = value
            
            elif op == "LABEL":
                # Labels are pre-processed, just skip
                pass
            
            elif op == "JMP":
                label = instr[1]
                if label in self.labels:
                    self.pc = self.labels[label] - 1  # -1 because pc will be incremented
                else:
                    raise RuntimeError(f"Label not found: {label}")
            
            elif op == "JIF":
                label = instr[1]
                condition = self.pop()
                if not self.is_truthy(condition):
                    if label in self.labels:
                        self.pc = self.labels[label] - 1
                    else:
                        raise RuntimeError(f"Label not found: {label}")
            
            elif op == "INPUT":
                var_name = instr[1]
                try:
                    user_input = input()
                    # Try to convert to number if possible
                    if user_input.isdigit():
                        self.push(int(user_input))
                    elif '.' in user_input:
                        try:
                            self.push(float(user_input))
                        except:
                            self.push(user_input)
                    else:
                        self.push(user_input)
                except EOFError:
                    self.push("")
            
            elif op == "CALL":
                func_name = instr[1]
                num_args = instr[2] if len(instr) > 2 else 0
                
                # Look for function label
                func_label = f"_FUNC_{func_name}"
                if func_label in self.labels:
                    # Create new local scope
                    local_scope = {}
                    self.local_vars_stack.append(local_scope)
                    # Push return address
                    self.return_stack.append(self.pc)
                    # Jump to function
                    self.pc = self.labels[func_label] - 1
                else:
                    raise RuntimeError(f"Function not found: {func_name}")
            
            elif op == "EXTERN_CALL":
                func_name = instr[1]
                header_file = instr[2]
                num_args = instr[3] if len(instr) > 3 else 0
                
                # Load library
                try:
                    lib = self.lib_scanner.load_library(header_file)
                except Exception as e:
                    raise RuntimeError(f"Failed to load library for {header_file}: {e}")
                
                # Get C function
                try:
                    c_func = getattr(lib, func_name)
                except AttributeError:
                    raise RuntimeError(f"Function '{func_name}' not found in {header_file}")
                
                # Pop arguments from stack (they're in reverse order)
                args = []
                for _ in range(num_args):
                    args.append(self.pop())
                args.reverse()
                
                # Call C function
                try:
                    result = c_func(*args)
                    # Push result back onto stack
                    self.push(result)
                except Exception as e:
                    raise RuntimeError(f"Error calling {func_name}: {e}")
            
            elif op == "EXTERN_WILDCARD_CALL":
                func_name = instr[1]
                header_files = instr[2]  # List of header files to search
                num_args = instr[3] if len(instr) > 3 else 0
                
                # Pop arguments first (before searching for function)
                args = []
                for _ in range(num_args):
                    args.append(self.pop())
                args.reverse()
                
                # Search for function in wildcard-imported libraries
                c_func = None
                found_lib = None
                
                for header_file in header_files:
                    try:
                        lib = self.lib_scanner.load_library(header_file)
                        func = self.lib_scanner.get_function_if_exists(lib, func_name)
                        if func is not None:
                            c_func = func
                            found_lib = header_file
                            break
                    except:
                        continue
                
                if c_func is None:
                    raise RuntimeError(f"Function '{func_name}' not found in any wildcard-imported library")
                
                # Call C function
                try:
                    result = c_func(*args)
                    # Push result back onto stack
                    self.push(result)
                except Exception as e:
                    raise RuntimeError(f"Error calling {func_name}: {e}")
            
            elif op == "STRUCT_CREATE":
                struct_name = instr[1]
                num_args = instr[2] if len(instr) > 2 else 0
                
                if struct_name not in RAYLIB_STRUCTS:
                    raise RuntimeError(f"Unknown struct type: {struct_name}")
                
                # Pop arguments (in reverse order)
                args = []
                for _ in range(num_args):
                    args.append(self.pop())
                args.reverse()
                
                # Create struct instance
                struct_class = RAYLIB_STRUCTS[struct_name]
                try:
                    if args:
                        # If arguments provided, assign them to fields
                        struct_instance = struct_class()
                        fields = [f[0] for f in struct_class._fields_]
                        for i, arg in enumerate(args):
                            if i < len(fields):
                                setattr(struct_instance, fields[i], arg)
                    else:
                        # Create empty struct with default values
                        struct_instance = struct_class()
                    
                    self.push(struct_instance)
                except Exception as e:
                    raise RuntimeError(f"Error creating struct {struct_name}: {e}")
            
            elif op == "STRUCT_GET":
                field_name = instr[1]
                struct_instance = self.pop()
                
                try:
                    value = getattr(struct_instance, field_name)
                    self.push(value)
                except AttributeError:
                    raise RuntimeError(f"Struct has no field '{field_name}'")
            
            elif op == "STRUCT_SET":
                field_name = instr[1]
                value = self.pop()
                struct_instance = self.pop()
                
                try:
                    setattr(struct_instance, field_name, value)
                    self.push(struct_instance)  # Push it back for potential chaining
                except AttributeError:
                    raise RuntimeError(f"Struct has no field '{field_name}'")
            
            # Arithmetic operations (pop 2 values, push result)
            elif op == "+" or op == "ADD":
                right = self.pop()
                left = self.pop()
                if isinstance(left, str) or isinstance(right, str):
                    self.push(str(left) + str(right))
                else:
                    self.push(left + right)
            
            elif op == "-" or op == "SUB":
                right = self.pop()
                left = self.pop()
                self.push(left - right)
            
            elif op == "*" or op == "MUL":
                right = self.pop()
                left = self.pop()
                self.push(left * right)
            
            elif op == "/" or op == "DIV":
                right = self.pop()
                left = self.pop()
                if isinstance(left, str) or isinstance(right, str):
                    raise RuntimeError("Cannot divide strings")
                if right == 0:
                    raise RuntimeError("Division by zero")
                # Integer division if both are ints
                if isinstance(left, int) and isinstance(right, int):
                    self.push(left // right)
                else:
                    self.push(left / right)
            
            elif op == "%" or op == "MOD":
                right = self.pop()
                left = self.pop()
                if right == 0:
                    raise RuntimeError("Modulo by zero")
                self.push(left % right)
            
            # Comparison operations
            elif op == "==":
                right = self.pop()
                left = self.pop()
                self.push(left == right)
            
            elif op == "!=":
                right = self.pop()
                left = self.pop()
                self.push(left != right)
            
            elif op == "<":
                right = self.pop()
                left = self.pop()
                self.push(left < right)
            
            elif op == ">":
                right = self.pop()
                left = self.pop()
                self.push(left > right)
            
            elif op == "<=":
                right = self.pop()
                left = self.pop()
                self.push(left <= right)
            
            elif op == ">=":
                right = self.pop()
                left = self.pop()
                self.push(left >= right)
            
            # Logical operations
            elif op == "and":
                right = self.pop()
                left = self.pop()
                self.push(self.is_truthy(left) and self.is_truthy(right))
            
            elif op == "or":
                right = self.pop()
                left = self.pop()
                self.push(self.is_truthy(left) or self.is_truthy(right))
            
            elif op == "not":
                operand = self.pop()
                self.push(not self.is_truthy(operand))
            
            # Unary plus/minus
            elif op == "+":
                operand = self.pop()
                self.push(+operand)
            
            elif op == "-":
                operand = self.pop()
                self.push(-operand)
            
            else:
                raise RuntimeError(f"Unknown instruction: {op}")
    
    def is_truthy(self, value: Any) -> bool:
        """Determine if a value is truthy"""
        if isinstance(value, bool):
            return value
        elif isinstance(value, (int, float)):
            return value != 0
        elif isinstance(value, str):
            return len(value) > 0
        else:
            return bool(value)
    
    def run(self):
        """Execute the bytecode"""
        if not self.load_bytecode():
            return False
        
        try:
            while self.running and self.pc < len(self.instructions):
                instr = self.instructions[self.pc]
                self.execute_instruction(instr)
                self.pc += 1
            return True
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            return False

def run_bytecode(bytecode_file: str) -> bool:
    """Main execution function"""
    vm = VirtualMachine(bytecode_file)
    return vm.run()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: lb.py <bytecode_file>")
        sys.exit(1)
    
    bytecode_file = sys.argv[1]
    
    try:
        success = run_bytecode(bytecode_file)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

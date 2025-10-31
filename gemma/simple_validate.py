import ast
import sys

files = [
    "src/gemma_cli/config/models.py",
    "src/gemma_cli/config/prompts.py",
    "src/gemma_cli/commands/model.py",
    "src/gemma_cli/cli.py",
]

print("Syntax Validation Results:")
print("-" * 50)

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            ast.parse(fp.read(), filename=f)
        print(f"✅ {f.split('/')[-1]:30s} PASS")
    except SyntaxError as e:
        print(f"❌ {f.split('/')[-1]:30s} FAIL (line {e.lineno})")
    except FileNotFoundError:
        print(f"⚠️ {f.split('/')[-1]:30s} NOT FOUND")
    except Exception as e:
        print(f"⚠️ {f.split('/')[-1]:30s} ERROR: {e}")

print("-" * 50)

# Check dependencies
print("\nDependency Check:")
print("-" * 50)

deps = ["psutil", "tomllib", "pydantic", "rich", "click", "tomli_w"]
missing = []

for dep in deps:
    if dep == "tomllib":
        try:
            import tomllib
            print(f"✅ {dep:15s} Available (built-in)")
        except ImportError:
            try:
                import tomli
                print(f"✅ {dep:15s} Available (via tomli)")
            except ImportError:
                print(f"📦 {dep:15s} MISSING")
                missing.append(dep)
        continue

    try:
        __import__(dep)
        print(f"✅ {dep:15s} Available")
    except ImportError:
        print(f"📦 {dep:15s} MISSING")
        missing.append(dep)

if missing:
    print(f"\n🔧 Install missing: pip install {' '.join(missing)}")

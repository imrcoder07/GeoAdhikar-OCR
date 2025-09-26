import sys
sys.path.insert(0, '.')
import app as app_module

app = app_module.app

if __name__ == "__main__":
    app.run()

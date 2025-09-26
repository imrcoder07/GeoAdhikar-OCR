import sys
sys.path.insert(0, '.')
exec(open('app.py').read(), globals())
app = globals()['app']

if __name__ == "__main__":
    app.run()

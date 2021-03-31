# from waitress import serve
# from app import app

# serve(app, host='0.0.0.0', port=8080)
from app import app

if __name__ == "__main__":
    app.run()

# if __name__ == "__main__":
#     app.run()
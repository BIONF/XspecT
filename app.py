from WebApp import app

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=5000, debug=False, threaded=True)  # Threading makes Multiple Users at once possible
    # host= '0.0.0.0'

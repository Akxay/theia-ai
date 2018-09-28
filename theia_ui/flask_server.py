from flask import render_template
from jinja2 import Environment
import os
import sys
import cv2
import warnings
warnings.filterwarnings('ignore')
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('/Users/home/MSAN/App_Dev_Project/theia/theia_ui/'))))
# from face_detection import *

from flask import Flask, flash, redirect, render_template, request, session, abort, Response, url_for

import psycopg2

from recognition import *


app = Flask(__name__)
ans = 'No matches yet'
flag = 0

def table_ops(conn, schema, table):
    cur = conn.cursor()
    cur.execute("select * from %s.%s ;" % (schema, table))
    rows = cur.fetchall()
    # conn.close()
    print("---done---")
    return rows

def connect_db():
    try:
        conn = psycopg2.connect("dbname='postgres' user='akshayt' host='ec2-54-213-4-220.us-west-2.compute.amazonaws.com' port='5432' password='Theia628'")
    except:
        print("I am unable to connect to the database")
    
    print("--connected---")
    return conn

conn = connect_db()

@app.route('/user_home', methods=['POST'])
def do_admin_login():
    # conn = connect_db()
    table = table_ops(conn, 'msan', 'user')
    for row in table:
        if request.form['password'] == row[4] and request.form['username'] == row[3]:
            session['logged_in'] = True
        else:
            flash('wrong password!')
    """
    if request.form['password'] == 'password' and request.form['username'] == 'admin':
        session['logged_in'] = True
    else:
        flash('wrong password!')
    POST_USERNAME = str(request.form['username'])
    POST_PASSWORD = str(request.form['password'])
 
    Session = sessionmaker(bind=engine)
    s = Session()
    query = s.query(User).filter(User.username.in_([POST_USERNAME]), User.password.in_([POST_PASSWORD]) )
    result = query.first()
    if result:
        session['logged_in'] = True
    else:
        flash('wrong password!')
    """
    return main()

#added this 4/26
@app.route('/register_teacher')
def register_teacher():

    return render_template('register_teacher.html')
#end added this 4/26

@app.route('/registerteacher', methods=['POST'])
def do_admin_register():
    table = table_ops(connect_db(), 'msan', 'user')

    cur = conn.cursor()
    print('hi')
    idx = len(table)

    firstname = request.form['firstname']
    lastname = request.form['lastname']
    username = request.form['username']
    password = request.form['password']
    school = request.form['school']
    active = 1

    cur.execute("""insert into msan.user (tid,firstname,lastname,
        username,password, school, active) values \
        (%s,%s,%s,%s,%s,%s,%s)""", 
        (idx,firstname,lastname,username,password,school,active))
    conn.commit()
    return render_template('registered.html',firstname=firstname)

@app.route("/logout")
def logout():
    session['logged_in'] = False
    return main()


@app.route("/")
def main():
    if not session.get('logged_in'): #added the following lines
        print('------IP-------', request.remote_addr)
        return render_template('login_page.html')
    else:
        return main2()


@app.route("/user_home")
def main2():
    #need a logout button

    # conn = connect_db()
    table = table_ops(conn, 'msan', 'class')
    class_names = [row[2] for row in table]

    if not session.get('logged_in'):
        return render_template('login_page.html')
    elif (session['logged_in'] == True):
        return render_template('user_home.html', class_names = class_names)
    else:
        return render_template('login_page.html')


@app.route("/404.html")
def page_not_found():
    return render_template('404.html')

@app.route("/take_attendance/<name>")
def display_data(name):
    global flag
    flag = 0
    # conn = connect_db()
    table = table_ops(conn, 'msan', 'attendance')
    return render_template('take_attendance.html', table=table, class_names=name)


@app.route("/check_attendance/<name>")
def hi(name):
    table = table_ops(conn, 'msan', 'attendance')
    table_class = table_ops(conn, 'msan', 'class')
    class_names = [row[2] for row in table_class]

    sid = [row[0] for row in table]
    cid = [row[1] for row in table]
    date = [str(row[2].month) + "/"+str(row[2].day) + "/" + str(row[2].year) for row in table]
    time = [str(row[2].time())[:5] for row in table]
    length = len(sid)

    return render_template('check_attendance.html', table=table, class_name = name, sid = sid, cid = cid,
                           date=date, length=length, time=time, class_names=class_names)


@app.route("/get_date/<name>", methods=['POST'])
def date(name):
    # conn = connect_db()
    this_date = request.form['partydate']
    month, day, year = this_date.split("/")

    table = table_ops(conn, 'msan', 'attendance')
    table_class = table_ops(conn, 'msan', 'class')
    class_names = [row[2] for row in table_class]
    sid = []
    cid = []
    date = []
    time = []
    for row in table:
        if (int(row[2].day) == int(day)) and (int(row[2].month) == int(month)) and (int(row[2].year) == int(year)):
            sid.append(row[0])
            cid.append(row[1])
            time.append(str(row[2].time())[:5])

            date.append(str(row[2].month) + "/"+str(row[2].day) + "/" + str(row[2].year))
    length = len(sid)
    return render_template('check_attendance.html', table=table, class_name = name, sid = sid, cid = cid, class_names=class_names,
                           date=date, length=length, time=time)


@app.route('/servercamera.htm')
def servercamera():
    table = table_ops(conn, 'msan', 'class')
    class_names = [row[2] for row in table]
    return render_template('servercamera2.html', class_names = class_names)


def prediction(frame, counter):
    # print(frame.shape)
    if frame is not None :
        if(len(frame)>0):
            # cv2.imwrite("./data/tmp/face.jpg", frame)
            pred, faces_cord = who_is_it(frame)
            # print('------faces_cord----', faces_cord)

            if (isinstance(faces_cord, (list, tuple, np.ndarray)) == True and len(faces_cord)==4):
                x, y, w, h = faces_cord
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                font = cv2.AGAST_FEATURE_DETECTOR_THRESHOLD
                bottomLeftCornerOfText = (x, y-5)
                fontScale = 0.8
                fontColor = (25, 25, 250)
                lineType = 4
                counter = 0

                cv2.putText(frame, pred[1],
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)
                # print ('--1--', frame.shape)
                cv2.waitKey(1)
                return pred, frame, counter

            else :
                counter += 1
                # print ('--2--', frame.shape)

                return pred, frame, counter
    else:
        counter += 1
        return ('', ''), np.random.randint(100, size=(720, 1280, 3)), counter


def gen_pred():
    while True:
        yield render_template('prediction.html', prediction=ans)

def gen(camera):
    global ans
    global flag
    counter = 0
    while counter < 100:
        if flag==1 :
            break
        frame = camera.get_frame()

        # print ('----Predicted Name :  ', ans)
        ans, frame, counter = prediction(frame, counter)

        _, jpeg = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    # temp()
    del camera

def temp():
    return redirect("http://www.example.com", code=302)


@app.route('/video_feed')
def video_feed():
    remote_ip = request.remote_addr
    # print ('------IP-------', remote_ip)
    # remote = 'http://'+str(remote_ip)+':8082'
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pred_res')
def pred_res():
    # print ("--------here in pred_res--------")
    return Response(gen_pred(), mimetype='text/html')

@app.route('/break_cam', methods=['GET'])
def break_cam():
    global flag
    print('##############################')
    flag = 1
    return

@app.route('/register/<name>')
def register(name):
    # conn = connect_db()
    table = table_ops(conn, 'msan', 'class')
    class_names = [row[2] for row in table]

    webcam = Register(cascPath,'./')
    webcam.capture_images(name=name)
    retrain()
    return render_template('justregistered.html', class_names=class_names)


@app.route('/registerclass')
def registerclass():
    # conn = connect_db()
    table = table_ops(conn, 'msan', 'class')
    class_names = [row[2] for row in table]
    return render_template('register_class.html', class_names=class_names)


@app.route('/thanks', methods=['POST'])
def post_new_class_to_db():

    cur = conn.cursor()

    name = request.form["name"]
    cid = request.form["cid"]
    starttime = request.form["starttime"]
    endtime = request.form["endtime"]
    tid = request.form["tid"]

    cur.execute("INSERT INTO msan.class (tid, cid, classname) VALUES (%s, %s, %s)", (tid, cid, name))
    conn.commit()

    return redirect('/user_home')

@app.route('/registerstudent')
def registerstudent():
    # conn = connect_db()
    table = table_ops(conn, 'msan', 'class')
    class_names = [row[2] for row in table]
    return render_template('register_student.html', class_names=class_names)


@app.route('/smile', methods=['POST'])
def post_new_student_to_db():
    # conn = connect_db()
    table = table_ops(conn, 'msan', 'class')
    class_names = [row[2] for row in table]

    cur = conn.cursor()

    firstname = request.form["fname"]
    lastname = request.form["lname"]
    sid = request.form["sid"]

    cur.execute("INSERT INTO msan.students (sid, firstname, lastname) VALUES (%s, %s, %s)", (sid, firstname, lastname))
    conn.commit()

    return render_template('smile.html', name=sid, class_names=class_names)

@app.route('/good_job')
def good_job():

    return redirect('/user_home')


@app.route('/mark_attendance/<reply>', methods=['GET'])
def mark_attendance(reply):
    # data = json.loads(request.data)
    print ('------------inside marker---------------', int(reply))
    global ans
    cur = conn.cursor()
    resp = int(reply)
    # print(ans[1], resp)
    cur.execute("INSERT INTO msan.attendance (sid, cid, dt, res, name) VALUES (%s, %s, CURRENT_TIMESTAMP, %s, %s)", (1, 2, resp, ans[1]))
    conn.commit()
    # data = {'name': ans[1]}
    # resp = jsonify(data)
    # resp.status_code = 200
    return (ans[1])




if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(host='0.0.0.0', port=80, debug=True, threaded=True, use_reloader=False)

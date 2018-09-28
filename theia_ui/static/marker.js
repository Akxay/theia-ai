/**
 * Created by Akshay on 4/24/18.
 */


var buttonMark_yes = document.getElementById("mark_yes");
var buttonMark_no = document.getElementById("mark_no");
var button_register = document.getElementById("register_break");

buttonMark_yes.onclick = function() {
    console.log('Here in marker');
    // var url = window.location.href + "record_status";

    // XMLHttpRequest
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState == 4 && xhr.status == 200) {
            // alert(xhr.responseText);
        }
    };
    var val = this.value;
    xhr.open("GET", "/mark_attendance/"+String(val));
    // xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.send();
    console.log(xhr);
    // console.log('status code: ' + request.status);
};

buttonMark_no.onclick = function() {
    console.log('Here in marker no');
    // var url = window.location.href + "record_status";

    // XMLHttpRequest
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState == 4 && xhr.status == 200) {
            // alert(xhr.responseText);
        }
    };
    var val = this.value;
    xhr.open("GET", "/mark_attendance/"+String(val));
    // xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    console.log(xhr);
    // return xhr.send(JSON.stringify({ res: val }));
};

button_register.onclick = function() {
    console.log('Here in marker inside register break');
    // var url = window.location.href + "record_status";

    // XMLHttpRequest
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState == 4 && xhr.status == 200) {
            // alert(xhr.responseText);
        }
    };
    var val = this.value;
    xhr.open("POST", "/mark_attendance");
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    return xhr.send(JSON.stringify({ res: val }));
};

function break_cam() {
    console.log('---in break_cam marker.js----');
    // var url = window.location.href + "record_status";

    var xhr = new XMLHttpRequest();

    xhr.open("GET", "/break_cam");
    // xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    console.log(xhr);
};

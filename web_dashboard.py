from flask import Flask, request, redirect, render_template_string, url_for
import os, json

app = Flask(__name__, static_folder="/home/wrecker888/static")
user_db_path = "users.json"
alert_style_file = "alert_style.json"
GOOGLE_PLACES_API_KEY = "AIzaSyA3ZwWFnG2rVBpbUOd83MqE4BJtpbBxcS0"

if os.path.exists(user_db_path):
    with open(user_db_path, "r") as f:
        users = json.load(f)
else:
    users = {}

def save_users():
    with open(user_db_path, "w") as f:
        json.dump(users, f, indent=2)

def load_alert_style(username):
    if os.path.exists(alert_style_file):
        with open(alert_style_file, "r") as f:
            data = json.load(f)
        return data.get(username, {"type": "default", "message": ""})
    return {"type": "default", "message": ""}

def save_alert_style(username, style_data):
    name = users.get(username, {}).get("name", "Friend")
    style_data["name"] = name
    with open(alert_style_file, "w") as f:
        json.dump({username: style_data}, f, indent=2)

html_base = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{{ title }}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { background: #f2f2f2; font-family: 'Segoe UI', sans-serif;
           display: flex; justify-content: center; align-items: center;
           height: 100vh; margin: 0; }
    .card { background: white; padding: 2rem; border-radius: 1rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            width: 90%; max-width: 400px; }
    .logo-title { display: flex; align-items: center;
                  justify-content: center; gap: 0.5rem;
                  margin-bottom: 1.5rem; }
    .logo-title img { height: 32px; }
    h2 { margin: 0; color: #333; }
    form { display: flex; flex-direction: column; }
    input { padding: 0.8rem; margin-bottom: 1rem;
            border: 1px solid #ccc; border-radius: 0.5rem;
            font-size: 1rem; }
    button { background: #4CAF50; color: white; padding: 0.8rem;
             font-size: 1rem; border: none; border-radius: 0.5rem;
             cursor: pointer; transition: background 0.3s ease; }
    button:hover { background: #43a047; }
    .switch { text-align: center; margin-top: 1rem; }
    .switch a { color: #4CAF50; text-decoration: none; font-weight: bold; }
    #toast { position: fixed; top: 20px; left: 50%;
             transform: translateX(-50%); padding: 1rem 1.5rem;
             border-radius: 0.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
             font-weight: bold; z-index: 1000; animation: fadeOut 2.5s ease-out forwards; }
    @keyframes fadeOut { 0% { opacity: 1; } 80% { opacity: 1; } 100% { opacity: 0; } }
    ul { padding-left: 1.2rem; }
  </style>
</head>
<body>
  <div class="card">
    <div class="logo-title">
      <img src="{{ logo }}" alt="Logo">
      <h2>{{ title }}</h2>
    </div>
    {{ content | safe }}
  </div>
</body>
</html>
'''

@app.route("/")
def home():
    return redirect("/login")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username in users:
            return "Username already exists.", 400
        users[username] = {"password": password, "name": "", "contacts": []}
        save_users()
        return redirect("/login")

    content = '''
    <form method="POST">
      <input type="text" name="username" placeholder="Username" required />
      <input type="password" name="password" placeholder="Password" required />
      <button type="submit">Sign Up</button>
    </form>
    <div class="switch">Already have an account? <a href="/login">Log in</a></div>
    '''
    return render_template_string(html_base, title="Create an Account",
                                  content=content,
                                  logo=url_for('static', filename='SafeSenseLogo.png'))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = request.args.get("msg", "")
    toast = f'<div id="toast" style="background:#e53935;color:white;">{error}</div>' if error else ""

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username in users and users[username]["password"] == password:
            return redirect(f"/main?user={username}")
        else:
            return redirect("/login?msg=No+account+found+or+wrong+password.")

    content = f'''
    {toast}
    <form method="POST">
      <input type="text" name="username" placeholder="Username" required />
      <input type="password" name="password" placeholder="Password" required />
      <button type="submit">Log In</button>
    </form>
    <div class="switch">New here? <a href="/signup">Sign up</a></div>
    '''
    return render_template_string(html_base, title="SafeSense",
                                  content=content,
                                  logo=url_for('static', filename='SafeSenseLogo.png'))

@app.route("/main")
def main_menu():
    username = request.args.get("user")
    if not username or username not in users:
        return redirect("/login")

    message = request.args.get("msg", "")
    banner = f'<div id="toast" style="background:#4CAF50;color:white;">{message}</div>' if message else ""

    saved_name = users[username].get("name", "")
    greeting = (f'<h2 style="margin:0.5rem 0 1.2rem;">Welcome, {saved_name}!</h2>'
                if saved_name else '<h2 style="margin:0.5rem 0 1.2rem;">Welcome</h2>')

    content = f'''
    <div style="text-align:center;">{greeting}</div>
    {banner}
    <form action="/setname" method="GET"><input type="hidden" name="user" value="{username}" />
      <button type="submit">Set Name</button></form>
    <form action="/reststops" method="GET" style="margin-top:1rem;">
      <input type="hidden" name="user" value="{username}" />
      <button type="submit">Find Nearby Rest Stops</button></form>
    <form action="/alertstyle" method="GET" style="margin-top:1rem;">
      <input type="hidden" name="user" value="{username}" />
      <button type="submit">Set Alert Style</button></form>
    <form action="/contacts" method="GET" style="margin-top:1rem;">
      <input type="hidden" name="user" value="{username}" />
      <button type="submit">Emergency Contacts</button></form>
    <form action="/logout" method="GET" style="margin-top:1rem;">
      <button type="submit" style="background:#e53935;">Sign Out</button></form>
    '''
    return render_template_string(html_base, title="SafeSense",
                                  content=content,
                                  logo=url_for('static', filename='SafeSenseLogo.png'))

@app.route("/setname", methods=["GET", "POST"])
def setname():
    username = request.args.get("user")
    if not username or username not in users:
        return redirect("/login")

    if request.method == "POST":
        name = request.form["name"]
        users[username]["name"] = name
        save_users()
        return redirect(f"/main?user={username}&msg=Name+set!")

    content = '''
    <form method="POST">
      <input type="text" name="name" placeholder="Enter your name" required />
      <button type="submit">Save Name</button>
    </form>
    <form action="/main" method="GET" style="margin-top:1rem;">
      <input type="hidden" name="user" value="{}" />
      <button type="submit">Back to Main</button>
    </form>
    '''.format(username)
    return render_template_string(html_base, title="Set Name of Account",
                                  content=content,
                                  logo=url_for('static', filename='SafeSenseLogo.png'))

@app.route("/reststops")
def reststops():
    username = request.args.get("user")
    if not username or username not in users:
        return redirect("/login")

    content = f'''
    <iframe width="100%" height="400" style="border:0; border-radius:0.5rem; margin-bottom:1.5rem;"
      loading="lazy" allowfullscreen referrerpolicy="no-referrer-when-downgrade"
      src="https://www.google.com/maps/embed/v1/search?key={GOOGLE_PLACES_API_KEY}&q=convenience+stores+near+me">
    </iframe>
    <form action="/main" method="GET">
      <input type="hidden" name="user" value="{username}" />
      <button type="submit">Back to Main</button>
    </form>
    '''
    return render_template_string(html_base, title="Find Nearby Rest Stops",
                                  content=content,
                                  logo=url_for('static', filename='SafeSenseLogo.png'))

@app.route("/alertstyle", methods=["GET", "POST"])
def alertstyle():
    username = request.args.get("user")
    if not username or username not in users:
        return redirect("/login")

    current = load_alert_style(username)

    if request.method == "POST":
        selected = request.form["alert_type"]
        message = request.form.get("custom_message", "")
        save_alert_style(username, {"type": selected, "message": message})
        return redirect(f"/main?user={username}&msg=Alert+style+saved!")

    checked = lambda val: "checked" if current["type"] == val else ""
    shown = 'style="display:block;"' if current["type"] == "tts" else 'style="display:none;"'

    content = f'''
    <form method="POST" style="display:flex; flex-direction:column;">
      <label><input type="radio" name="alert_type" value="default" {checked("default")} onchange="toggleTTS()"> Default Wake-Up Name</label>
      <label><input type="radio" name="alert_type" value="tts" {checked("tts")} onchange="toggleTTS()"> Custom TTS Message</label>
      <div id="tts-field" {shown}><input type="text" name="custom_message" value="{current['message']}" /></div>
      <label><input type="radio" name="alert_type" value="song" {checked("song")} onchange="toggleTTS()"> Annoying Song</label>
      <button type="submit">Save Alert Style</button>
    </form>
    <form action="/main" method="GET" style="margin-top:1rem;">
      <input type="hidden" name="user" value="{username}" />
      <button type="submit">Back to Main</button>
    </form>
    <script>
      function toggleTTS() {{
        const tf = document.getElementById("tts-field");
        tf.style.display = document.querySelector('input[value="tts"]').checked ? "block" : "none";
      }}
      window.onload = toggleTTS;
    </script>
    '''
    return render_template_string(html_base, title="Set Alert Style",
                                  content=content,
                                  logo=url_for('static', filename='SafeSenseLogo.png'))

@app.route("/contacts", methods=["GET", "POST"])
def contacts():
    username = request.args.get("user")
    if not username or username not in users:
        return redirect("/login")

    users[username].setdefault("contacts", [])
    error = request.args.get("error", "")
    toast = f'<div id="toast" style="background:#e53935;color:white;">{error}</div>' if error else ""

    if request.method == "POST":
        contact = request.form["contact"].strip()
        if contact.isdigit() and len(contact) == 10:
            users[username]["contacts"].append(contact)
            save_users()
            return redirect(f"/contacts?user={username}")
        else:
            return redirect(f"/contacts?user={username}&error=Invalid+phone+number.+Must+be+10+digits.")

    existing = users[username]["contacts"]
    list_html = "<ul>" + "".join(f"<li>{c}</li>" for c in existing) + "</ul>" if existing else "<p>No contacts yet.</p>"

    content = f'''
    {toast}
    <h3>Your Emergency Contacts</h3>
    {list_html}
    <form method="POST">
      <input type="text" name="contact" placeholder="Enter 10-digit phone number" required />
      <button type="submit">Add Contact</button>
    </form>
    <form action="/main" method="GET" style="margin-top:1rem;">
      <input type="hidden" name="user" value="{username}" />
      <button type="submit">Back to Main</button>
    </form>
    '''
    return render_template_string(html_base, title="Emergency Contacts",
                                  content=content,
                                  logo=url_for('static', filename='SafeSenseLogo.png'))

@app.route("/logout")
def logout():
    return redirect("/login")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

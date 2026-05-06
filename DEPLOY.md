# Deploy

## Recommended: Railway

1. Create a new Railway project.
2. Add a MySQL database service.
3. Deploy this app folder as the web service.
4. Set these environment variables on the web service:

```txt
MYSQLHOST=<from Railway MySQL>
MYSQLPORT=<from Railway MySQL>
MYSQLUSER=<from Railway MySQL>
MYSQLPASSWORD=<from Railway MySQL>
MYSQLDATABASE=<from Railway MySQL>
SECRET_KEY=<any long random text>
ADMIN_EMAIL=admin
ADMIN_PASSWORD=admin
```

Railway should detect the `Procfile` and run:

```txt
gunicorn app:app
```

The app creates the `users` and `predictions` tables automatically when it connects.

## Local

```txt
python app.py
```

Open:

```txt
http://localhost:5000
```

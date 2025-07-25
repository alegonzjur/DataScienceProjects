#NoEnv
SendMode Input
SetWorkingDir %A_ScriptDir%

EnvName := "ML"
EnvScript := "check_env.bat"
PythonScript := "app.py"

; 1. Comprobar/crear el entorno
RunWait, %ComSpec% /c call "%EnvScript%", , Hide

; 2. Iniciar Flask app en segundo plano y guardar su PID
Run, %ComSpec% /c "conda activate %EnvName% && python %PythonScript%", , Hide, flaskPID

; 3. Esperar un momento para que Flask arranque
Sleep, 4000

; 4. Abrir el navegador predeterminado y guardar su PID
Run, http://127.0.0.1:5000, , , browserPID

; 5. Esperar a que el navegador se cierre
Process, WaitClose, %browserPID%

; 6. Cerrar Flask (si aún está activo)
Process, Close, %flaskPID%
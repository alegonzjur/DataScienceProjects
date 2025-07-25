#NoEnv
SendMode Input
SetWorkingDir %A_ScriptDir%

EnvName := "ML"
EnvScript := "check_env.bat"
PythonScript := "app.py"

; 1. Comprobar/crear el entorno (consola visible)
RunWait, %ComSpec% /k call "%EnvScript%"

; 2. Ejecutar Flask en consola visible (para depuración)
Run, %ComSpec% /k "conda activate %EnvName% && python %PythonScript%"

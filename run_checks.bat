@echo off
echo Running Black formatter...
black . || goto :error

echo.
echo Sorting imports with isort...
isort . || goto :error

echo.
echo Running Flake8 linter...
flake8 . || goto :error

echo.
echo All checks passed!
goto :EOF

:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel% 

@echo off
echo Running code quality checks...

echo.
echo Running flake8...
flake8 matrix_multiplication.py matrix_multiplication_gui.py test_matrix_multiplication.py

echo.
echo Running black (check only)...
black --check matrix_multiplication.py matrix_multiplication_gui.py test_matrix_multiplication.py

echo.
echo Running pytest...
pytest test_matrix_multiplication.py -v

echo.
echo All checks completed.
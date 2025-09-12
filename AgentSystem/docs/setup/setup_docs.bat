@echo off
echo ========================================
echo AgentSystem Ultimate API Documentation Setup
echo The most powerful AI documentation system ever created
echo ========================================
echo.

echo [1/8] Creating documentation directories...
md ..\ultimate 2>nul
md ..\sdks 2>nul
md ..\sdks\python 2>nul
md ..\sdks\javascript 2>nul
md ..\sdks\go 2>nul
md ..\sdks\java 2>nul
md ..\sdks\csharp 2>nul
md ..\examples 2>nul
md ..\postman 2>nul

echo [2/8] Installing documentation dependencies...
pip install pyyaml swagger-ui-bundle openapi-generator-cli 2>nul

echo [3/8] Generating OpenAPI specification...
python ..\api_generator.py

echo [4/8] Generating multi-language SDKs...
echo Generating Python SDK...
python generate_python_sdk.py

echo Generating JavaScript SDK...
python generate_js_sdk.py

echo Generating Go SDK...
python generate_go_sdk.py

echo [5/8] Creating interactive documentation...
echo Generating Swagger UI...
copy swagger-template.html ..\ultimate\index.html

echo [6/8] Generating code examples...
python generate_examples.py

echo [7/8] Creating Postman collections...
python generate_postman.py

echo [8/8] Setting up development server...
echo Starting documentation server on http://localhost:8080
cd ..\ultimate
python -m http.server 8080

echo.
echo ========================================
echo SUCCESS! AgentSystem Ultimate API Documentation Ready!
echo ========================================
echo.
echo Access your documentation at:
echo - Interactive Docs: http://localhost:8080
echo - API Spec: ultimate_api.json
echo - Python SDK: sdks/python/agentsystem.py
echo - JavaScript SDK: sdks/javascript/agentsystem.js
echo - Examples: examples/
echo - Postman: postman/AgentSystem.postman_collection.json
echo.
pause

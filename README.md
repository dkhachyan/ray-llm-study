python deploy_serve_apps.py --serve-config config.yaml --ray-dashboard http://127.0.0.1:8265 --delete-all-apps-before-upload

curl -X POST http://localhost:8000/llama -H "Content-Type: application/json"   -d '{ "prompt": "hi. who are you" }'
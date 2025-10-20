python deploy_serve_apps.py --serve-config config.yaml --ray-dashboard http://127.0.0.1:8265 --delete-all-apps-before-upload

curl -X POST http://localhost:8000/llama -H "Content-Type: application/json"   -d '{ "prompt": "hi. who are you" }'

curl -X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{ "model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "What is your model name?"}] }' && \

curl -X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{ "model": "unsloth/Meta-Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "What is your model name?"}] }'
import requests, json

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2Y2E5MGI0ZC05MjkxLTRiZWItYTVmNy0xNjViYzQ3M2RiYjgiLCJleHAiOjE3NzgzMTMzMzR9.MgX-Gn-YV-83VGCeFX_ExbjPDvkJJnJ1M5kqK8o0Y6Y"

resp = requests.post(
    "http://localhost:8000/api/v1/chat",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json={"message": "我最近在减脂，想知道有什么适合减脂的早餐推荐？"},
    timeout=60,
)
print("status:", resp.status_code)
try:
    data = resp.json()
    print(json.dumps(data, ensure_ascii=False, indent=2))
except Exception:
    print(resp.text)

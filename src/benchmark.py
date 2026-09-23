import statistics, time, httpx

texts = ["My card has not arrived yet", "I was charged twice",
         "How do I top up my account?", "Why was my transfer declined?"]
with httpx.Client(base_url="http://localhost:8000") as c:
    for t in texts[:2]:
        c.post("/predict", json={"text": t})
    lat = []
    for i in range(200):
        r = c.post("/predict", json={"text": texts[i % len(texts)]})
        lat.append(r.json()["latency_ms"])

lat.sort()
print(f"n={len(lat)} p50={statistics.median(lat):.1f}ms "
      f"p95={lat[int(.95*len(lat))]:.1f}ms p99={lat[int(.99*len(lat))]:.1f}ms")

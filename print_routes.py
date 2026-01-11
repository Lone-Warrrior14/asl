from app import app
print("URL map:")
print(app.url_map)
for rule in app.url_map.iter_rules():
    print(f"{rule} -> {rule.endpoint}")
